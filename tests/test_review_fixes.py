"""Regression tests for review round 2 (Codex findings D, E, G)."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from tui.backends.vllm import backend_runtime as rt
from tui.common import exit_codes, profile_store


class _Store:
    @staticmethod
    def load_profile(name, backend_name):
        return profile_store.StoredProfile(name=name, backend=backend_name)

    @staticmethod
    @contextmanager
    def storage_transaction():
        yield


# D — registry unreachable while no local image -> 5, not 1
@pytest.mark.asyncio
async def test_unknown_release_without_local_image_is_unreachable(monkeypatch):
    profile = rt.Profile(name="p", container_name="p", port="8000")
    monkeypatch.setattr(rt, "profile_store", _Store)
    monkeypatch.setattr(rt, "load_profile", lambda _n: profile)
    monkeypatch.setattr(rt, "check_port_conflict", AsyncMock(return_value=None))
    monkeypatch.setattr(rt, "_ensure_common_env", lambda _p: (True, []))
    monkeypatch.setattr(rt, "_ensure_profile_config", lambda _p, **_k: (True, []))
    monkeypatch.setattr(rt, "_render_profile_snapshot", lambda _n, **_k: (profile, "/x"))
    monkeypatch.setattr(rt, "_gpu_conflict_messages", AsyncMock(return_value=[]))
    monkeypatch.setattr(rt, "get_local_latest_tag", AsyncMock(return_value="none"))
    monkeypatch.setattr(rt, "get_dockerhub_release_version", AsyncMock(return_value="unknown"))
    events = [e async for e in rt.stream_container_up("p")]
    assert events[-1] == ("rc", exit_codes.UNREACHABLE)


# E — wait never reports success after its deadline
@pytest.mark.asyncio
async def test_wait_does_not_succeed_after_deadline(monkeypatch):
    from tui.common import docker, readiness

    snapshot = docker.ContainerSnapshot(
        name="p", lifecycle=docker.ContainerLifecycle.RUNNING,
        health=docker.ContainerHealth.NONE, raw_state="running", raw_status="running",
    )
    calls = {"n": 0}

    async def probe(*_a, **_k):
        calls["n"] += 1
        return ("ready", ["m"], "") if calls["n"] > 1 else ("loading", [], "")

    monkeypatch.setattr(readiness.docker, "container_snapshots", AsyncMock(return_value={"p": snapshot}))
    monkeypatch.setattr(readiness, "probe_served", probe)
    stored = profile_store.StoredProfile(name="p", backend="vllm", port=8000)
    loop_start = __import__("asyncio").get_running_loop().time()
    rc, _msg, _models = await readiness.wait_ready(stored, timeout=0.02, interval=0.1)
    elapsed = __import__("asyncio").get_running_loop().time() - loop_start
    assert rc == exit_codes.NOT_READY
    assert elapsed < 0.09


# F — readiness lookup errors are not "loading"
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "state"),
    [
        ("refused", "loading"),
        ("503", "loading"),
        ("401", "error"),
        ("keyfile", "error"),
    ],
)
async def test_probe_classifies_failures(monkeypatch, failure, state):
    import urllib.error

    from tui.common import http, readiness

    async def key(*_a):
        if failure == "keyfile":
            raise RuntimeError("could not read api-key-file inside the server container")
        return ""

    async def served(*_a, **_k):
        cause = {
            "refused": ConnectionRefusedError(111, "refused"),
            "503": urllib.error.HTTPError("u", 503, "Loading model", {}, None),
            "401": urllib.error.HTTPError("u", 401, "Unauthorized", {}, None),
        }[failure]
        try:
            raise cause
        except Exception as exc:
            raise RuntimeError("model discovery failed") from exc

    monkeypatch.setattr(http, "profile_api_key", key)
    monkeypatch.setattr(http, "list_served_models", served)
    got, models, detail = await readiness.probe_served("vllm", "p", 8000)
    assert got == state and models == []
    if state == "error":
        assert detail


@pytest.mark.asyncio
async def test_wait_reports_probe_errors_with_their_own_code(monkeypatch):
    from tui.common import docker, readiness

    snapshot = docker.ContainerSnapshot(
        name="p", lifecycle=docker.ContainerLifecycle.RUNNING,
        health=docker.ContainerHealth.NONE, raw_state="running", raw_status="running",
    )
    monkeypatch.setattr(readiness.docker, "container_snapshots", AsyncMock(return_value={"p": snapshot}))
    monkeypatch.setattr(readiness, "probe_served", AsyncMock(return_value=("error", [], "HTTP 401")))
    stored = profile_store.StoredProfile(name="p", backend="vllm", port=8000)
    rc, msg, _ = await readiness.wait_ready(stored, timeout=5, interval=0.01)
    assert rc == exit_codes.ERROR and "401" in msg


# G1 — --json runs are never interactive (no update notice/prompt on stdout)
def test_json_runs_are_not_interactive(monkeypatch):
    from tui import cli

    monkeypatch.setattr(sys, "argv", ["llmux", "ps", "--json"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("LLMUX_NONINTERACTIVE", raising=False)
    assert cli._interactive_session() is False


# G3 — render-env --json partial failure carries a top-level error and stderr lines
def test_render_env_partial_failure_reports_error(monkeypatch):
    from tui.cli import container

    good = profile_store.StoredProfile(name="good", backend="vllm")
    bad = profile_store.StoredProfile(name="bad", backend="vllm")
    monkeypatch.setattr(container.profile_store, "list_profiles", lambda bk: [good, bad] if bk == "vllm" else [])

    def render(name, _bk):
        if name == "bad":
            raise ValueError("bad env value")
        return f"/r/{name}.env"

    monkeypatch.setattr(container.profile_store, "render_env_for_profile", render)
    result = CliRunner().invoke(container.app, ["render-env", "--json"])
    assert result.exit_code == 1
    record = json.loads(result.stdout)
    assert record["ok"] is False and "bad env value" in record["error"]
    assert "bad env value" in result.stderr


# dry-run validates in memory and writes nothing
def test_check_env_validates_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_store, "RUNTIME_DIR", tmp_path, raising=False)
    good = profile_store.StoredProfile(name="p", backend="vllm", env_vars={"FOO": "1"})
    path = profile_store.check_env(good)
    assert path == profile_store.runtime_env_path("p", "vllm")
    assert not path.exists()
    bad = profile_store.StoredProfile(name="p", backend="vllm", env_vars={"CONTAINER_NAME": "x"})
    with pytest.raises(ValueError):
        profile_store.check_env(bad)


@pytest.mark.asyncio
async def test_vllm_dry_run_renders_the_env_in_memory(monkeypatch):
    profile = rt.Profile(name="p", container_name="p", port="8000", image_tag="vllm/vllm-openai:v1")
    seen: list = []

    def snapshot(name, **kwargs):
        seen.append(kwargs)
        return profile, "/x"

    monkeypatch.setattr(rt, "profile_store", _Store)
    monkeypatch.setattr(rt, "load_profile", lambda _n: profile)
    monkeypatch.setattr(rt, "_ensure_common_env", lambda _p: (True, []))
    monkeypatch.setattr(rt, "_ensure_profile_config", lambda _p, **_k: (True, []))
    monkeypatch.setattr(rt, "_render_profile_snapshot", snapshot)
    monkeypatch.setattr(rt, "_gpu_conflict_messages", AsyncMock(return_value=[]))
    events = [e async for e in rt.stream_container_up("p", dry_run=True)]
    assert events[-1] == ("rc", 0)
    assert seen == [{"write": False}]


def test_render_override_check_mode_writes_nothing(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    import yaml

    root = tmp_path
    (root / "compose").mkdir()
    (root / "config" / "llamacpp").mkdir(parents=True)
    (root / "profiles.yaml").write_text(yaml.safe_dump({
        "version": 1,
        "profiles": [{"name": "lp", "backend": "llamacpp", "port": 8080, "gpu_id": "0",
                      "hf_repo": "org/X-GGUF", "hf_file": "x.gguf"}],
    }))
    (root / "config" / "llamacpp" / "lp.yaml").write_text("ctx-size: 4096\nn-gpu-layers: 99\n")
    script = Path(__file__).resolve().parents[1] / "scripts" / "llamacpp" / "render-override.py"
    env = {**os.environ, "LLMUX_ROOT": str(root)}
    ok = subprocess.run([sys.executable, str(script), "--check", "lp"], env=env, capture_output=True, text=True, timeout=30)
    assert ok.returncode == 0, ok.stderr
    # Only the storage lock (taken by every reader, `ps` included) may exist.
    assert [f.name for f in (root / ".runtime").rglob("*") if f.is_file()] in ([], [".profiles.lock"])
    (root / "config" / "llamacpp" / "lp.yaml").write_text("extra-args: --port 1\n")
    bad = subprocess.run([sys.executable, str(script), "--check", "lp"], env=env, capture_output=True, text=True, timeout=30)
    assert bad.returncode == 1 and bad.stderr


# --dry-run must not reach onboarding (.env.common) or the update check (cache/self-update)
def test_dry_run_runs_are_not_interactive(monkeypatch):
    from tui import cli

    monkeypatch.setattr(sys, "argv", ["llmux", "up", "p", "--dry-run"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("LLMUX_NONINTERACTIVE", raising=False)
    assert cli._interactive_session() is False
