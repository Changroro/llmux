"""Regression tests for review round 2 (Codex findings D, E, G, H, I)."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from tui.backends.vllm import backend_runtime as rt
from tui.common import exit_codes, mem, profile_store
from tui.common.docker import GpuInfo


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
    monkeypatch.setattr(rt, "exposure_warning", AsyncMock(return_value=""))
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


# H — mem-estimate follows the real TP/PP/DP layout
def _gpu(index: str) -> GpuInfo:
    return GpuInfo(index=index, name="g", memory_used="0", memory_total="16384", utilization="0", temperature="0")


@pytest.mark.parametrize(
    ("gpu_id", "tp", "extra", "used", "split"),
    [
        ("0,1", 1, {}, ["0"], 1),
        ("1,0", 2, {}, ["1", "0"], 2),
        ("0,1,2,3", 2, {"pipeline-parallel-size": 2}, ["0", "1", "2", "3"], 4),
        ("0,1", 1, {"data-parallel-size": 2}, ["0", "1"], 1),
    ],
)
def test_vllm_estimate_uses_parallel_layout(monkeypatch, gpu_id, tp, extra, used, split):
    from tui.backends.vllm import backend_storage
    from tui.backends.vllm.backend_common import Config

    cfg = Config(name="c", model="org/m", extra_params=extra)
    monkeypatch.setattr(backend_storage, "load_config", lambda _n: cfg)
    stored = profile_store.StoredProfile(
        name="p", backend="vllm", gpu_id=gpu_id, tensor_parallel_size=tp, config_name="c"
    )
    _model, _options, gpu_ids, _util, got_split = mem.profile_estimate_request(stored)
    assert gpu_ids == used and got_split == split
    per_gpu, rows = mem.gpu_fit(8.0, [_gpu("0"), _gpu("1"), _gpu("2"), _gpu("3")], gpu_ids=gpu_ids, split=got_split)
    assert per_gpu == 8.0 / split and [r["index"] for r in rows] == used


def test_vllm_estimate_rejects_layout_larger_than_gpu_list(monkeypatch):
    from tui.backends.vllm import backend_storage
    from tui.backends.vllm.backend_common import Config

    monkeypatch.setattr(backend_storage, "load_config", lambda _n: Config(name="c", model="org/m"))
    stored = profile_store.StoredProfile(name="p", backend="vllm", gpu_id="0", tensor_parallel_size=2, config_name="c")
    with pytest.raises(ValueError, match="2 GPU"):
        mem.profile_estimate_request(stored)


# I — prepare keys downloads on (repo, revision); weights win over tokenizer-only
def test_prepare_keeps_distinct_revisions_and_prefers_weights():
    config = rt.Config(
        name="c",
        model="org/main",
        extra_params={
            "revision": "r1",
            "tokenizer": "org/main",
            "tokenizer-revision": "r2",
            "speculative-config": {"method": "eagle", "model": "org/draft"},
        },
    )
    assert rt._prepare_repos(config) == [
        ("model", "org/main", "r1"),
        ("tokenizer", "org/main", "r2"),
        ("draft model", "org/draft", ""),
    ]
    shared = rt.Config(
        name="c", model="org/main",
        extra_params={"tokenizer": "org/shared", "speculative-config": {"model": "org/shared"}},
    )
    assert rt._prepare_repos(shared) == [("model", "org/main", ""), ("draft model", "org/shared", "")]


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
    monkeypatch.setattr(rt, "exposure_warning", AsyncMock(return_value=""))
    events = [e async for e in rt.stream_container_up("p", dry_run=True)]
    assert events[-1] == ("rc", 0)
    assert seen == [{"write": False}]


def test_dry_run_render_validates_without_the_script():
    # `up --dry-run` renders in-process (backend_runtime._preview_command); same validation.
    from tui.backends.llamacpp import backend_runtime as rt

    render = rt._renderer().render_command
    assert render({"ctx-size": 4096}, hf_repo="org/X-GGUF", hf_file="x.gguf")[-2:] == ["--ctx-size", "4096"]
    with pytest.raises(ValueError):
        render({"extra-args": "--port 1"}, hf_repo="org/X-GGUF", hf_file="x.gguf")


# --dry-run must not reach onboarding (.env.common) or the update check (cache/self-update)
def test_dry_run_runs_are_not_interactive(monkeypatch):
    from tui import cli

    monkeypatch.setattr(sys, "argv", ["llmux", "up", "p", "--dry-run"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("LLMUX_NONINTERACTIVE", raising=False)
    assert cli._interactive_session() is False
