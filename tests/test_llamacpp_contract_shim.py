"""llama.cpp honors the shared `up` contract (dry_run, timeouts, --replace precheck)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from tui.backends.llamacpp import backend_runtime as rt
from tui.common import profile_store

_REAL_ENSURE_PROFILE_CONFIG = rt._ensure_profile_config


@pytest.fixture
def llama_profile(monkeypatch, tmp_path):
    gguf = {"hf_repo": "org/m-GGUF", "hf_file": "m-Q4_K_M.gguf"}
    stored = profile_store.StoredProfile(name="lp", backend="llamacpp", container_name="lp", port=8080, gpu_id="0", **gguf)
    profile = rt.Profile(name="lp", container_name="lp", port=8080, image_tag="ghcr.io/ggml-org/llama.cpp:server-cuda", **gguf)
    commands: list = []
    seen: dict = {}

    class Store:
        @staticmethod
        def load_profile(_name, _backend):
            return stored

        @staticmethod
        @contextmanager
        def storage_transaction():
            yield

    async def compose(cmd, **_kwargs):
        commands.append(cmd)
        yield ("rc", 0)

    async def validation(_profile, **kwargs):
        seen.update(kwargs)
        yield ("result", True, [])

    monkeypatch.setattr(rt, "profile_store", Store)
    monkeypatch.setattr(rt, "validate_common_env", lambda _p: (True, []))
    monkeypatch.setattr(rt, "load_profile", lambda _n: profile)
    monkeypatch.setattr(rt, "check_port_conflict", AsyncMock(return_value="profile 'old'"))
    monkeypatch.setattr(rt, "_gpu_conflict_messages", AsyncMock(return_value=[]))
    monkeypatch.setattr(rt, "_ensure_profile_config", lambda *_a, **_k: (True, []))
    monkeypatch.setattr(rt, "_render_profile_snapshot", lambda _n, **_k: (profile, tmp_path / "lp.env"))
    monkeypatch.setattr(rt, "_render_override", AsyncMock(return_value=(0, "")))
    monkeypatch.setattr(rt, "_compose_env", lambda _p: {})
    monkeypatch.setattr(rt, "_stream", compose)
    monkeypatch.setattr(rt, "_post_start_validation", validation)
    monkeypatch.setattr(rt.prepare, "image_present", AsyncMock(return_value=True))
    return commands, seen


@pytest.mark.asyncio
async def test_dry_run_plans_without_port_probe_or_compose(llama_profile):
    commands, _ = llama_profile
    events = [e async for e in rt.stream_container_up("lp", dry_run=True)]
    plans = [data for kind, data in events if kind == "plan"]
    assert events[-1] == ("rc", 0)
    assert plans and plans[0]["image"] == "ghcr.io/ggml-org/llama.cpp:server-cuda"
    assert "up" in plans[0]["command"]
    assert {"-hf", "org/m-GGUF", "-hff", "m-Q4_K_M.gguf"} <= set(plans[0]["server_args"])
    assert commands == []
    rt.check_port_conflict.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeouts_reach_validation(llama_profile, monkeypatch):
    commands, seen = llama_profile
    monkeypatch.setattr(rt, "check_port_conflict", AsyncMock(return_value=None))
    events = [e async for e in rt.stream_container_up("lp", stall_timeout=90.0, ready_timeout=300.0)]
    assert events[-1] == ("rc", 0)
    assert seen["timeout"] == 90.0 and seen["max_wait"] == 300.0


def _cli(monkeypatch, *args):
    from tui.cli import app, container

    monkeypatch.setattr(container, "detect_backend", lambda *a, **k: "llamacpp")
    monkeypatch.setattr(container, "gather_conflict_warnings", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        container.profile_store, "load_profile",
        lambda name, backend: profile_store.StoredProfile(name=name, backend=backend, port=8080),
    )
    return CliRunner().invoke(app, ["up", "lp", *args])


def test_cli_up_dry_run_on_llamacpp(llama_profile, monkeypatch):
    result = _cli(monkeypatch, "--dry-run", "--json")
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["plan"]["image"] == "ghcr.io/ggml-org/llama.cpp:server-cuda"


def test_cli_up_timeout_on_llamacpp(llama_profile, monkeypatch):
    commands, seen = llama_profile
    monkeypatch.setattr(rt, "check_port_conflict", AsyncMock(return_value=None))
    result = _cli(monkeypatch, "--timeout", "60", "--stall-timeout", "30")
    assert result.exit_code == 0, result.stderr
    assert seen["max_wait"] == 60 and seen["timeout"] == 30


def test_cli_up_replace_on_llamacpp(llama_profile, monkeypatch):
    from tui.common import model_switch

    commands, _ = llama_profile
    order: list = []
    old = profile_store.StoredProfile(name="old", backend="vllm", container_name="old", port=8080, gpu_id="0")
    monkeypatch.setattr(model_switch, "plan_switch", AsyncMock(return_value=(object(), [old])))

    async def stop(_target, _sources):
        order.append("stop")
        rt.check_port_conflict.return_value = None

    monkeypatch.setattr(model_switch, "stop_switch_sources", stop)
    real_up = rt.stream_container_up

    def tracking_up(name, **kwargs):
        order.append("check" if kwargs.get("dry_run") else "up")
        return real_up(name, **kwargs)

    monkeypatch.setattr(rt, "stream_container_up", tracking_up)
    result = _cli(monkeypatch, "--replace")
    assert result.exit_code == 0, result.stderr
    assert order == ["check", "stop", "up"]
    assert len(commands) == 1


def _real_config_env(monkeypatch, tmp_path, config_text):
    from tui.backends.llamacpp import backend as lbackend

    stored = profile_store.StoredProfile(name="lp", backend="llamacpp", container_name="lp", port=8080, config_name="lp")
    if config_text is not None:
        (tmp_path / "lp.yaml").write_text(config_text)
    monkeypatch.setattr(rt, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(lbackend, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(rt, "_ensure_profile_config", _REAL_ENSURE_PROFILE_CONFIG)
    return stored


def test_dry_run_still_rejects_an_empty_config(llama_profile, monkeypatch, tmp_path):
    from tui.backends.llamacpp import backend as lbackend

    stored = _real_config_env(monkeypatch, tmp_path, "")
    monkeypatch.setattr(rt.profile_store, "load_profile", lambda *_a: stored, raising=False)
    monkeypatch.setattr(rt, "load_config", lbackend.load_config)
    ok, messages = rt._ensure_profile_config(stored, rt.Profile(name="lp"), dry_run=True)
    assert ok is False and any("비어" in m for m in messages)


def test_dry_run_reports_but_never_creates_a_missing_config(llama_profile, monkeypatch, tmp_path):
    from tui.backends.llamacpp import backend as lbackend

    stored = _real_config_env(monkeypatch, tmp_path, None)
    monkeypatch.setattr(rt.profile_store, "load_profile", lambda *_a: stored, raising=False)
    monkeypatch.setattr(rt, "load_config", lbackend.load_config)
    ok, messages = rt._ensure_profile_config(stored, rt.Profile(name="lp"), dry_run=True)
    assert ok is True and any("Dry run" in m for m in messages)
    assert not (tmp_path / "lp.yaml").exists()


def test_cli_replace_on_llamacpp_with_bad_config_stops_nothing(llama_profile, monkeypatch):
    from tui.common import model_switch

    old = profile_store.StoredProfile(name="old", backend="vllm", container_name="old", port=8080, gpu_id="0")
    monkeypatch.setattr(model_switch, "plan_switch", AsyncMock(return_value=(object(), [old])))
    stop = AsyncMock()
    monkeypatch.setattr(model_switch, "stop_switch_sources", stop)
    monkeypatch.setattr(
        rt, "_ensure_profile_config",
        lambda *_a, **_k: (False, ["✗ config 'lp' 가 비어 있습니다"]),
    )
    result = _cli(monkeypatch, "--replace")
    assert result.exit_code == 1
    stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_dry_run_with_missing_config_still_plans(llama_profile, monkeypatch, tmp_path):
    commands, _ = llama_profile
    monkeypatch.setattr(rt, "CONFIG_DIR", tmp_path)
    render = AsyncMock(return_value=(1, "config 파일 없음"))
    monkeypatch.setattr(rt, "_render_override", render)
    events = [e async for e in rt.stream_container_up("lp", dry_run=True)]
    assert events[-1] == ("rc", 0)
    render.assert_not_awaited()
    assert not list(tmp_path.iterdir())


def test_cli_replace_stops_nothing_when_the_image_cannot_be_pulled(llama_profile, monkeypatch):
    from tui.common import model_switch

    old = profile_store.StoredProfile(name="old", backend="vllm", container_name="old", port=8080, gpu_id="0")
    monkeypatch.setattr(model_switch, "plan_switch", AsyncMock(return_value=(object(), [old])))
    stop = AsyncMock()
    monkeypatch.setattr(model_switch, "stop_switch_sources", stop)
    monkeypatch.setattr(rt.prepare, "image_present", AsyncMock(return_value=False))

    async def failing_pull(_ref):
        yield ("log", "manifest unknown")
        yield ("rc", 1)

    monkeypatch.setattr(rt.prepare, "stream_pull", failing_pull)
    result = _cli(monkeypatch, "--replace")
    assert result.exit_code == 1
    assert "could not pull" in result.stderr
    stop.assert_not_awaited()


def _dry_run_env(monkeypatch, tmp_path, *, config="ctx-size: 1\n", **profile_fields):
    from tui.backends.llamacpp import backend as lbackend

    stored = profile_store.StoredProfile(**{
        "name": "lp", "backend": "llamacpp", "container_name": "lp", "port": 8080, "config_name": "lp",
        "hf_repo": "org/m-GGUF", "hf_file": "m.gguf", **profile_fields,
    })
    (tmp_path / "lp.yaml").write_text(config)
    monkeypatch.setattr(rt, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(lbackend, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(rt, "load_config", lbackend.load_config)
    monkeypatch.setattr(rt.profile_store, "load_profile", lambda *_a: stored, raising=False)
    monkeypatch.setattr(rt.prepare, "hf_cache_path", lambda: str(tmp_path / "hf"))
    monkeypatch.setattr(rt.prepare, "secure_image", _ok_stream)


async def _ok_stream(*_a, **_k):
    yield ("rc", 0)


@pytest.mark.asyncio
async def test_dry_run_without_cached_flag_list_plans_without_server_args(llama_profile, monkeypatch, tmp_path):
    from tui.backends.llamacpp.backend import FlagListNotCached

    _dry_run_env(monkeypatch, tmp_path, config="jinja: false\n")
    extract = AsyncMock(side_effect=FlagListNotCached("flag list not cached"))
    monkeypatch.setattr(rt, "extract_llama_server_flags", extract)
    events = [e async for e in rt.stream_container_up("lp", dry_run=True)]
    assert events[-1] == ("rc", 0)
    plan = next(data for kind, data in events if kind == "plan")
    assert plan["server_args"] is None
    assert extract.await_args.kwargs == {"cache_only": True}


@pytest.mark.asyncio
async def test_replace_precheck_may_read_flags_from_the_image(llama_profile, monkeypatch, tmp_path):
    _dry_run_env(monkeypatch, tmp_path, config="jinja: false\n")
    extract = AsyncMock(return_value={"jinja", "no-jinja"})
    monkeypatch.setattr(rt, "extract_llama_server_flags", extract)
    events = [e async for e in rt.stream_container_up("lp", dry_run=True, stage_image=True)]
    assert events[-1] == ("rc", 0)
    assert "--no-jinja" in next(data for kind, data in events if kind == "plan")["server_args"]
    assert extract.await_args.kwargs == {"cache_only": False}


@pytest.mark.asyncio
async def test_dry_run_plans_an_uncached_lora(llama_profile, monkeypatch, tmp_path):
    _dry_run_env(monkeypatch, tmp_path, enable_lora=True, lora_modules="org/lora/a.gguf")
    download = AsyncMock(side_effect=AssertionError("plain dry-run downloads nothing"))
    monkeypatch.setattr(rt, "_download_loras", download)
    events = [e async for e in rt.stream_container_up("lp", dry_run=True)]
    assert events[-1] == ("rc", 0)
    assert any("would download LoRA org/lora/a.gguf" in data for kind, data in events if kind == "log")
    args = next(data for kind, data in events if kind == "plan")["server_args"]
    assert args[args.index("--lora") + 1].endswith("/a.gguf")


@pytest.mark.asyncio
async def test_replace_precheck_downloads_loras_first(llama_profile, monkeypatch, tmp_path):
    _dry_run_env(monkeypatch, tmp_path, enable_lora=True, lora_modules="org/lora/a.gguf")
    calls = []

    async def failing_download(name, *_a, **_k):
        calls.append(name)
        yield ("log", "download failed")
        yield ("rc", 1)

    monkeypatch.setattr(rt, "_download_loras", failing_download)
    events = [e async for e in rt.stream_container_up("lp", dry_run=True, stage_image=True)]
    assert calls == ["lp"] and events[-1] == ("rc", 1)
    assert not [e for e in events if e[0] == "plan"]


@pytest.mark.asyncio
async def test_dry_run_plan_hides_api_keys(llama_profile, monkeypatch, tmp_path):
    _dry_run_env(monkeypatch, tmp_path, config="api-key: sk-secret\nextra-args: [--hf-token=hf_abc, -hft, hf_def]\n")
    events = [e async for e in rt.stream_container_up("lp", dry_run=True)]
    assert events[-1] == ("rc", 0)
    args = next(data for kind, data in events if kind == "plan")["server_args"]
    assert not {"sk-secret", "hf_def"} & set(args) and not any("hf_abc" in a for a in args)
    assert args[args.index("--api-key") + 1] == "<redacted>" and "--hf-token=<redacted>" in args


@pytest.mark.asyncio
async def test_staged_image_is_not_pulled_again(llama_profile, monkeypatch):
    commands, _ = llama_profile
    monkeypatch.setattr(rt, "check_port_conflict", AsyncMock(return_value=None))
    events = [e async for e in rt.stream_container_up("lp", pull=True, image_staged=True)]
    assert events[-1] == ("rc", 0)
    assert commands[0][-2:] == ["--pull", "never"]


@pytest.mark.asyncio
async def test_replace_precheck_stores_the_resolved_quant(llama_profile, monkeypatch):
    resolve = AsyncMock(return_value="▸ org/m-GGUF:Q8_0 → m-Q8_0.gguf")
    preview = AsyncMock(side_effect=AssertionError("staged precheck resolves for real"))
    monkeypatch.setattr(rt, "_resolve_pending_quant", resolve)
    monkeypatch.setattr(rt, "_preview_quant", preview)
    monkeypatch.setattr(rt.prepare, "secure_image", _ok_stream)
    events = [e async for e in rt.stream_container_up("lp", dry_run=True, stage_image=True)]
    assert events[-1] == ("rc", 0)
    resolve.assert_awaited_once_with("lp")


@pytest.mark.asyncio
async def test_dry_run_applies_the_gguf_migration_in_memory(llama_profile, monkeypatch, tmp_path):
    _dry_run_env(monkeypatch, tmp_path, config="model-file: cfg.gguf\nctx-size: 1\n", hf_file="", model_file="old.gguf")
    profile = rt.Profile(name="lp", config_name="lp", hf_repo="org/m-GGUF", hf_file="")
    args = await rt._preview_command(profile, "img", staged=False)
    assert args[args.index("-hff") + 1] == "cfg.gguf" and "--model-file" not in args
    assert "model-file" in (tmp_path / "lp.yaml").read_text()  # nothing written



@pytest.mark.asyncio
async def test_dry_run_plan_names_the_profile_env_file_up_will_write(llama_profile, monkeypatch, tmp_path):
    from tui.backends.llamacpp import backend as lbackend

    monkeypatch.setattr(lbackend, "RUNTIME_DIR", tmp_path / "runtime")  # no rendered .env yet
    events = [e async for e in rt.stream_container_up("lp", dry_run=True)]
    command = next(data for kind, data in events if kind == "plan")["command"]
    env_file = str(tmp_path / "runtime" / "lp.env")
    assert command[command.index(env_file) - 1] == "--env-file"
