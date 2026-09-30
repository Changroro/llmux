from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("module,name", [
    ("tui.backends.vllm.backend_process", "run_command"),
    ("tui.backends.vllm.backend_process", "run_command_with_options"),
    ("tui.backends.vllm.backend_process", "stream_command"),
    ("tui.backends.llamacpp.backend_runtime", "_run"),
    ("tui.backends.llamacpp.backend_runtime", "_stream"),
    ("tui.common.prepare", "_run"),
    ("tui.common.prepare", "stream_lines"),
    ("tui.common.dev_build", "_run"),
    ("tui.common.dev_build", "_stream"),
    ("tui.common.docker", "run_command"),
])
async def test_cancelled_commands_reap_their_child(module, name):
    backend_process = importlib.import_module(module)

    started = asyncio.Event()

    async def blocked(*args):
        started.set()
        await asyncio.Event().wait()

    proc = MagicMock(returncode=None)
    proc.communicate = blocked
    proc.stdout.readline = AsyncMock(return_value=b"ready\n")
    proc.stdout.read = AsyncMock(return_value=b"ready\n")
    proc.wait = AsyncMock()
    with patch.object(asyncio, "create_subprocess_exec", AsyncMock(return_value=proc)):
        if "stream" in name:
            stream = getattr(backend_process, name)(["dummy"])
            await anext(stream)
            await stream.aclose()
        else:
            task = asyncio.create_task(getattr(backend_process, name)("dummy"))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    proc.kill.assert_called_once()
    proc.wait.assert_awaited_once()


def test_onboarding_does_not_replace_existing_invalid_settings(tmp_path):
    from tui.common import onboarding

    path = tmp_path / ".env.common"
    original = "HF_CACHE_PATH=/cache\nHF_TOKEN=keep-me\nCUSTOM_SETTING=keep-too\n"
    path.write_text(original)
    path.chmod(0o644)
    example = tmp_path / "example"
    example.write_text("HF_CACHE_PATH=/cache\nMODEL_DIR=/models\nHF_TOKEN=\n")
    with patch.object(onboarding, "COMMON_ENV", path), patch.object(onboarding, "COMMON_ENV_EXAMPLE", example), patch("rich.prompt.Prompt.ask", side_effect=[str(tmp_path / "cache"), str(tmp_path / "models"), ""]):
        assert onboarding.run_onboarding() is False
    assert path.read_text() == original


def test_recipe_arguments_preserve_equals_and_multiple_values():
    from tui.common.recipes import args_to_config

    assert args_to_config(["--max-model-len=4096", "--served-model-name", "first", "second", "--trust-remote-code"]) == {
        "max-model-len": 4096,
        "served-model-name": ["first", "second"],
        "trust-remote-code": True,
    }


def test_locale_precedence_and_monitor_toggle(monkeypatch):
    from tui.common.i18n import lang
    from tui.common.plain_monitor import _toggle_lang

    monkeypatch.delenv("LLMUX_LANG", raising=False)
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("LANG", "ko_KR.UTF-8")
    assert lang() == "en"
    monkeypatch.setenv("LC_ALL", "ko_KR.UTF-8")
    _toggle_lang()
    assert lang() == "en"


def test_profile_clone_preserves_existing_destination(tmp_path, monkeypatch):
    from tui.common import profile_store

    path = tmp_path / "profiles.yaml"
    path.write_text("version: 1\nprofiles:\n- {name: src, backend: vllm}\n- {name: dst, backend: vllm}\n")
    monkeypatch.setattr(profile_store, "PROFILES_YAML", path)
    monkeypatch.setattr(profile_store, "RUNTIME_DIR", tmp_path / "runtime")
    before = path.read_text()
    with pytest.raises(ValueError, match="already exists"):
        profile_store.clone_profile("src", "dst", "vllm")
    assert path.read_text() == before


def test_nested_config_comments_survive_edits():
    from tui.common.config_markers import dump_active_config

    original = "compilation-config:\n  level: 3 # keep this\n  mode: old # keep mode\n"
    rendered = dump_active_config(original, {"compilation-config": {"level": 3, "mode": "new"}})
    assert "# keep this" in rendered
    assert "new # keep mode" in rendered
    import yaml

    typed = dump_active_config("# keep\nflag: true\nvalues: [true]\n", {"flag": 1, "values": [1]})
    assert type(yaml.safe_load(typed)["flag"]) is int
    assert type(yaml.safe_load(typed)["values"][0]) is int


def test_histogram_restart_does_not_produce_negative_latency():
    from tui.common.metrics import Hist
    from tui.common.monitor_render import MonitorState

    state = MonitorState()
    state._win_avg("ttft", Hist(sum=100, count=1))
    assert state._win_avg("ttft", Hist(sum=2, count=2)) == 1


def test_literal_env_value_is_not_expanded(tmp_path, monkeypatch):
    from tui.common.env import parse_env_file

    monkeypatch.setenv("LLMUX_TEST_VALUE", "expanded")
    path = tmp_path / "env"
    path.write_text("TOKEN='literal$LLMUX_TEST_VALUE'\nPATH_VALUE=/tmp/$LLMUX_TEST_VALUE\n")
    assert parse_env_file(path, expand=True) == {
        "TOKEN": "literal$LLMUX_TEST_VALUE", "PATH_VALUE": "/tmp/expanded"
    }


@pytest.mark.asyncio
async def test_authenticated_http_requests_keep_the_key_out_of_urls():
    from tui.common import http

    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'{"data":[{"id":"model"}],"usage":{"completion_tokens":1}}'
    with patch.object(http, "open_url", return_value=response) as request:
        assert await http.list_served_models(8000, api_key="secret") == ["model"]
        await http.chat_completion_bench(8000, "model", api_key="secret")
    for call in request.call_args_list:
        req = call.args[0]
        assert req.get_header("Authorization") == "Bearer secret"
        assert "secret" not in req.full_url


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vllm", "llamacpp"])
async def test_readiness_authentication_and_error_protocol(backend):
    runtime = importlib.import_module(f"tui.backends.{backend}.backend_runtime")
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'{"data":[{"id":"model"}]}'
    with patch.object(runtime, "open_url", return_value=response) as request:
        assert await runtime._models_endpoint_ready(8000, api_key="secret")
        assert request.call_args.args[0].get_header("Authorization") == "Bearer secret"
    with patch.object(runtime, "load_config", side_effect=ValueError("invalid config")):
        events = [event async for event in runtime._post_start_validation(runtime.Profile(name="p"))]
    assert events == [("result", False, ["API authentication configuration failed: invalid config"])]


@pytest.mark.asyncio
async def test_auth_key_sources():
    from tui.common.http import config_api_key

    assert await config_api_key({}, "server", env={"VLLM_API_KEY": "env-key"}) == "env-key"
    assert await config_api_key({"api-key": "first,second"}, "server", backend="llamacpp") == "first"
    with patch("tui.common.docker.run_command", AsyncMock(return_value=(0, "# comment\n\nfile-key\nsecond\n"))) as read:
        assert await config_api_key({"api-key-file": "/keys"}, "server", backend="llamacpp") == "file-key"
    read.assert_awaited_once_with("docker", "exec", "server", "cat", "--", "/keys")
    with patch("tui.common.docker.run_command", AsyncMock(return_value=(1, "sensitive output"))):
        with pytest.raises(RuntimeError, match="could not read") as error:
            await config_api_key({"api-key-file": "/keys"}, "server")
    assert "sensitive" not in str(error.value)


def test_prepare_revision_reaches_snapshot_download(monkeypatch):
    from tui.common.prepare import _VLLM_DOWNLOAD_SNIPPET, resolve_gguf_file

    snapshot = MagicMock()
    monkeypatch.setenv("LLMUX_PREPARE_MODEL", "org/model")
    monkeypatch.setenv("LLMUX_PREPARE_IGNORE", "original/**")
    monkeypatch.setenv("LLMUX_PREPARE_REVISION", "pinned-revision")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    exec(_VLLM_DOWNLOAD_SNIPPET, {})
    assert snapshot.call_args.kwargs["revision"] == "pinned-revision"
    assert resolve_gguf_file("configured.gguf", "old.gguf", "") == "configured.gguf"
    assert resolve_gguf_file("configured.gguf", "old.gguf", "explicit.gguf") == "explicit.gguf"


@pytest.mark.asyncio
async def test_closing_log_screen_keeps_other_work_running():
    from textual.app import App
    from tui.backends.vllm.screens.container import LogScreen

    app = App()
    with patch.object(LogScreen, "on_mount", return_value=None):
        async with app.run_test() as pilot:
            unrelated = app.run_worker(asyncio.Event().wait())
            screen = LogScreen("server")
            await app.push_screen(screen)
            await pilot.pause()
            screen.action_go_back()
            assert not unrelated.is_cancelled


@pytest.mark.parametrize("backend", ["vllm", "llamacpp"])
def test_runtime_env_keeps_literal_values(tmp_path, monkeypatch, backend):
    runtime = importlib.import_module(f"tui.backends.{backend}.backend_runtime")
    monkeypatch.setenv("LLMUX_TEST_VALUE", "expanded")
    common = tmp_path / "common.env"
    common.write_text("HF_TOKEN='literal$LLMUX_TEST_VALUE'\nHF_CACHE_PATH=/tmp/$LLMUX_TEST_VALUE\n")
    monkeypatch.setattr(runtime, "COMMON_ENV", common)
    profile = runtime.Profile(name="p")
    env = runtime._compose_env(profile, **({"use_dev": False} if backend == "vllm" else {}))
    assert env["HF_TOKEN"] == "literal$LLMUX_TEST_VALUE"
    assert env["HF_CACHE_PATH"] == "/tmp/expanded"
