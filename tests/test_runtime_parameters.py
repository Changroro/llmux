import asyncio
import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_profile_discovery_uses_running_image_not_saved_pin():
    from tui.common import flag_discovery, profile_store

    profile = profile_store.StoredProfile(name="p", backend="vllm", image_tag="old:tag")
    state = json.dumps({"Image": "sha256:actual", "State": {"Running": True}})
    with patch.object(profile_store, "load_profile", return_value=profile), patch.object(
        flag_discovery, "run_command", AsyncMock(return_value=(0, state))
    ):
        assert await flag_discovery.profile_target("vllm", "p") == ("sha256:actual", "p")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vllm", "llamacpp"])
async def test_failed_discovery_has_no_unverified_completion(backend):
    module = importlib.import_module(f"tui.backends.{backend}.screens.config")
    screen = module.ConfigFormScreen()
    known = screen._known_params if backend == "vllm" else screen._known_flags
    assert not known


@pytest.mark.asyncio
async def test_grouped_vllm_help_expands_all_flags(tmp_path):
    from tui.backends.vllm import backend_inspect as inspect

    command = AsyncMock(side_effect=[(0, "Use --help=all for all flags\n--host HOST"), (0, "--host HOST\n--max-model-len N")])
    with patch.object(inspect, "run_command", command), patch("tui.common.docker.image_identity", AsyncMock(return_value="sha256:123")), patch.object(inspect, "_VLLM_PARAMS_CACHE_DIR", tmp_path):
        flags = await inspect.extract_vllm_params("repo/image:v1")
    assert "max-model-len" in flags
    assert command.call_args.args[-1] == "--help=all"


@pytest.mark.asyncio
async def test_legacy_help_is_not_filtered_with_help_all(tmp_path):
    from tui.backends.vllm import backend_inspect as inspect

    command = AsyncMock(return_value=(0, "--host HOST\n--max-model-len N\n--allowed-local-media-path PATH"))
    with patch.object(inspect, "run_command", command), patch("tui.common.docker.image_identity", AsyncMock(return_value="sha256:old")), patch.object(inspect, "_VLLM_PARAMS_CACHE_DIR", tmp_path):
        assert "max-model-len" in await inspect.extract_vllm_params("repo/image:old")
    command.assert_awaited_once()
    assert command.call_args.args[-1] == "--help"


@pytest.mark.asyncio
async def test_live_vllm_queries_container_and_does_not_reuse_image_cache(tmp_path):
    from tui.backends.vllm import backend_inspect as inspect

    command = AsyncMock(side_effect=[(0, "--before-upgrade"), (0, "--after-upgrade")])
    with patch.object(inspect, "run_command", command), patch.object(inspect, "_VLLM_PARAMS_CACHE_DIR", tmp_path):
        assert await inspect.extract_vllm_params("sha256:actual", container_name="live") == {"before-upgrade"}
        assert await inspect.extract_vllm_params("sha256:actual", container_name="live") == {"after-upgrade"}
    assert command.call_args.args == ("docker", "exec", "live", "vllm", "serve", "--help")
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_live_llamacpp_queries_container_without_image_probe():
    from tui.backends.llamacpp import backend

    proc = MagicMock(returncode=0)
    proc.communicate = AsyncMock(return_value=(b"--ctx-size N\n--live-only", b""))
    with patch.object(asyncio, "create_subprocess_exec", AsyncMock(return_value=proc)) as execute, patch("tui.common.docker.image_identity", AsyncMock()) as inspect:
        assert await backend.extract_llama_server_flags("sha256:live", container_name="live") == {"ctx-size", "live-only"}
    inspect.assert_not_awaited()
    assert execute.call_args.args == ("docker", "exec", "live", "/app/llama-server", "--help")


@pytest.mark.parametrize("backend", ["vllm", "llamacpp"])
def test_cli_profile_flags_uses_runtime_target(backend):
    from typer.testing import CliRunner
    from tui.cli import app
    from tui.common import profile_store

    profile = profile_store.StoredProfile(name="p", backend=backend, image_tag="old:tag")
    function = "tui.backends.vllm.backend_inspect.extract_vllm_params" if backend == "vllm" else "tui.backends.llamacpp.backend.extract_llama_server_flags"
    with patch.object(profile_store, "find_name_owner", return_value=backend), patch.object(profile_store, "load_profile", return_value=profile), patch("tui.common.flag_discovery.profile_target", AsyncMock(return_value=("sha256:live", "live"))), patch(function, AsyncMock(return_value={"live-only"})) as extract:
        result = CliRunner().invoke(app, ["config", "flags", "--profile", "p", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["image"] == "sha256:live"
    extract.assert_awaited_once_with("sha256:live", container_name="live")


@pytest.mark.asyncio
async def test_inspection_failure_is_not_treated_as_stopped():
    from tui.common import flag_discovery, profile_store

    profile = profile_store.StoredProfile(name="p", backend="vllm")
    with patch.object(profile_store, "load_profile", return_value=profile), patch.object(flag_discovery, "run_command", AsyncMock(return_value=(1, "daemon unavailable"))):
        with pytest.raises(RuntimeError, match="daemon unavailable"):
            await flag_discovery.profile_target("vllm", "p")


@pytest.mark.asyncio
async def test_release_discovery_handles_more_than_five_pages_and_rejects_cycles():
    from tui.backends.vllm import backend_inspect as inspect

    pages = [{"results": [{"name": f"v0.{i}.0"}], "next": f"?page={i+1}" if i < 7 else None} for i in range(1, 8)]
    with patch.object(inspect, "_fetch_json_url", AsyncMock(side_effect=pages)):
        assert await inspect.get_dockerhub_release_version() == "v0.7.0"
    url = "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100"
    with patch.object(inspect, "_fetch_json_url", AsyncMock(return_value={"results": [], "next": url})) as fetch:
        with pytest.raises(RuntimeError, match="repeated"):
            await inspect.get_dockerhub_release_version()
    assert fetch.await_count == 1


@pytest.mark.asyncio
async def test_authenticated_calls_do_not_follow_cross_origin_redirects():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from tui.common import http
    from tui.backends.vllm import backend_runtime as vllm
    from tui.backends.llamacpp import backend_runtime as llama

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.server is source:
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/stolen")
                self.end_headers()
            else:
                received.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data":[{"id":"model"}]}')

        do_POST = do_GET

        def log_message(self, *args):
            pass

    target = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    source = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threads = [Thread(target=server.serve_forever, daemon=True) for server in (source, target)]
    for thread in threads:
        thread.start()
    try:
        with pytest.raises(RuntimeError, match="302"):
            await http.list_served_models(source.server_port, api_key="secret")
        assert not await vllm._models_endpoint_ready(source.server_port, api_key="secret")
        assert not await llama._models_endpoint_ready(source.server_port, api_key="secret")
        with pytest.raises(Exception, match="302"):
            await http.chat_completion_bench(source.server_port, "model", api_key="secret")
        assert not received
    finally:
        for server in (source, target):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()


@pytest.mark.asyncio
async def test_downloader_close_waits_for_inner_process():
    from tui.common import prepare

    proc = MagicMock(returncode=None)
    proc.stdout.read = AsyncMock(return_value=b"progress\n")
    proc.wait = AsyncMock()
    proc.kill.side_effect = lambda: setattr(proc, "returncode", -9)
    original = prepare.stream_lines
    streams = []

    def captured(*args, **kwargs):
        stream = original(*args, **kwargs)
        streams.append(stream)
        return stream

    with patch.object(prepare, "stream_lines", captured), patch.object(prepare, "_remove_prepare_container", AsyncMock(return_value=(0, ""))), patch.object(prepare, "workers_env", return_value=[]), patch.object(asyncio, "create_subprocess_exec", AsyncMock(return_value=proc)):
        outer = prepare.stream_vllm_download(image_ref="image", model_id="org/model", cache_path="/cache", token="", container_name="test")
        await anext(outer)
        try:
            await outer.aclose()
            proc.kill.assert_called_once()
            proc.wait.assert_awaited_once()
        finally:
            for stream in streams:
                await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("pull,present", [(False, True), (True, True), (True, False)])
async def test_explicit_dev_image_is_never_pulled(pull, present):
    from tui.backends.vllm import backend_runtime as runtime

    profile = runtime.Profile(name="p", container_name="p", config_name="p")
    commands = []

    async def compose(command, **kwargs):
        commands.append(command)
        yield ("rc", 1)

    with patch.object(runtime.profile_store, "load_profile", return_value=object()), patch.object(runtime, "load_profile", return_value=profile), patch.object(runtime, "check_port_conflict", AsyncMock(return_value=None)), patch.object(runtime, "_ensure_common_env", return_value=(True, [])), patch.object(runtime, "_ensure_profile_config", return_value=(True, [])), patch.object(runtime, "_render_profile_snapshot", return_value=(profile, Path("/unused"))), patch.object(runtime, "_gpu_conflict_messages", AsyncMock(return_value=[])), patch.object(runtime, "_compose_env", return_value={}), patch.object(runtime.dev_build, "image_exists_locally", AsyncMock(return_value=present)), patch.object(runtime, "stream_command", compose):
        events = [event async for event in runtime.stream_container_up("p", tag="vllm-dev:main", pull=pull)]
    if present:
        assert commands[0][-2:] == ["--pull", "never"]
    else:
        assert not commands
        assert any("build-dev" in str(value) for _, value in events)
        assert events[-1] == ("rc", 1)
