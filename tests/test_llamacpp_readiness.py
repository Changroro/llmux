from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tui.backends.llamacpp import backend as lbackend
from tui.backends.llamacpp import backend_runtime as lrt
from tui.common.exit_codes import CONFLICT, NOT_READY, UNREACHABLE

PROFILE = lbackend.Profile(
    name="p", container_name="p", port=8000, config_name="p", hf_repo="o/r", hf_file="m.gguf"
)


async def _validate(docker_run, ready=False, **kwargs):
    events = []
    with patch.dict(
        lrt._post_start_validation.__globals__,
        {"_docker_run": docker_run, "_models_endpoint_ready": AsyncMock(return_value=ready)},
    ):
        async for event in lrt._post_start_validation(PROFILE, poll_interval=0.02, **kwargs):
            events.append(event)
    return events


def _docker(status: str, *, du=None, inspect_rc=0):
    async def run(*args, **kwargs):
        if args[0] == "du":
            return du if du is not None else (0, "0\t/x\n")
        if args[:2] == ("docker", "inspect"):
            return inspect_rc, f"{status}\tcid123" if inspect_rc == 0 else "no daemon"
        if args[:2] == ("docker", "logs"):
            return 0, "boom\n"
        return 0, ""
    return run


@pytest.mark.asyncio
async def test_exited_container_is_a_crash() -> None:
    events = await _validate(_docker("exited\tnone"), timeout=1)
    assert events[-1][0] == "result" and events[-1][1] is False
    assert events[-1][3] == "crashed"
    assert events[-1][4] == "cid123"  # the container that crashed, for a targeted cleanup


@pytest.mark.asyncio
async def test_running_but_unhealthy_is_still_loading_not_a_crash() -> None:
    # A first-time GGUF download can outlast the healthcheck start_period.
    events = await _validate(_docker("running\tunhealthy"), timeout=0.05)
    assert events[-1][3] == "stalled"


@pytest.mark.asyncio
async def test_inspect_failure_is_unreachable() -> None:
    events = await _validate(_docker("", inspect_rc=1), timeout=1)
    assert events[-1][3] == "unreachable"


@pytest.mark.asyncio
async def test_stall_is_reported_as_stalled() -> None:
    events = await _validate(_docker("running\tstarting"), timeout=0.05)
    assert events[-1][3] == "stalled"


@pytest.mark.asyncio
async def test_du_failure_only_warns_and_keeps_waiting(tmp_path) -> None:
    events = await _validate(
        _docker("running\tstarting", du=(1, "du: timed out")),
        timeout=0.05,
        hf_cache_path=str(tmp_path),
    )
    warnings = [e for e in events if e[0] == "log" and e[1].startswith("Warning: startup progress probe (du)")]
    assert len(warnings) == 1
    assert events[-1][3] == "stalled"


@pytest.mark.asyncio
async def test_missing_progress_dir_counts_as_empty() -> None:
    du = AsyncMock(side_effect=AssertionError("du must not run on a missing dir"))
    events = await _validate(_docker("running\tstarting"), timeout=0.05, hf_cache_path="/no/such/dir")
    assert events[-1][3] == "stalled"
    du.assert_not_called()


async def _up(validation_result, validation=None, port=None, **kwargs):
    async def fixed(*_a, **_k):
        yield validation_result

    async def compose(_cmd, **_k):
        yield ("rc", 0)

    down = AsyncMock(return_value=(0, "✓ 'p' 중지 완료"))
    docker = AsyncMock(return_value=(0, ""))
    globals_patch = {
        "validate_common_env": lambda _p: (True, []),
        "load_profile": lambda _n: PROFILE,
        "check_port_conflict": port or AsyncMock(return_value=None),
        "_gpu_conflict_messages": AsyncMock(return_value=[]),
        "_ensure_profile_config": lambda *_a: (True, []),
        "_resolve_pending_quant": AsyncMock(return_value=""),
        "_render_profile_snapshot": lambda _n: (PROFILE, None),
        "_render_override": AsyncMock(return_value=(0, "")),
        "_stream": compose,
        "_post_start_validation": validation or fixed,
        "container_down": down,
        "_docker_run": docker,
    }
    stored = lbackend.profile_store.StoredProfile(name="p", backend="llamacpp")
    with patch.dict(lrt.stream_container_up.__globals__, globals_patch), \
            patch.object(lrt.profile_store, "load_profile", return_value=stored):
        events = [e async for e in lrt.stream_container_up("p", **kwargs)]
    return events, docker


@pytest.mark.asyncio
async def test_crash_prints_logs_then_removes_exactly_that_container() -> None:
    # By ID, not by name: a newer `up` may already own the name (no restart loop either way).
    events, docker = await _up(("result", False, ["Error: exited", "  boom"], "crashed", "cid123"))
    docker.assert_awaited_once()
    assert docker.await_args.args == ("docker", "rm", "-f", "cid123")
    assert events[-1] == ("rc", 1)
    assert any("boom" in e[1] for e in events if e[0] == "log")


@pytest.mark.asyncio
async def test_stall_leaves_container_and_exits_not_ready() -> None:
    events, docker = await _up(("result", False, ["Error: not ready"], "stalled"))
    docker.assert_not_awaited()
    assert events[-1] == ("rc", NOT_READY)
    assert any("llmux wait p" in e[1] for e in events if e[0] == "log")


@pytest.mark.asyncio
async def test_inspect_failure_exits_unreachable() -> None:
    events, docker = await _up(("result", False, ["Error: could not inspect"], "unreachable"))
    docker.assert_not_awaited()
    assert events[-1] == ("rc", UNREACHABLE)


@pytest.mark.asyncio
async def test_other_failures_exit_1() -> None:
    events, docker = await _up(("result", False, ["API authentication configuration failed"]))
    docker.assert_not_awaited()
    assert events[-1] == ("rc", 1)


@pytest.mark.asyncio
async def test_timeouts_reach_the_validation() -> None:
    seen = {}

    async def validation(_profile, **kwargs):
        seen.update(kwargs)
        yield ("result", True, [])

    events, _ = await _up(None, validation=validation, stall_timeout=7.0, ready_timeout=99.0)
    assert (seen["timeout"], seen["max_wait"]) == (7.0, 99.0)
    assert events[-1] == ("rc", 0)


@pytest.mark.asyncio
async def test_port_conflict_and_probe_failure_have_distinct_codes() -> None:
    events, _ = await _up(None, port=AsyncMock(return_value="profile 'x'"))
    assert events[-1] == ("rc", CONFLICT)
    events, _ = await _up(None, port=AsyncMock(side_effect=RuntimeError("docker ps failed")))
    assert events[-1] == ("rc", UNREACHABLE)


@pytest.mark.asyncio
async def test_dry_run_plans_without_writing_or_starting(tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    forbidden = AsyncMock(side_effect=AssertionError("dry-run must not do this"))
    stored = lbackend.profile_store.StoredProfile(name="p", backend="llamacpp")
    globals_patch = {
        "validate_common_env": lambda _p: (True, []),
        "load_profile": lambda _n: lbackend.Profile(
            name="p", container_name="p", port=8000, config_name="p", hf_repo="o/r:Q4_K_M"
        ),
        "check_port_conflict": forbidden,
        "_gpu_conflict_messages": AsyncMock(return_value=[]),
        "_resolve_pending_quant": forbidden,
        "_render_override": forbidden,
        "_stream": forbidden,
        "_stream_build_dev_image": forbidden,
        "CONFIG_DIR": config_dir,  # no config yet → previewed with the defaults
    }
    with patch.dict(lrt.stream_container_up.__globals__, globals_patch), \
            patch.object(lrt.profile_store, "load_profile", return_value=stored), \
            patch.object(lrt.profile_store, "save_profile", side_effect=AssertionError("dry-run wrote")), \
            patch.object(lrt, "save_config", side_effect=AssertionError("dry-run wrote")), \
            patch.object(lbackend, "list_repo_ggufs", AsyncMock(return_value=["r-Q4_K_M.gguf"])):
        events = [e async for e in lrt.stream_container_up("p", dry_run=True)]

    plans = [e[1] for e in events if e[0] == "plan"]
    assert events[-1] == ("rc", 0), events
    assert len(plans) == 1
    assert plans[0]["command"][:2] == ["docker", "compose"] and "up" in plans[0]["command"]
    command = plans[0]["server_args"]
    assert command[command.index("-hf") + 1] == "o/r"
    assert command[command.index("-hff") + 1] == "r-Q4_K_M.gguf"
    assert command[command.index("--ctx-size") + 1] == "8192"
    assert plans[0]["build"] is False and plans[0]["image"]
    assert not list(config_dir.iterdir())


@pytest.mark.asyncio
async def test_dry_run_still_rejects_a_config_the_real_start_would_reject(tmp_path) -> None:
    # Otherwise `up --replace` stops the other model before discovering this one cannot start.
    (tmp_path / "p.yaml").write_text("# nothing set\n")
    stored = lbackend.profile_store.StoredProfile(name="p", backend="llamacpp", config_name="p")
    globals_patch = {
        "validate_common_env": lambda _p: (True, []),
        "load_profile": lambda _n: lbackend.Profile(
            name="p", container_name="p", port=8000, config_name="p", hf_repo="o/r", hf_file="m.gguf"
        ),
        "_gpu_conflict_messages": AsyncMock(return_value=[]),
        "CONFIG_DIR": tmp_path,
    }
    with patch.dict(lrt.stream_container_up.__globals__, globals_patch), \
            patch.object(lrt.profile_store, "load_profile", return_value=stored), \
            patch.object(lrt.profile_store, "save_profile", side_effect=AssertionError("dry-run wrote")):
        events = [e async for e in lrt.stream_container_up("p", dry_run=True)]
    assert events[-1] == ("rc", 1)
    assert not [e for e in events if e[0] == "plan"]
    assert any("비어 있습니다" in e[1] for e in events if e[0] == "log")


@pytest.mark.asyncio
async def test_crash_without_a_known_id_removes_nothing_and_warns() -> None:
    events, docker = await _up(("result", False, ["Error: exited"], "crashed"))
    docker.assert_not_awaited()
    assert events[-1] == ("rc", 1)
    assert any(e[1].startswith("Warning:") for e in events if e[0] == "log")



@pytest.mark.asyncio
async def test_models_endpoint_auth_error_is_not_loading() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            self.send_response(401 if self.path == "/v1/models" else 503)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(lrt.ServerAnswerError, match="401"):
            await lrt._models_endpoint_ready(server.server_address[1])
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_auth_error_during_startup_fails_up_instead_of_waiting() -> None:
    events = []
    with patch.dict(
        lrt._post_start_validation.__globals__,
        {"_docker_run": _docker("running"), "_models_endpoint_ready": AsyncMock(side_effect=lrt.ServerAnswerError("HTTP Error 401: Unauthorized"))},
    ):
        async for event in lrt._post_start_validation(PROFILE, poll_interval=0.02):
            events.append(event)
    result = [e for e in events if e[0] == "result"][-1]
    assert result[1] is False and "401" in " ".join(result[2])
    assert len(result) < 4 or result[3] not in ("stalled", "crashed")
