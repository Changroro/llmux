"""vLLM `up` readiness outcomes, timeouts and dry-run."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

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


def _profile():
    return rt.Profile(name="p", container_name="p", port="8000", image_tag="vllm/vllm-openai:v0.20.1")


def _patches(validation, commands, **extra):
    async def compose(cmd, **_kwargs):
        commands.append(cmd)
        yield ("rc", 0)

    profile = _profile()
    values = {
        "load_profile": lambda _name: profile,
        "check_port_conflict": AsyncMock(return_value=None),
        "_ensure_common_env": lambda _profile: (True, []),
        "_ensure_profile_config": lambda _profile, **_kw: (True, []),
        "_render_profile_snapshot": lambda _name, **_k: (profile, "/unused"),
        "_gpu_conflict_messages": AsyncMock(return_value=[]),
        "_compose_env": lambda *_a, **_k: {},
        "_post_start_validation": validation,
        "_verify_vllm_version": _no_events,
        "profile_store": _Store,
        "stream_command": compose,
        **extra,
    }
    return patch.dict(rt.stream_container_up.__globals__, values)


async def _no_events(*_args, **_kwargs):
    if False:
        yield ("log", "")


async def _collect(**kwargs):
    return [event async for event in rt.stream_container_up("p", **kwargs)]


@pytest.mark.asyncio
async def test_crashed_container_is_removed_by_the_id_seen_at_detection():
    async def crashed(*_a, **_k):
        yield ("result", False, ["Error: container 'p' exited during startup (exited)."], "crashed", "abc123")

    run = AsyncMock(return_value=(0, "abc123"))
    down = AsyncMock()
    commands: list = []
    with _patches(crashed, commands, run_command=run, container_down=down):
        events = await _collect()
    run.assert_awaited_once_with("docker", "rm", "-f", "abc123", timeout=30)
    down.assert_not_awaited()
    assert events[-1] == ("rc", exit_codes.ERROR)


@pytest.mark.asyncio
async def test_validation_reports_the_crashed_container_id():
    profile = _profile()

    async def run(*args, **_kwargs):
        if args[:2] == ("docker", "inspect"):
            return 0, "exited\tnone\tabc123"
        return 0, "tail\n"

    with patch.object(rt, "run_command", side_effect=run), patch.object(
        rt, "_models_endpoint_ready", AsyncMock(return_value=False)
    ), patch.object(rt, "load_config", return_value=rt.Config(name="p", model="m")):
        events = [e async for e in rt._post_start_validation(profile, timeout=0.05, poll_interval=0.01)]
    assert events[-1][3:] == ("crashed", "abc123")


@pytest.mark.asyncio
async def test_stalled_container_is_left_running_with_not_ready_code():
    async def stalled(*_a, **_k):
        yield ("result", False, ["Error: container started but /v1/models is not ready within timeout."], "stalled")

    down = AsyncMock()
    commands: list = []
    with _patches(stalled, commands, container_down=down):
        events = await _collect()
    down.assert_not_awaited()
    assert events[-1] == ("rc", exit_codes.NOT_READY)
    assert any("llmux wait p" in str(data) for kind, data in events if kind == "log")


@pytest.mark.asyncio
async def test_timeouts_reach_readiness_validation():
    seen: dict = {}

    async def ready(*_a, **kwargs):
        seen.update(kwargs)
        yield ("result", True, [])

    commands: list = []
    with _patches(ready, commands):
        events = await _collect(stall_timeout=90.0, ready_timeout=600.0)
    assert events[-1] == ("rc", 0)
    assert seen["timeout"] == 90.0 and seen["max_wait"] == 600.0


@pytest.mark.asyncio
async def test_port_conflict_returns_conflict_code():
    async def never(*_a, **_k):
        raise AssertionError("validation must not run")
        yield

    commands: list = []
    with _patches(never, commands, check_port_conflict=AsyncMock(return_value="profile 'q'")):
        events = await _collect()
    assert events[-1] == ("rc", exit_codes.CONFLICT)
    assert commands == []


@pytest.mark.asyncio
async def test_dry_run_plans_without_side_effects():
    async def never(*_a, **_k):
        raise AssertionError("validation must not run")
        yield

    ensure_calls: list = []
    port_probe = AsyncMock(return_value="profile 'q'")
    commands: list = []
    with _patches(
        never,
        commands,
        check_port_conflict=port_probe,
        _ensure_profile_config=lambda _p, **kw: ensure_calls.append(kw) or (True, []),
    ):
        events = await _collect(dry_run=True)
    assert commands == []
    port_probe.assert_not_awaited()
    assert ensure_calls == [{"dry_run": True}]
    plans = [data for kind, data in events if kind == "plan"]
    assert plans and plans[0]["image"] == "vllm/vllm-openai:v0.20.1"
    assert plans[0]["command"][:2] == ["docker", "compose"]
    assert events[-1] == ("rc", 0)


@pytest.mark.asyncio
async def test_validation_reports_crash_and_stall_outcomes():
    profile = _profile()

    async def run(*args, **_kwargs):
        if args[:2] == ("docker", "inspect"):
            return 0, state["value"]
        if args[:2] == ("docker", "logs"):
            return 0, "same\n"
        return 0, ""

    state = {"value": "exited\tnone"}
    with patch.object(rt, "run_command", side_effect=run), patch.object(
        rt, "_models_endpoint_ready", AsyncMock(return_value=False)
    ), patch.object(rt, "load_config", return_value=rt.Config(name="p", model="m")):
        crash = [e async for e in rt._post_start_validation(profile, timeout=0.05, poll_interval=0.01)]
        state["value"] = "running\tstarting"
        stall = [e async for e in rt._post_start_validation(profile, timeout=0.05, poll_interval=0.01)]
    assert crash[-1][3] == "crashed"
    assert stall[-1][3] == "stalled"


@pytest.mark.asyncio
async def test_running_but_unhealthy_is_still_loading_not_crashed():
    """Compose marks a slow first load unhealthy (~690s) while it keeps running."""
    profile = _profile()

    async def run(*args, **_kwargs):
        if args[:2] == ("docker", "inspect"):
            return 0, "running\tunhealthy"
        if args[:2] == ("docker", "logs"):
            return 0, "same\n"
        return 0, ""

    with patch.object(rt, "run_command", side_effect=run), patch.object(
        rt, "_models_endpoint_ready", AsyncMock(return_value=False)
    ), patch.object(rt, "load_config", return_value=rt.Config(name="p", model="m")):
        events = [e async for e in rt._post_start_validation(profile, timeout=0.05, poll_interval=0.01)]
    assert events[-1][3] == "stalled"


@pytest.mark.asyncio
async def test_wait_keeps_waiting_on_running_unhealthy(monkeypatch):
    from tui.common import docker, readiness

    snapshot = docker.ContainerSnapshot(
        name="p", lifecycle=docker.ContainerLifecycle.RUNNING,
        health=docker.ContainerHealth.UNHEALTHY, raw_state="running", raw_status="running",
    )
    monkeypatch.setattr(readiness.docker, "container_snapshots", AsyncMock(return_value={"p": snapshot}))
    monkeypatch.setattr(readiness, "probe_served", AsyncMock(return_value=("loading", [], "")))
    stored = profile_store.StoredProfile(name="p", backend="vllm", port=8000)
    rc, _msg, _models = await readiness.wait_ready(stored, timeout=0, interval=0.01)
    assert rc == exit_codes.NOT_READY


@pytest.mark.asyncio
async def test_tui_not_ready_says_still_loading_and_follows_logs(monkeypatch):
    from textual.app import App
    from tui.backends.vllm.screens import container as screen_mod

    profile = rt.Profile(name="p", container_name="p", port="8000", config_name="p", image_tag="vllm/vllm-openai:v1")
    monkeypatch.setattr(screen_mod, "load_profile", lambda _n: profile)
    monkeypatch.setattr(screen_mod.ContainerUpScreen, "on_mount", lambda self: None)
    monkeypatch.setattr(screen_mod, "check_port_conflict", AsyncMock(return_value=None))
    monkeypatch.setattr("tui.common.widgets.confirm_model_switch", AsyncMock(return_value=True))
    followed: list = []

    async def up(*_a, **_k):
        yield ("rc", exit_codes.NOT_READY)

    async def logs(name, **_k):
        followed.append(name)
        yield "loading..."

    monkeypatch.setattr(screen_mod, "stream_container_up", up)
    monkeypatch.setattr(screen_mod, "stream_container_logs", logs)
    app = App()
    async with app.run_test(size=(120, 45)) as pilot:
        screen = screen_mod.ContainerUpScreen("p")
        await app.push_screen(screen)
        await pilot.pause()
        screen.action_confirm_start()
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = str(screen.query_one("#startup-status").render())
    assert "still loading" in text.lower() or "로딩" in text
    assert followed == ["p"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("present", "pull_rc", "stage", "rc", "pulled"),
    [
        (True, 0, True, 0, False),
        (False, 0, True, 0, True),
        (False, 1, True, exit_codes.ERROR, True),
        (False, 0, False, 0, False),
    ],
)
async def test_dry_run_secures_the_image_when_staging(monkeypatch, present, pull_rc, stage, rc, pulled):
    from tui.common import prepare

    pulls: list = []

    async def pull(ref):
        pulls.append(ref)
        yield ("log", "pulling")
        yield ("rc", pull_rc)

    monkeypatch.setattr(prepare, "image_present", AsyncMock(return_value=present))
    monkeypatch.setattr(prepare, "stream_pull", pull)

    async def never(*_a, **_k):
        raise AssertionError("validation must not run")
        yield

    commands: list = []
    with _patches(never, commands):
        events = await _collect(dry_run=True, stage_image=stage)
    assert events[-1] == ("rc", rc)
    assert bool(pulls) is pulled
    assert commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tag", ["", "vllm/vllm-openai:v0.21.0"])
@pytest.mark.parametrize(("staged", "policy"), [(False, "always"), (True, "never")])
async def test_staged_start_reuses_the_image_the_precheck_secured(tag, staged, policy):
    """After --replace stopped the old model, the real start must not pull again."""

    async def ready(*_a, **_k):
        yield ("result", True, [])

    commands: list = []
    with _patches(ready, commands):
        events = await _collect(pull=True, tag=tag, image_staged=staged)
    assert events[-1] == ("rc", 0)
    assert commands[0][commands[0].index("--pull") + 1] == policy


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "outcome"), [(401, "error"), (503, None)])
async def test_models_probe_auth_failure_is_a_config_error_not_loading(code, outcome):
    import urllib.error

    def answer(*_a, **_k):
        raise urllib.error.HTTPError("http://127.0.0.1:8000/v1/models", code, "x", {}, None)

    async def run(*args, **_kwargs):
        if args[:2] == ("docker", "inspect"):
            return 0, "running\thealthy\tabc"
        return 0, "same\n"

    with patch.object(rt, "open_url", side_effect=answer), patch.object(
        rt, "run_command", side_effect=run
    ), patch.object(rt, "load_config", return_value=rt.Config(name="p", model="m")):
        events = [e async for e in rt._post_start_validation(_profile(), timeout=0.05, poll_interval=0.01)]
    result = events[-1]
    if outcome is None:
        assert result[3] == "stalled"
    else:
        assert result[1] is False and result[3] == outcome
        assert "401" in " ".join(result[2])


@pytest.mark.asyncio
async def test_misconfigured_server_exits_error_and_keeps_running():
    async def misconfigured(*_a, **_k):
        yield ("result", False, ["Error: /v1/models answered HTTP 401"], "error")

    commands: list = []
    with _patches(misconfigured, commands):
        events = await _collect()
    assert events[-1] == ("rc", exit_codes.ERROR)


@pytest.mark.asyncio
async def test_dry_run_plan_names_the_profile_env_the_real_up_would_render(tmp_path, monkeypatch):
    """The plan is the exact compose command, even before the runtime .env exists."""
    from tui.backends.vllm import backend_common

    monkeypatch.setattr(backend_common, "RUNTIME_DIR", tmp_path)

    async def never(*_a, **_k):
        raise AssertionError("validation must not run")
        yield

    commands: list = []
    with _patches(never, commands):
        events = await _collect(dry_run=True)
    command = [data for kind, data in events if kind == "plan"][0]["command"]
    env_file = str(tmp_path / "p.env")
    assert not (tmp_path / "p.env").exists()
    assert command[command.index(env_file) - 1] == "--env-file"
