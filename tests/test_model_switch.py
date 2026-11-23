from __future__ import annotations

import importlib
from unittest.mock import AsyncMock

import pytest
from textual.app import App

from tui.common import model_switch, profile_store
from tui.common.widgets import ConfirmModal


@pytest.fixture
def switch_state(monkeypatch):
    profiles = [
        profile_store.StoredProfile(name="new", backend="llamacpp", container_name="new", port=8000, gpu_id="0"),
        profile_store.StoredProfile(name="old", backend="vllm", container_name="old", port=8000, gpu_id="0"),
        profile_store.StoredProfile(name="unrelated", backend="vllm", container_name="unrelated", port=9000, gpu_id="1"),
    ]
    running = {"old", "unrelated"}
    events = []

    async def running_names():
        events.append("inspect")
        return set(running)

    async def down(name):
        events.append(f"down:{name}")
        running.discard(name)
        return 0, "stopped"

    monkeypatch.setattr(profile_store, "list_profiles", lambda backend: [p for p in profiles if p.backend == backend])
    monkeypatch.setattr(profile_store, "load_profile", lambda name, backend: next((p for p in profiles if (p.name, p.backend) == (name, backend)), None))
    monkeypatch.setattr(model_switch.docker, "running_container_names", running_names)
    monkeypatch.setattr(model_switch.docker, "running_container_ports", AsyncMock(return_value={"old": "0.0.0.0:8000->8000/tcp"}))
    for backend in ("vllm", "llamacpp"):
        module = importlib.import_module(f"tui.backends.{backend}.backend_runtime")
        monkeypatch.setattr(module, "container_down", down)
    return profiles, running, events


@pytest.mark.asyncio
async def test_switch_stops_only_overlapping_gpu_and_checks_exit(switch_state):
    profiles, running, events = switch_state
    target, sources = await model_switch.plan_switch("llamacpp", "new")
    assert [p.name for p in sources] == ["old"]
    await model_switch.stop_switch_sources(target, sources)
    assert running == {"unrelated"}
    assert events[events.index("down:old") + 1] == "inspect"


@pytest.mark.asyncio
async def test_new_conflict_requires_confirmation_again(switch_state):
    profiles, running, events = switch_state
    target, sources = await model_switch.plan_switch("llamacpp", "new")
    profiles.append(profile_store.StoredProfile(name="surprise", backend="vllm", gpu_id="0"))
    running.add("surprise")
    with pytest.raises(RuntimeError, match="changed"):
        await model_switch.stop_switch_sources(target, sources)
    assert not any(event.startswith("down:") for event in events)


@pytest.mark.asyncio
async def test_external_port_blocks_before_stopping(switch_state, monkeypatch):
    monkeypatch.setattr(model_switch.docker, "running_container_ports", AsyncMock(return_value={"external": "0.0.0.0:8000->8000/tcp"}))
    with pytest.raises(RuntimeError, match="external"):
        await model_switch.plan_switch("llamacpp", "new")
    assert not any(event.startswith("down:") for event in switch_state[2])


@pytest.mark.parametrize("backend", ["vllm", "llamacpp"])
@pytest.mark.parametrize("stop_fails", [False, True])
def test_cli_replace_uses_same_stop_then_start_contract(switch_state, monkeypatch, backend, stop_fails):
    from typer.testing import CliRunner
    from tui.cli import app, container

    profiles, running, events = switch_state
    profiles[0].backend = backend
    monkeypatch.setattr(container, "detect_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(container, "gather_conflict_warnings", AsyncMock(return_value=[]))
    runtime = importlib.import_module(f"tui.backends.{backend}.backend_runtime")

    staged: list = []

    async def start(*args, **kwargs):
        events.append("check:new" if kwargs.get("dry_run") else "up:new")
        if not kwargs.get("dry_run"):
            staged.append(kwargs.get("image_staged"))
        yield ("rc", 0)

    monkeypatch.setattr(runtime, "stream_container_up", start)
    if stop_fails:
        old = importlib.import_module("tui.backends.vllm.backend_runtime")
        monkeypatch.setattr(old, "container_down", AsyncMock(return_value=(1, "failed")))
    result = CliRunner().invoke(app, ["up", "new", "--replace"])
    assert result.exit_code == (1 if stop_fails else 0), result.output
    if stop_fails:
        assert "up:new" not in events
        assert "check:new" in events
    else:
        assert events.index("check:new") < events.index("down:old") < events.index("up:new")
        # The real start reuses the image the precheck pulled/built: no second pull.
        assert staged == [True]


def test_cli_replace_failed_precheck_stops_nothing(switch_state, monkeypatch):
    from typer.testing import CliRunner
    from tui.cli import app, container

    profiles, running, events = switch_state
    monkeypatch.setattr(container, "detect_backend", lambda *args, **kwargs: "llamacpp")
    runtime = importlib.import_module("tui.backends.llamacpp.backend_runtime")

    async def start(*args, **kwargs):
        events.append("check:new" if kwargs.get("dry_run") else "up:new")
        yield ("log", "Error: config is invalid")
        yield ("rc", 1)

    monkeypatch.setattr(runtime, "stream_container_up", start)
    result = CliRunner().invoke(app, ["up", "new", "--replace"])
    assert result.exit_code == 1
    assert "config is invalid" in result.stderr
    assert not any(event.startswith(("down:", "up:")) for event in events)
    assert running == {"old", "unrelated"}


@pytest.mark.asyncio
async def test_all_confirmed_gpu_overlaps_are_stopped(switch_state):
    profiles, running, events = switch_state
    profiles.append(profile_store.StoredProfile(name="second", backend="llamacpp", gpu_id="0", port=8001))
    running.add("second")
    target, sources = await model_switch.plan_switch("llamacpp", "new")
    assert {p.name for p in sources} == {"old", "second"}
    await model_switch.stop_switch_sources(target, sources)
    assert running == {"unrelated"}


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vllm", "llamacpp"])
@pytest.mark.parametrize("outcome", ["confirm", "cancel", "stop-fails", "still-running"])
async def test_tui_switch_confirmation_controls_actual_start(switch_state, monkeypatch, backend, outcome):
    profiles, running, events = switch_state
    profiles[0].backend = backend
    module = importlib.import_module(f"tui.backends.{backend}.screens.container")
    runtime = importlib.import_module(f"tui.backends.{backend}.backend_runtime")
    profile = runtime.Profile(name="new", container_name="new", port=8000, config_name="new", image_tag="custom/server:v1")
    loader = module if backend == "vllm" else module.backend
    monkeypatch.setattr(loader, "load_profile", lambda name: profile)
    monkeypatch.setattr(module.ContainerUpScreen, "on_mount", lambda self: None)
    monkeypatch.setattr(module, "check_port_conflict", AsyncMock(return_value=None))

    async def start(*args, **kwargs):
        if kwargs.get("dry_run"):
            events.append("check:new")
            yield ("rc", 0)
            return
        events.append("up:new")
        staged.append(kwargs.get("image_staged"))
        yield ("rc", 1)

    staged: list = []
    monkeypatch.setattr(module, "stream_container_up", start)
    if outcome in ("stop-fails", "still-running"):
        old = importlib.import_module("tui.backends.vllm.backend_runtime")
        monkeypatch.setattr(old, "container_down", AsyncMock(return_value=(1 if outcome == "stop-fails" else 0, "stop result")))

    app = App()
    async with app.run_test(size=(120, 45)) as pilot:
        screen = module.ContainerUpScreen("new")
        await app.push_screen(screen)
        await pilot.pause()
        assert not any(event.startswith("down:") for event in events)
        screen.action_confirm_start()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmModal)
        worker = screen._start_worker
        screen.action_confirm_start()
        assert screen._start_worker is worker
        assert "old" in app.screen._message
        await pilot.press("n" if outcome == "cancel" else "y")
        await app.workers.wait_for_complete()
        if outcome == "confirm":
            assert events.index("down:old") < events.index("up:new")
            assert "unrelated" in running
            if backend == "vllm":  # its TUI can force a pull; the start must not repeat it
                assert staged == [True]
        else:
            assert "up:new" not in events
        if outcome == "cancel":
            assert "old" in running


@pytest.mark.asyncio
async def test_tui_switch_failed_precheck_stops_nothing(switch_state, monkeypatch):
    profiles, running, events = switch_state
    profiles[0].backend = "vllm"
    module = importlib.import_module("tui.backends.vllm.screens.container")
    runtime = importlib.import_module("tui.backends.vllm.backend_runtime")
    profile = runtime.Profile(name="new", container_name="new", port=8000, config_name="new", image_tag="custom/server:v1")
    monkeypatch.setattr(module, "load_profile", lambda name: profile)
    monkeypatch.setattr(module.ContainerUpScreen, "on_mount", lambda self: None)
    monkeypatch.setattr(module, "check_port_conflict", AsyncMock(return_value=None))

    async def start(*args, **kwargs):
        events.append("check:new" if kwargs.get("dry_run") else "up:new")
        yield ("log", "Error: config is invalid")
        yield ("rc", 1)

    monkeypatch.setattr(module, "stream_container_up", start)
    app = App()
    async with app.run_test(size=(120, 45)) as pilot:
        screen = module.ContainerUpScreen("new")
        await app.push_screen(screen)
        await pilot.pause()
        screen.action_confirm_start()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmModal)
    assert "check:new" in events
    assert not any(event.startswith(("down:", "up:")) for event in events)
    assert running == {"old", "unrelated"}


@pytest.mark.asyncio
async def test_host_process_on_target_port_blocks_before_stopping(switch_state, monkeypatch):
    profiles, running, events = switch_state
    monkeypatch.setattr(model_switch.docker, "running_container_ports", AsyncMock(return_value={}))
    probed: list = []
    monkeypatch.setattr(
        model_switch, "local_port_in_use", lambda port, host: probed.append((port, host)) or True
    )
    with pytest.raises(RuntimeError, match="local process"):
        await model_switch.plan_switch("llamacpp", "new")
    assert probed == [(8000, "127.0.0.1")]
    assert not any(event.startswith("down:") for event in events)


@pytest.mark.asyncio
async def test_port_held_by_a_model_being_replaced_is_exempt(switch_state, monkeypatch):
    monkeypatch.setattr(
        model_switch, "local_port_in_use",
        lambda *_a: (_ for _ in ()).throw(AssertionError("must not probe a port the old model holds")),
    )
    target, sources = await model_switch.plan_switch("llamacpp", "new")
    assert [p.name for p in sources] == ["old"]


def test_local_port_probe_detects_a_listener():
    import socket

    from tui.common.conflicts import local_port_in_use

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    try:
        assert local_port_in_use(listener.getsockname()[1], "127.0.0.1") is True
    finally:
        listener.close()
