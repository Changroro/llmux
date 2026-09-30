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

    async def start(*args, **kwargs):
        events.append("up:new")
        yield ("rc", 0)

    monkeypatch.setattr(runtime, "stream_container_up", start)
    if stop_fails:
        old = importlib.import_module("tui.backends.vllm.backend_runtime")
        monkeypatch.setattr(old, "container_down", AsyncMock(return_value=(1, "failed")))
    result = CliRunner().invoke(app, ["up", "new", "--replace"])
    assert result.exit_code == (1 if stop_fails else 0), result.output
    if stop_fails:
        assert "up:new" not in events
    else:
        assert events.index("down:old") < events.index("up:new")


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
        events.append("up:new")
        yield ("rc", 1)

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
        else:
            assert "up:new" not in events
        if outcome == "cancel":
            assert "old" in running
