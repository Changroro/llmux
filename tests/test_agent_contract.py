"""Agent-facing CLI contract: stderr routing, exit codes, JSON records."""

from __future__ import annotations

import json

import pytest
import typer

from tui.cli import _runtime as cli_runtime
from tui.common import exit_codes


async def _events(*events):
    for event in events:
        yield event


def test_stream_async_routes_error_lines_to_stderr(capsys):
    rc = cli_runtime.stream_async(
        _events(
            ("log", "Using image: x"),
            ("log", "Error: port busy"),
            ("log", "✗ profile not found"),
            ("log", "⚠ Warning: mismatch"),
            ("rc", 1),
        )
    )
    out, err = capsys.readouterr()
    assert rc == 1
    assert out.splitlines() == ["Using image: x"]
    assert "Error: port busy" in err and "✗ profile not found" in err
    assert "⚠ Warning: mismatch" in err


def test_json_mode_sends_logs_to_stderr_and_one_record_to_stdout(capsys):
    sink: dict = {}
    rc = cli_runtime.stream_async(
        _events(("log", "Using image: x"), ("log", "Error: port busy"), ("rc", exit_codes.CONFLICT)),
        json_out=True,
        sink=sink,
    )
    with pytest.raises(typer.Exit) as stop:
        cli_runtime.finish_stream(rc, json_out=True, record={"profile": "p"}, sink=sink)
    out, err = capsys.readouterr()
    assert stop.value.exit_code == exit_codes.CONFLICT
    assert "Using image: x" in err
    assert json.loads(out) == {
        "profile": "p",
        "ok": False,
        "code": exit_codes.CONFLICT,
        "error": "Error: port busy",
    }


def test_plan_event_is_printed_and_recorded(capsys):
    sink: dict = {}
    rc = cli_runtime.stream_async(
        _events(("plan", {"command": ["docker", "compose", "up"], "image": "img"}), ("rc", 0)),
        sink=sink,
    )
    out, _ = capsys.readouterr()
    assert rc == 0
    assert "docker compose up" in out
    assert sink["plan"]["image"] == "img"


def _snapshot(state: str, health: str = "none"):
    from tui.common import docker

    return docker.ContainerSnapshot(
        name="p",
        lifecycle=docker.ContainerLifecycle(state),
        health=docker.ContainerHealth(health),
        raw_state=state,
        raw_status=state,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshots", "models", "code"),
    [
        ({"p": "running"}, ["m"], exit_codes.OK),
        ({"p": "running"}, [], exit_codes.NOT_READY),
        ({"p": "exited"}, [], exit_codes.ERROR),
        ({}, [], exit_codes.ERROR),
        (RuntimeError("daemon down"), [], exit_codes.UNREACHABLE),
    ],
)
async def test_wait_ready_maps_states_to_exit_codes(monkeypatch, snapshots, models, code):
    from unittest.mock import AsyncMock

    from tui.common import profile_store, readiness

    if isinstance(snapshots, Exception):
        probe = AsyncMock(side_effect=snapshots)
    else:
        probe = AsyncMock(return_value={k: _snapshot(v) for k, v in snapshots.items()})
    monkeypatch.setattr(readiness.docker, "container_snapshots", probe)
    monkeypatch.setattr(
        readiness, "probe_served",
        AsyncMock(return_value=("ready" if models else "loading", models, "")),
    )
    stored = profile_store.StoredProfile(name="p", backend="vllm", port=8000)
    rc, _message, served = await readiness.wait_ready(stored, timeout=0, interval=0.01)
    assert rc == code
    assert served == (models if code == exit_codes.OK else [])


def test_up_conflict_emits_json_error_record(monkeypatch):
    from unittest.mock import AsyncMock

    from typer.testing import CliRunner

    from tui.cli import app, container
    from tui.common import profile_store

    monkeypatch.setattr(container, "detect_backend", lambda *a, **k: "vllm")
    monkeypatch.setattr(
        container.profile_store, "load_profile",
        lambda name, backend: profile_store.StoredProfile(name=name, backend=backend, port=8001),
    )
    monkeypatch.setattr(
        container, "gather_conflict_warnings",
        AsyncMock(return_value=["port conflict (llmux): Port 8001 is occupied"]),
    )
    result = CliRunner().invoke(app, ["up", "p", "--json"])
    assert result.exit_code == exit_codes.CONFLICT
    record = json.loads(result.stdout)
    assert record["ok"] is False and record["code"] == exit_codes.CONFLICT
    assert "Port 8001" in record["error"] and record["port"] == 8001
    assert "Port 8001" in result.stderr


def test_up_dry_run_json_carries_plan(monkeypatch):
    from unittest.mock import AsyncMock

    from typer.testing import CliRunner

    from tui.backends.vllm import backend_runtime
    from tui.cli import app, container
    from tui.common import profile_store

    seen: dict = {}

    async def fake_up(name, **kwargs):
        seen.update(kwargs)
        yield ("log", "Using image: img")
        yield ("plan", {"command": ["docker", "compose", "up", "-d"], "image": "img"})
        yield ("rc", 0)

    monkeypatch.setattr(container, "detect_backend", lambda *a, **k: "vllm")
    monkeypatch.setattr(
        container.profile_store, "load_profile",
        lambda name, backend: profile_store.StoredProfile(name=name, backend=backend, port=8001),
    )
    monkeypatch.setattr(container, "gather_conflict_warnings", AsyncMock(return_value=[]))
    monkeypatch.setattr(backend_runtime, "stream_container_up", fake_up)
    result = CliRunner().invoke(app, ["up", "p", "--dry-run", "--json", "--timeout", "60"])
    assert result.exit_code == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["ok"] is True and record["plan"]["image"] == "img"
    assert seen["dry_run"] is True and seen["ready_timeout"] == 60
