from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import sys
from time import monotonic

from rich.console import Console

from tui.common import docker as common_docker
from tui.common.adapter import DashboardRow
from tui.common.i18n import lang, t
from tui.common.metrics import MetricsSnapshot, MetricsUnavailableError, fetch_snapshot
from tui.common.monitor_render import (
    INTERVAL_STEP,
    MAX_INTERVAL,
    MIN_INTERVAL,
    ModelEntry,
    MonitorState,
    render_dashboard,
)


class TerminalRestoreError(RuntimeError):
    pass


class TerminalSetupError(RuntimeError):
    pass


METRICS_POLL_CONCURRENCY = 4


@dataclass
class MonitorPoll:
    entries: list[ModelEntry]
    gpus: list
    pcie: dict[str, tuple[float, float]]
    lag_ms: float
    notices: list[str]


def _key_ready() -> bool:
    import select

    return bool(select.select([sys.stdin], [], [], 0)[0])


def _toggle_lang() -> None:
    os.environ["LLMUX_LANG"] = "en" if lang() == "ko" else "ko"


async def _running_rows() -> tuple[list[DashboardRow], list[str]]:
    from tui.backends.llamacpp.adapter import LlamacppAdapter
    from tui.backends.vllm.adapter import VllmAdapter

    from tui.common.i18n import t

    errors: list[str] = []
    try:
        running = await common_docker.running_container_names()
    except Exception as exc:
        return [], [t(f"Docker status scan failed: {exc}",
                      f"Docker 상태 스캔 실패: {exc}")]
    rows: list[DashboardRow] = []
    for adapter in (VllmAdapter(), LlamacppAdapter()):
        label = type(adapter).__name__.replace("Adapter", "")
        try:
            rows.extend(adapter.rows(running))
        except Exception as exc:
            errors.append(t(f"{label} scan failed: {exc}",
                            f"{label} 스캔 실패: {exc}"))
    return [r for r in rows if r.running], errors


async def sample_entries(
    focus: str | None,
    states: dict[str, MonitorState],
    _now: float,
    _lag_ms: float,
) -> tuple[list[ModelEntry], list[str]]:
    rows, errors = await _running_rows()
    if focus:
        rows = [r for r in rows if r.profile_name == focus]
    semaphore = asyncio.Semaphore(METRICS_POLL_CONCURRENCY)

    async def fetch(row: DashboardRow) -> tuple[MetricsSnapshot | None, str | None]:
        if not row.port:
            return None, t(
                f"Metrics port unavailable for {row.profile_name}",
                f"{row.profile_name} metrics 포트 확인 불가",
            )
        try:
            async with semaphore:
                snapshot = await fetch_snapshot(row.port)
        except MetricsUnavailableError as exc:
            return None, t(
                f"Metrics scan failed for {row.profile_name}: {exc}",
                f"{row.profile_name} metrics 스캔 실패: {exc}",
            )
        if snapshot is None:
            return None, None
        if snapshot.backend not in ("unknown", row.backend):
            return None, t(
                f"Metrics backend mismatch for {row.profile_name}: "
                f"expected {row.backend}, got {snapshot.backend}",
                f"{row.profile_name} metrics backend 불일치: "
                f"{row.backend} 예상, {snapshot.backend} 수신",
            )
        return snapshot, None

    samples = await asyncio.gather(*(fetch(row) for row in rows))
    sampled_at = monotonic()
    entries: list[ModelEntry] = []
    for row, (snap, error) in zip(rows, samples):
        if error is not None:
            errors.append(error)
        state = states.setdefault(f"{row.backend}:{row.profile_name}", MonitorState())
        entries.append(
            ModelEntry(row, snap, state, state.update(snap, sampled_at, _lag_ms))
        )
    return entries, errors


async def poll_monitor(
    focus: str | None,
    states: dict[str, MonitorState],
    last_gpus: list,
) -> MonitorPoll:
    started = monotonic()
    notices: list[str] = []
    try:
        gpus = await common_docker.get_gpu_info()
    except RuntimeError as exc:
        gpus = last_gpus
        notices.append(t(f"GPU scan failed: {exc}", f"GPU 스캔 실패: {exc}"))
    try:
        pcie = await common_docker.get_pcie_stats()
    except RuntimeError as exc:
        pcie = {}
        notices.append(t(f"PCIe scan failed: {exc}", f"PCIe 스캔 실패: {exc}"))
    entries, model_notices = await sample_entries(
        focus, states, started, 0.0
    )
    return MonitorPoll(
        entries=entries,
        gpus=gpus,
        pcie=pcie,
        lag_ms=(monotonic() - started) * 1000,
        notices=notices + model_notices,
    )


def _stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except Exception:
        return True


def _enable_terminal_input() -> tuple[int | None, object | None]:
    import termios
    import tty

    try:
        fd = sys.stdin.fileno()
    except Exception as exc:
        if not _stdin_is_tty():
            return None, None
        raise TerminalSetupError(
            f"terminal setup failed: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        old_attrs = termios.tcgetattr(fd)
    except Exception as exc:
        if not _stdin_is_tty():
            return None, None
        raise TerminalSetupError(
            f"terminal setup failed: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        tty.setcbreak(fd)
    except Exception as exc:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        except Exception as restore_exc:
            raise TerminalSetupError(
                f"terminal setup failed: {type(exc).__name__}: {exc}; "
                f"terminal restore failed: {type(restore_exc).__name__}: {restore_exc}"
            ) from exc
        raise TerminalSetupError(
            f"terminal setup failed: {type(exc).__name__}: {exc}"
        ) from exc
    return fd, old_attrs


async def run_plain_monitor(focus: str | None = None, interval: float = 1.0) -> None:
    from rich.live import Live

    console = Console()
    states: dict[str, MonitorState] = {}
    started = monotonic()
    paused = False
    fd, old_attrs = _enable_terminal_input()

    last: dict = {"entries": [], "gpus": [], "pcie": {}, "lag": 0.0, "ready": False,
                  "notices": []}
    body_error: BaseException | None = None
    try:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                if not paused:
                    poll = await poll_monitor(focus, states, last["gpus"])
                    last.update(
                        entries=poll.entries,
                        gpus=poll.gpus,
                        pcie=poll.pcie,
                        lag=poll.lag_ms,
                        ready=True,
                        notices=poll.notices,
                    )
                if last["ready"]:
                    live.update(render_dashboard(
                        last["entries"], last["gpus"], last["pcie"], console.size.width,
                        paused=paused, interval=interval,
                        uptime=monotonic() - started, lag_ms=last["lag"],
                        notices=last["notices"],
                    ))
                    live.refresh()

                waited = 0.0
                while waited < interval:
                    await asyncio.sleep(0.1)
                    waited += 0.1
                    if fd is not None and _key_ready():
                        ch = sys.stdin.read(1)
                        if ch in ("q", "Q", "\x03", "\x1b"):
                            return
                        if ch in ("p", "P"):
                            paused = not paused
                        elif ch in ("r", "R"):
                            for st in states.values():
                                st.reset_peaks()
                        elif ch in ("+", "="):
                            interval = max(MIN_INTERVAL, round(interval - INTERVAL_STEP, 2))
                        elif ch in ("-", "_"):
                            interval = min(MAX_INTERVAL, round(interval + INTERVAL_STEP, 2))
                        elif ch in ("l", "L"):
                            _toggle_lang()
                        break
    except KeyboardInterrupt:
        return
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        if old_attrs is not None and fd is not None:
            try:
                import termios

                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception as exc:
                restore_error = TerminalRestoreError(
                    f"terminal restore failed: {type(exc).__name__}: {exc}"
                )
                if body_error is not None:
                    print(str(restore_error), file=sys.stderr)
                else:
                    raise restore_error from exc


async def _resolve_and_run(profile: str | None) -> int:
    if profile:
        rows, errors = await _running_rows()
        if any(r.profile_name == profile for r in rows):
            await run_plain_monitor(profile)
            return 0
        if errors:
            for line in errors:
                print(line, file=sys.stderr)
            print(t(f"Cannot tell whether '{profile}' is running.",
                    f"'{profile}' 의 실행 여부를 확인할 수 없습니다."), file=sys.stderr)
            return 2
        names = ", ".join(r.profile_name for r in rows) or "—"
        print(t(f"'{profile}' is not running. Running: {names}",
                f"'{profile}' 은 실행 중이 아닙니다. 실행 중: {names}"))
        return 1
    await run_plain_monitor(profile)
    return 0


def run_cli(profile: str | None = None) -> int:
    try:
        return asyncio.run(_resolve_and_run(profile))
    except KeyboardInterrupt:
        return 0
    except (TerminalSetupError, TerminalRestoreError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
