from __future__ import annotations

import os
from time import monotonic

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from tui.common import docker as common_docker
from tui.common.adapter import DashboardRow
from tui.common.i18n import lang, t
from tui.common.monitor_render import (
    INTERVAL_STEP,
    MAX_INTERVAL,
    MIN_INTERVAL,
    MonitorState,
    render_dashboard,
)
from tui.common.plain_monitor import sample_entries


class MonitorScreen(Screen):
    BINDINGS = [
        Binding("q", "go_back", t("Back", "뒤로")),
        Binding("escape", "go_back", show=False),
        Binding("p", "pause", t("Pause", "일시정지")),
        Binding("r", "reset_peaks", t("Reset", "초기화")),
        Binding("plus", "faster", t("Interval", "주기"), key_display="+/-"),
        Binding("minus", "slower", show=False),
        Binding("equals_sign", "faster", show=False),
        Binding("l", "toggle_lang", t("Lang", "언어")),
    ]

    DEFAULT_CSS = """
    #mon-scroll { padding: 0 1; }
    """

    def __init__(self, row: DashboardRow | None = None) -> None:
        super().__init__()
        self._focus = row.profile_name if row is not None and row.running else None
        self._states: dict[str, MonitorState] = {}
        self._started = monotonic()
        self._interval = 1.0
        self._paused = False
        self._polling = False
        self._timer = None
        self._rendered = None
        self._last = {"entries": [], "gpus": [], "pcie": {}, "lag": 0.0,
                      "ready": False, "notices": []}

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="mon-scroll"):
            yield Static("", id="mon")
        yield Footer()

    def on_mount(self) -> None:
        self._timer = self.set_interval(self._interval, self._poll)
        self.call_after_refresh(self._poll)

    async def _poll(self) -> None:
        if self._paused or self._polling:
            return
        self._polling = True
        try:
            started = monotonic()
            gpu_notices: list[str] = []
            try:
                gpus = await common_docker.get_gpu_info()
            except RuntimeError as exc:
                gpus = self._last["gpus"]
                gpu_notices.append(
                    t(f"GPU scan failed: {exc}", f"GPU 스캔 실패: {exc}")
                )
            try:
                pcie = await common_docker.get_pcie_stats()
            except RuntimeError as exc:
                pcie = {}
                gpu_notices.append(
                    t(f"PCIe scan failed: {exc}", f"PCIe 스캔 실패: {exc}")
                )
            entries, notices = await sample_entries(
                self._focus, self._states, started, 0.0
            )
            self._last.update(
                entries=entries,
                gpus=gpus,
                pcie=pcie,
                lag=(monotonic() - started) * 1000,
                ready=True,
                notices=gpu_notices + notices,
            )
            self._repaint()
        finally:
            self._polling = False

    def _repaint(self) -> None:
        if not self._last["ready"]:
            return
        self._rendered = render_dashboard(
            self._last["entries"], self._last["gpus"], self._last["pcie"],
            self.size.width or 100, paused=self._paused, interval=self._interval,
            uptime=monotonic() - self._started, lag_ms=self._last["lag"],
            notices=self._last["notices"],
        )
        self.query_one("#mon", Static).update(self._rendered)

    def _restart_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(self._interval, self._poll)

    def action_pause(self) -> None:
        self._paused = not self._paused
        self._repaint()

    def action_reset_peaks(self) -> None:
        for state in self._states.values():
            state.reset_peaks()
        self._repaint()

    def action_faster(self) -> None:
        self._interval = max(MIN_INTERVAL, round(self._interval - INTERVAL_STEP, 2))
        self._restart_timer()
        self._repaint()

    def action_slower(self) -> None:
        self._interval = min(MAX_INTERVAL, round(self._interval + INTERVAL_STEP, 2))
        self._restart_timer()
        self._repaint()

    def action_toggle_lang(self) -> None:
        os.environ["LLMUX_LANG"] = "en" if lang() == "ko" else "ko"
        self._repaint()

    def action_go_back(self) -> None:
        self.dismiss()
