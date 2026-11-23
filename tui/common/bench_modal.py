"""Benchmark options asked before a dashboard bench — the TUI twin of `llmux bench` flags."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from tui.common.bench import KINDS
from tui.common.i18n import t


class BenchOptionsModal(ModalScreen[dict | None]):
    """Returns {"kind", "concurrency", "requests"} (None = flag left blank), or None on cancel."""

    BINDINGS = [Binding("escape", "cancel", show=False)]
    DEFAULT_CSS = """
    BenchOptionsModal { align: center middle; }
    BenchOptionsModal > Vertical {
        background: $surface; border: round $primary; padding: 1 2; width: 64; height: auto;
    }
    BenchOptionsModal Horizontal { height: auto; }
    BenchOptionsModal Label { width: 16; padding-top: 1; }
    BenchOptionsModal Input, BenchOptionsModal Select { width: 1fr; }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self.profile_name = profile_name

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(t(f"[b]Benchmark {self.profile_name}[/b]", f"[b]{self.profile_name} 벤치마크[/b]"))
            with Horizontal():
                yield Label(t("Kind", "종류"))
                yield Select([(k, k) for k in ("auto", *KINDS)], value="auto", allow_blank=False, id="bench-kind")
            with Horizontal():
                yield Label(t("Concurrency", "동시 요청"))
                yield Input(placeholder=t("blank = sequential", "비우면 순차"), id="bench-concurrency")
            with Horizontal():
                yield Label(t("Requests", "요청 수"))
                yield Input(placeholder=t("blank = default", "비우면 기본값"), id="bench-requests")
            with Horizontal():
                yield Button(t("Run", "실행"), variant="primary", id="bench-run")
                yield Button(t("Cancel", "취소"), id="bench-cancel")

    def _count(self, widget_id: str, label: str) -> int | None:
        raw = self.query_one(widget_id, Input).value.strip()
        if not raw:
            return None
        if not raw.isdigit() or int(raw) < 1:
            raise ValueError(t(f"{label} must be a positive integer", f"{label} 은(는) 양의 정수여야 합니다"))
        return int(raw)

    @on(Button.Pressed, "#bench-run")
    def _run(self) -> None:
        try:
            options = {
                "kind": str(self.query_one("#bench-kind", Select).value),
                "concurrency": self._count("#bench-concurrency", "Concurrency"),
                "requests": self._count("#bench-requests", "Requests"),
            }
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self.dismiss(options)

    @on(Button.Pressed, "#bench-cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)
