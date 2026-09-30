"""PrepareScreen — streams a profile's `prepare` run (download + render only)."""

from __future__ import annotations

from contextlib import aclosing

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, RichLog, Static

from tui.common.i18n import t


class PrepareScreen(ModalScreen[None]):
    """Download the model and render runtime files without starting anything."""

    BINDINGS = [
        Binding("escape,q", "close", t("Close", "닫기"), show=True),
        Binding("f", "toggle_follow", t("Follow on/off", "자동 스크롤")),
    ]

    DEFAULT_CSS = """
    PrepareScreen {
        align: center middle;
    }
    PrepareScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        height: 80%;
    }
    PrepareScreen #prepare-title {
        text-style: bold;
        color: $primary;
        width: 100%;
        margin-bottom: 1;
    }
    PrepareScreen #prepare-status {
        height: 1;
    }
    PrepareScreen #prepare-log {
        height: 1fr;
    }
    """

    def __init__(self, profile_name: str, backend: str) -> None:
        super().__init__()
        self.profile_name = profile_name
        self.backend = backend

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(
                t(
                    f"Prepare — {self.profile_name} ({self.backend})   "
                    "[dim](esc/q:close  f:auto-follow)[/dim]",
                    f"준비 — {self.profile_name} ({self.backend})   "
                    "[dim](esc/q:닫기  f:자동 추적)[/dim]",
                ),
                id="prepare-title",
            )
            yield Static(
                t(
                    "[dim]Downloading the model and rendering the runtime files. "
                    "The server is not started.[/dim]",
                    "[dim]모델을 받고 런타임 파일을 렌더합니다. 서버는 시작하지 않습니다.[/dim]",
                ),
                id="prepare-status",
            )
            yield RichLog(
                id="prepare-log",
                highlight=False,
                markup=False,
                wrap=False,
                auto_scroll=True,
                max_lines=5000,
            )

    def on_mount(self) -> None:
        self._run()

    @work(exclusive=True, group="prepare-stream")
    async def _run(self) -> None:
        if self.backend == "vllm":
            from tui.backends.vllm.backend_runtime import stream_container_prepare
        else:
            from tui.backends.llamacpp.backend_runtime import stream_container_prepare

        log = self.query_one("#prepare-log", RichLog)
        rc = -1
        async with aclosing(stream_container_prepare(self.profile_name)) as _owned_stream:
            async for (msg_type, data) in _owned_stream:
                if msg_type == "rc":
                    rc = int(data)
                    continue
                try:
                    log.write(str(data))
                except Exception:
                    return

        try:
            status = self.query_one("#prepare-status", Static)
        except Exception:
            return
        if rc == 0:
            status.update(
                t(
                    f"[green bold]Prepared. Start it with 'u' or `llmux up {self.profile_name}`.[/green bold]",
                    f"[green bold]준비 완료. 'u' 또는 `llmux up {self.profile_name}` 로 시작하세요.[/green bold]",
                )
            )
        else:
            status.update(
                t(f"[red bold]Prepare failed (rc={rc})[/red bold]",
                  f"[red bold]준비 실패 (rc={rc})[/red bold]")
            )

    def action_toggle_follow(self) -> None:
        log = self.query_one("#prepare-log", RichLog)
        log.auto_scroll = not log.auto_scroll
        if log.auto_scroll:
            log.scroll_end(animate=False)

    def action_close(self) -> None:
        self.workers.cancel_group(self, "prepare-stream")
        self.dismiss(None)
