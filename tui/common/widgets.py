"""공통 위젯 — 두 backend 가 공유할 modal 패턴."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static, Switch
from textual.widgets.option_list import Option

from tui.common.dev_build import repo_url_error
from tui.common.i18n import t


async def confirm_model_switch(screen, backend: str, name: str) -> bool:
    from tui.common.model_switch import plan_switch, stop_switch_sources

    try:
        target, sources = await plan_switch(backend, name)
        if not sources:
            return True
        names = ", ".join(f"{p.backend}/{p.name}" for p in sources)
        confirmed = await screen.app.push_screen_wait(ConfirmModal(
            t(
                f"Stop {names} and start {name}?\nActive requests will be interrupted.\nIf the new model fails, the old models stay stopped.",
                f"{names}을(를) 중지하고 {name}을(를) 시작할까요?\n진행 중인 요청이 중단됩니다.\n새 모델 기동이 실패하면 기존 모델은 중지된 상태로 남습니다.",
            ),
            confirm_label=t("Stop and start", "중지 후 시작"),
            variant="warning",
        ))
        if not confirmed:
            return False
        screen.notify(t(f"Stopping {names}…", f"{names} 중지 중…"))
        await stop_switch_sources(target, sources)
        return True
    except (OSError, RuntimeError, ValueError) as exc:
        screen.notify(t(f"Model switch aborted: {exc}", f"모델 교체 중단: {exc}"), severity="error", timeout=10)
        return False


class BackendPickerModal(ModalScreen[str]):
    """새 프로필 생성 시 backend 선택 모달. 반환값: 'vllm' | 'llamacpp' | ''."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("q", "cancel", "Cancel", show=False),
        Binding("1", "pick('vllm')", show=False),
        Binding("2", "pick('llamacpp')", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(t("Select backend", "백엔드 선택"), id="picker-title")
            yield OptionList(
                Option("vLLM", id="vllm"),
                Option("llama.cpp", id="llamacpp"),
                id="picker-list",
            )
            yield Static(
                t(
                    "[dim]1/2 or ↑↓ + Enter · esc to cancel[/dim]",
                    "[dim]1/2 또는 ↑↓ + Enter · esc 취소[/dim]",
                ),
                id="picker-foot",
            )

    def on_mount(self) -> None:
        self.query_one("#picker-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id or "")

    def action_pick(self, backend: str) -> None:
        self.dismiss(backend)

    def action_cancel(self) -> None:
        self.dismiss("")


class ConfirmModal(ModalScreen[bool]):
    """공용 확인 다이얼로그. 두 backend 의 destructive 액션에 재사용."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("y", "confirm", "Yes", show=False),
        Binding("n", "cancel", "No", show=False),
    ]

    def __init__(
        self,
        message: str,
        *,
        confirm_label: str | None = None,
        cancel_label: str | None = None,
        variant: str = "error",
    ) -> None:
        super().__init__()
        self._message = message
        # Resolve translated defaults in compose because default arguments freeze imports.
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label
        self._variant = variant

    def compose(self) -> ComposeResult:
        confirm = self._confirm_label if self._confirm_label is not None else t("Yes", "네")
        cancel = self._cancel_label if self._cancel_label is not None else t("Cancel", "취소")
        yield Vertical(
            Static(self._message, id="confirm-message"),
            Horizontal(
                Button(confirm, id="confirm-yes", variant=self._variant),  # type: ignore[arg-type]
                Button(cancel, id="confirm-no", variant="default"),
                classes="form-buttons",
            ),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class TextPromptModal(ModalScreen[str | None]):
    """공용 텍스트 입력 다이얼로그. 취소 시 None, 확인 시 입력 문자열."""

    DEFAULT_CSS = """
    TextPromptModal > Vertical {
        width: 52;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: round $primary;
    }
    TextPromptModal #prompt-message {
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        message: str,
        *,
        default: str = "",
        placeholder: str = "",
        confirm_label: str | None = None,
        cancel_label: str | None = None,
    ) -> None:
        super().__init__()
        self._message = message
        self._default = default
        self._placeholder = placeholder
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        confirm = self._confirm_label if self._confirm_label is not None else t("OK", "확인")
        cancel = self._cancel_label if self._cancel_label is not None else t("Cancel", "취소")
        yield Vertical(
            Static(self._message, id="prompt-message"),
            Input(
                value=self._default,
                placeholder=self._placeholder,
                id="prompt-input",
            ),
            Horizontal(
                Button(confirm, id="prompt-ok", variant="primary"),
                Button(cancel, id="prompt-cancel", variant="default"),
                classes="form-buttons",
            ),
        )

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prompt-ok":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    def _submit(self) -> None:
        value = self.query_one("#prompt-input", Input).value.strip()
        self.dismiss(value or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class DevBuildPromptModal(ModalScreen[dict[str, object] | None]):
    BINDINGS = [Binding("escape", "cancel", show=False)]

    DEFAULT_CSS = """
    DevBuildPromptModal > Vertical {
        width: 66;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: round $primary;
    }
    DevBuildPromptModal Label { margin-top: 1; color: $text-muted; }
    DevBuildPromptModal .switch-row { height: 3; }
    """

    def __init__(self, backend: str, repo_url: str, branch: str) -> None:
        super().__init__()
        self._backend = backend
        self._repo_url_error = repo_url_error(repo_url)
        self._repo_url = "" if self._repo_url_error else repo_url
        self._branch = branch

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(t("Build Development Image", "개발 이미지 빌드"), id="prompt-message")
            yield Label(t("Repository URL", "저장소 URL"))
            yield Input(value=self._repo_url, id="build-repo-input")
            yield Label(t("Branch", "브랜치"))
            yield Input(value=self._branch, id="build-branch-input")
            yield Label(t("Output tag (optional)", "출력 태그 (선택)"))
            yield Input(id="build-tag-input")
            if self._backend == "vllm":
                with Horizontal(classes="switch-row"):
                    yield Label(t("Use upstream Dockerfile defaults", "upstream Dockerfile 기본값 사용"))
                    yield Switch(id="build-official-switch")
            else:
                yield Label(t("CUDA architectures (optional, e.g. 89 or 86;89)", "CUDA architecture (선택, 예: 89 또는 86;89)"))
                yield Input(id="build-cuda-input")
                with Horizontal(classes="switch-row"):
                    yield Label(t("Build for all architectures", "모든 architecture용 빌드"))
                    yield Switch(id="build-multiarch-switch")
            with Horizontal(classes="form-buttons"):
                yield Button(t("Build", "빌드"), id="build-submit", variant="primary")
                yield Button(t("Cancel", "취소"), id="build-cancel")

    def on_mount(self) -> None:
        self.query_one("#build-repo-input", Input).focus()
        if self._repo_url_error:
            self.notify(self._repo_url_error, severity="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "build-cancel":
            self.dismiss(None)
            return
        repo_input = self.query_one("#build-repo-input", Input)
        repo_url = repo_input.value.strip()
        branch = self.query_one("#build-branch-input", Input).value.strip()
        if not repo_url or not branch:
            self.notify(t("Repository and branch are required", "저장소와 브랜치는 필수입니다"), severity="error")
            return
        error = repo_url_error(repo_url)
        if error:
            repo_input.value = ""
            self.notify(error, severity="error")
            return
        result: dict[str, object] = {
            "repo_url": repo_url,
            "branch": branch,
            "custom_tag": self.query_one("#build-tag-input", Input).value.strip(),
            "official": False,
            "cuda_arch": "",
            "multi_arch": False,
        }
        if self._backend == "vllm":
            result["official"] = self.query_one("#build-official-switch", Switch).value
        else:
            result["cuda_arch"] = self.query_one("#build-cuda-input", Input).value.strip()
            result["multi_arch"] = self.query_one("#build-multiarch-switch", Switch).value
        self.dismiss(result)

    def action_cancel(self) -> None:
        self.dismiss(None)
