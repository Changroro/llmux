from __future__ import annotations

from contextlib import aclosing

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    RichLog,
    Static,
)

from tui.backends.llamacpp import backend
from tui.backends.llamacpp.backend import format_gpu_bar, get_gpu_info
from tui.backends.llamacpp.backend_runtime import (
    LLAMACPP_DEV_SPEC,
    _resolve_runtime_image,
    check_port_conflict,
    get_dev_build_defaults,
    stream_container_up,
)
from tui.common.dev_build import list_local_dev_images
from tui.common.i18n import t


VER_PINNED = "pinned_image"
VER_DEFAULT = "default_image"
VER_DEV = "dev_build"
VER_CUSTOM = "custom_tag"


class ContainerUpScreen(Screen):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("q", "cancel", "Quit", show=False),
        Binding("f", "toggle_follow", t("Follow on/off", "자동 스크롤")),
        Binding("enter", "confirm_start", t("Start", "시작"), priority=True),
    ]

    DEFAULT_CSS = """
    ContainerUpScreen {
        layout: vertical;
    }

    ContainerUpScreen > Vertical {
        width: 100%;
        height: 100%;
        padding: 0 1;
    }

    ContainerUpScreen #title-label {
        text-style: bold;
        color: $primary;
        width: 100%;
        text-align: center;
    }

    ContainerUpScreen #profile-label {
        color: $text-muted;
    }

    ContainerUpScreen #version-label {
        margin-bottom: 0;
        color: $text-muted;
    }

    ContainerUpScreen #version-help {
        color: $text-muted;
        margin-top: 0;
        margin-bottom: 0;
    }

    ContainerUpScreen #version-scroll {
        /* VerticalScroll requires an explicit height to scroll on short terminals. */
        height: 1fr;
        min-height: 8;
    }

    ContainerUpScreen RadioSet {
        height: auto;
    }

    ContainerUpScreen RadioSet > RadioButton.-on > .toggle--button {
        color: #34d399;
        background: $surface;
        text-style: bold;
    }

    ContainerUpScreen RadioSet > RadioButton.-on > .toggle--label {
        color: #34d399;
        text-style: bold;
    }

    ContainerUpScreen #custom-tag-input {
        margin-bottom: 1;
        display: none;
    }

    ContainerUpScreen #dev-build-options {
        display: none;
        height: auto;
        margin-bottom: 1;
    }

    ContainerUpScreen .dev-build-row {
        height: auto;
        margin-bottom: 1;
    }

    ContainerUpScreen .dev-build-row Label {
        width: 10;
        color: $text-muted;
        padding-top: 1;
    }

    ContainerUpScreen .buttons {
        height: 1;
        align: center middle;
        margin-top: 1;
    }

    ContainerUpScreen #startup-area {
        display: none;
        height: 1fr;
    }

    ContainerUpScreen #startup-status {
        height: 1;
        margin-bottom: 0;
    }

    ContainerUpScreen #startup-log {
        height: 1fr;
        margin: 0;
    }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self.profile_name = profile_name
        self._profile = backend.load_profile(profile_name)
        self._dev_repo_url, self._dev_branch = get_dev_build_defaults()
        self._has_local_dev: bool = False
        self._gpu_timer = None
        self._start_worker = None
        self._gpu_probe_error = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-dialog"):
            yield Static(t("Start llama.cpp Container", "llama.cpp 컨테이너 시작"), id="title-label")
            yield Static(t(f"Profile: [b]{self.profile_name}[/b]", f"프로필: [b]{self.profile_name}[/b]"), id="profile-label")
            if not self._profile.config_name:
                yield Static(
                    t(
                        "[yellow]No config linked. A default config will be generated on start.[/yellow]",
                        "[yellow]연결된 config 가 없습니다. 시작 시 기본 config 가 생성됩니다.[/yellow]",
                    )
                )

            pinned = self._profile.image_tag
            if pinned:
                yield Static(
                    t(
                        f"[cyan]Profile pinned to: {pinned}[/cyan]  "
                        "[dim](pick another option to override for this run)[/dim]",
                        f"[cyan]프로필 고정 이미지: {pinned}[/cyan]  "
                        "[dim](이번 실행만 재정의하려면 다른 옵션 선택)[/dim]",
                    ),
                    id="pinned-label",
                )

            with VerticalScroll(id="version-scroll"):
                yield Label(t("Version", "버전"), id="version-label")
                yield Static(
                    t(
                        "[dim]↑↓ to choose · Enter to start · Esc to cancel[/dim]",
                        "[dim]↑↓ 선택 · Enter 로 시작 · Esc 취소[/dim]",
                    ),
                    id="version-help",
                )
                with RadioSet(id="version-radio"):
                    if pinned:
                        yield RadioButton(
                            t(f"Pinned Image  ({pinned})", f"고정 이미지  ({pinned})"), id=VER_PINNED, value=True
                        )
                    yield RadioButton(
                        t(
                            f"Default Image  ({_resolve_runtime_image(self._profile, use_default_image=True)})",
                            f"기본 이미지  ({_resolve_runtime_image(self._profile, use_default_image=True)})",
                        ),
                        id=VER_DEFAULT,
                        value=not pinned,
                    )
                    yield RadioButton(
                        t(
                            "Dev Build  (llamacpp-dev:<branch>)  (loading...)",
                            "개발 빌드  (llamacpp-dev:<branch>)  (불러오는 중...)",
                        ),
                        id=VER_DEV,
                    )
                    yield RadioButton(t("Custom Tag", "커스텀 태그"), id=VER_CUSTOM)
                yield Input(
                    placeholder=t(
                        "Enter custom image tag (e.g. llamacpp-dev:mtp-clean)...",
                        "커스텀 이미지 태그 입력 (예: llamacpp-dev:mtp-clean)...",
                    ),
                    id="custom-tag-input",
                )
                with Vertical(id="dev-build-options"):
                    with Horizontal(classes="dev-build-row"):
                        yield Label(t("Repo URL", "저장소 URL"))
                        yield Input(
                            value=self._dev_repo_url,
                            placeholder="https://github.com/ggml-org/llama.cpp.git",
                            id="dev-repo-input",
                        )
                    with Horizontal(classes="dev-build-row"):
                        yield Label(t("Branch", "브랜치"))
                        yield Input(
                            value=self._dev_branch,
                            placeholder="master",
                            id="dev-branch-input",
                        )
            with Vertical(id="startup-area"):
                yield Static("", id="startup-status")
                yield RichLog(
                    highlight=False,
                    markup=False,
                    wrap=False,
                    auto_scroll=True,
                    max_lines=5000,
                    id="startup-log",
                )
            yield Static("", id="gpu-bar")
            with Horizontal(classes="buttons"):
                yield Button(t("Start", "시작"), variant="primary", id="start-btn")
                yield Button(t("Cancel", "취소"), variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        self._refresh_dev_label()
        self._fetch_gpu_info()
        try:
            self.query_one("#version-radio", RadioSet).focus()
        except Exception:
            pass
        self._gpu_timer = self.set_interval(3, self._fetch_gpu_info)

    @work(exclusive=False)
    async def _fetch_gpu_info(self) -> None:
        try:
            gpus = await get_gpu_info()
        except RuntimeError as exc:
            message = t(f"GPU probe failed: {exc}", f"GPU 조회 실패: {exc}")
            try:
                self.query_one("#gpu-bar", Static).update(f"[red]{message}[/red]")
            except Exception:
                pass
            if message != self._gpu_probe_error:
                self.notify(message, severity="error", timeout=8)
            self._gpu_probe_error = message
            return
        self._gpu_probe_error = ""
        try:
            self.query_one("#gpu-bar", Static).update(format_gpu_bar(gpus))
        except Exception:
            pass

    @work(exclusive=False)
    async def _refresh_dev_label(self) -> None:
        try:
            radio_set = self.query_one("#version-radio", RadioSet)
            btn = radio_set.query_one(f"#{VER_DEV}", RadioButton)
        except Exception:
            return
        try:
            images = await list_local_dev_images(LLAMACPP_DEV_SPEC)
        except RuntimeError as exc:
            btn.label = t("Dev Build (image lookup failed)", "개발 빌드 (이미지 조회 실패)")
            self.notify(str(exc), severity="error")
            return
        self._has_local_dev = bool(images)
        label = (
            t(
                f"Dev Build  ({len(images)} local tag{'s' if len(images) != 1 else ''})",
                f"개발 빌드  (로컬 태그 {len(images)} 개)",
            )
            if images
            else t(
                "Dev Build  (no local tags — will build from source)",
                "개발 빌드  (로컬 태그 없음 — 소스에서 빌드)",
            )
        )
        btn.label = label

    @on(RadioSet.Changed, "#version-radio")
    def _on_version_changed(self, event: RadioSet.Changed) -> None:
        custom_input = self.query_one("#custom-tag-input", Input)
        dev_options = self.query_one("#dev-build-options", Vertical)
        pressed = event.pressed
        if pressed and pressed.id == VER_CUSTOM:
            custom_input.styles.display = "block"
            dev_options.styles.display = "none"
            custom_input.focus()
        elif pressed and pressed.id == VER_DEV:
            custom_input.styles.display = "none"
            dev_options.styles.display = "block"
            self.query_one("#dev-repo-input", Input).focus()
        else:
            custom_input.styles.display = "none"
            dev_options.styles.display = "none"

    def _cleanup(self) -> None:
        if self._gpu_timer is not None:
            self._gpu_timer.stop()
        self.workers.cancel_node(self)

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel(self) -> None:
        self._cleanup()
        self.app.pop_screen()

    def action_cancel(self) -> None:
        self._cleanup()
        self.app.pop_screen()

    def action_confirm_start(self) -> None:
        if self._start_worker is not None and not self._start_worker.is_finished:
            return
        try:
            if str(self.query_one("#version-scroll").styles.display) == "none":
                return
            radio_set = self.query_one("#version-radio", RadioSet)
        except Exception:
            return
        sel = getattr(radio_set, "_selected", None)
        if sel is not None:
            try:
                btn = radio_set._nodes[sel]
            except (IndexError, AttributeError, TypeError):
                btn = None
            if isinstance(btn, RadioButton) and not btn.disabled and not btn.value:
                btn.value = True
        pressed = radio_set.pressed_button
        if pressed is None:
            return
        if pressed.id == VER_CUSTOM:
            custom = self.query_one("#custom-tag-input", Input)
            value = custom.value.strip()
            if not value:
                custom.focus()
                return
            from tui.common.dev_build import image_tag_error

            error = image_tag_error(value)
            if error:
                self.notify(error, severity="error", timeout=8)
                custom.focus()
                return
        if pressed.id == VER_DEV and not self.query_one("#dev-repo-input", Input).value.strip():
            self.query_one("#dev-repo-input", Input).focus()
            return
        self._start_worker = self._do_start()

    @on(Button.Pressed, "#start-btn")
    def _on_start(self) -> None:
        self.action_confirm_start()

    @work(exclusive=True, group="llamacpp-start")
    async def _do_start(self) -> None:
        radio_set = self.query_one("#version-radio", RadioSet)
        pressed = radio_set.pressed_button
        default_id = VER_PINNED if self._profile.image_tag else VER_DEFAULT
        selected_id = pressed.id if pressed else default_id

        use_dev = False
        use_default_image = False
        tag = ""
        repo_url = ""
        branch = ""

        if selected_id == VER_DEV:
            use_dev = True
            repo_url = self.query_one("#dev-repo-input", Input).value.strip()
            branch = self.query_one("#dev-branch-input", Input).value.strip()
            if not repo_url:
                self.app.notify(t("Please enter a repository URL.", "저장소 URL 을 입력하세요."), severity="error")
                return
            if not branch:
                self.app.notify(t("Please enter a branch.", "브랜치를 입력하세요."), severity="error")
                return
        elif selected_id == VER_CUSTOM:
            tag = self.query_one("#custom-tag-input", Input).value.strip()
            if not tag:
                self.app.notify(t("Please enter a custom tag.", "커스텀 태그를 입력하세요."), severity="error")
                return
        elif selected_id != VER_PINNED:
            use_default_image = True

        from tui.common.widgets import confirm_model_switch

        if not await confirm_model_switch(self, "llamacpp", self.profile_name):
            return

        try:
            conflict = await check_port_conflict(self._profile)
        except RuntimeError as exc:
            self.app.notify(
                t(f"Port probe failed: {exc}", f"포트 조회 실패: {exc}"),
                severity="error",
                timeout=8,
            )
            return
        if conflict:
            self.app.notify(
                t(
                    f"Port {self._profile.port} is already used by {conflict}.",
                    f"Port {self._profile.port} 는 이미 {conflict} 이(가) 사용 중입니다.",
                ),
                severity="error",
                timeout=5,
            )
            return

        try:
            self.query_one("#startup-area").styles.display = "block"
            self.query_one("#version-scroll").styles.display = "none"
            self.query_one("#start-btn").styles.display = "none"
            status = self.query_one("#startup-status", Static)
            log_widget = self.query_one("#startup-log", RichLog)
            status.update(t("[bold]Starting container...[/bold]", "[bold]컨테이너 시작 중...[/bold]"))
        except Exception:
            return

        rc = -1
        async with aclosing(stream_container_up(self.profile_name, use_dev=use_dev, use_default_image=use_default_image, tag=tag, repo_url=repo_url, branch=branch)) as _owned_stream:
            async for (msg_type, data) in _owned_stream:
                if msg_type == "log":
                    try:
                        log_widget.write(backend.strip_ansi(data))
                    except Exception:
                        pass
                elif msg_type == "rc":
                    rc = int(data)

        try:
            if rc == 0:
                status.update(
                    t(
                        "[green bold]Container started. Logs: (Esc/q to close)[/green bold]",
                        "[green bold]컨테이너 시작됨. 로그: (Esc/q 로 닫기)[/green bold]",
                    )
                )
                try:
                    async with aclosing(backend.stream_logs(self._profile.container_name)) as _owned_stream:
                        async for line in _owned_stream:
                            log_widget.write(backend.strip_ansi(line))
                except Exception as exc:
                    log_widget.write(t(f"Log stream error: {exc}", f"로그 스트림 오류: {exc}"))
            else:
                status.update(t(f"[red bold]Failed to start (rc={rc})[/red bold]", f"[red bold]시작 실패 (rc={rc})[/red bold]"))
        except Exception:
            pass

    def action_toggle_follow(self) -> None:
        try:
            log_widget = self.query_one("#startup-log", RichLog)
        except Exception:
            return
        log_widget.auto_scroll = not log_widget.auto_scroll
        if log_widget.auto_scroll:
            log_widget.scroll_end(animate=False)
        self.notify(
            t(
                f"auto-follow: {'ON' if log_widget.auto_scroll else 'OFF'}",
                f"자동 추적: {'켜짐' if log_widget.auto_scroll else '꺼짐'}",
            ),
            timeout=2,
        )
