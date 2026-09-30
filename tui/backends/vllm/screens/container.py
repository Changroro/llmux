"""Container management screens: up, down, and log viewer."""

from __future__ import annotations

from contextlib import aclosing

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Static,
    Label,
    Input,
    RadioSet,
    RadioButton,
    RichLog,
    Header,
)
from textual import work, on

from tui.backends.vllm.backend import (
    check_port_conflict,
    format_gpu_bar,
    get_dev_build_defaults,
    get_gpu_info,
    load_profile,
    stream_container_up,
    stream_container_logs,
    get_local_latest_tag,
    get_dockerhub_release_version,
    get_dockerhub_nightly_date,
)
from tui.common.i18n import t
from tui.common.dev_build import image_tag_error
from tui.backends.vllm.backend_inspect import resolve_vllm_image_ref


VER_PINNED = "pinned_image"
VER_LOCAL = "local_latest"
VER_OFFICIAL = "official"
VER_NIGHTLY = "nightly"
VER_DEV = "dev_build"
VER_CUSTOM = "custom_tag"

# Registry lookups retry on a timer; stop after this many so an unreachable
# DockerHub stops looking like "still loading" forever.
_VERSION_LOOKUP_RETRIES = 3


class ContainerUpScreen(Screen):
    """Full-screen container start and log viewer."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("q", "cancel", "Quit", show=False),
        Binding("f", "toggle_follow", t("Follow on/off", "자동 스크롤")),
        # Enter picks the highlighted version and starts, so you don't have to
        # tab to the Start button. Priority so it fires even while the RadioSet
        # (which would otherwise consume Enter to just toggle) is focused.
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
        /* Fill the available space and scroll when the version options +
           dev-build inputs exceed the viewport. Without an explicit
           `height: 1fr`, VerticalScroll defaults to auto and never
           scrolls on short terminals. */
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
        self._profile = load_profile(profile_name)
        self._gpu_timer = None
        self._local_tag: str = ""
        self._release_version: str = ""
        self._version_retries: int = 0
        self._gpu_error_notified = False
        self._dev_repo_url, self._dev_branch = get_dev_build_defaults()

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-dialog"):
            yield Static(t("Start Container", "컨테이너 시작"), id="title-label")
            yield Static(t(f"Profile: [b]{self.profile_name}[/b]", f"프로필: [b]{self.profile_name}[/b]"), id="profile-label")
            if not self._profile.config_name:
                yield Static(
                    t(
                        "[yellow]No config linked. A default config will be generated on start.[/yellow]",
                        "[yellow]연결된 config 가 없습니다. 시작 시 기본 config 가 생성됩니다.[/yellow]",
                    )
                )
            # A pinned image_tag overrides this selection (except an explicit
            # tag), so surface it as its own radio option that can be selected
            # or deselected knowingly.
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
                        t("Local Latest  (loading...)", "로컬 최신  (불러오는 중...)"),
                        id=VER_LOCAL,
                        value=not pinned,
                    )
                    yield RadioButton(t("Official Release  (loading...)", "공식 릴리스  (불러오는 중...)"), id=VER_OFFICIAL)
                    yield RadioButton(t("Nightly  (loading...)", "나이틀리  (불러오는 중...)"), id=VER_NIGHTLY)
                    yield RadioButton(t("Dev Build  (vllm-dev)", "개발 빌드  (vllm-dev)"), id=VER_DEV)
                    yield RadioButton(t("Custom Tag", "커스텀 태그"), id=VER_CUSTOM)
                yield Input(
                    placeholder=t("Enter custom image tag...", "커스텀 이미지 태그 입력..."),
                    id="custom-tag-input",
                )
                with Vertical(id="dev-build-options"):
                    with Horizontal(classes="dev-build-row"):
                        yield Label(t("Repo URL", "저장소 URL"))
                        yield Input(
                            value=self._dev_repo_url,
                            placeholder="https://github.com/owner/vllm.git",
                            id="dev-repo-input",
                        )
                    with Horizontal(classes="dev-build-row"):
                        yield Label(t("Branch", "브랜치"))
                        yield Input(
                            value=self._dev_branch,
                            placeholder="main",
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
        self._fetch_version_info()
        self._fetch_gpu_info()
        try:
            self.query_one("#version-radio", RadioSet).focus()
        except Exception:
            pass
        self._gpu_timer = self.set_interval(3, self._fetch_gpu_info)

    @work(exclusive=False)
    async def _fetch_version_info(self) -> None:
        try:
            radio_set = self.query_one("#version-radio", RadioSet)
        except Exception:
            return

        local_probe_failed = False
        try:
            local_tag = await get_local_latest_tag()
        except RuntimeError as exc:
            local_probe_failed = True
            local_tag = "none"
            self.notify(t(f"Local image lookup failed: {exc}",
                          f"로컬 이미지 조회 실패: {exc}"),
                        severity="error", timeout=8)
        self._local_tag = "" if local_tag == "none" else local_tag
        try:
            btn = radio_set.query_one(f"#{VER_LOCAL}", RadioButton)
            if local_probe_failed:
                btn.label = t("Local Latest  (lookup failed)", "로컬 최신  (조회 실패)")
                btn.disabled = True
            elif local_tag == "none":
                btn.label = t("Local Latest  (no images)", "로컬 최신  (이미지 없음)")
                btn.disabled = True
            else:
                btn.label = t(f"Local Latest  ({local_tag})", f"로컬 최신  ({local_tag})")
                btn.disabled = False
        except Exception:
            pass

        try:
            release_ver = await get_dockerhub_release_version()
        except RuntimeError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            release_ver = "unknown"
        self._release_version = release_ver if release_ver != "unknown" else ""
        try:
            btn = radio_set.query_one(f"#{VER_OFFICIAL}", RadioButton)
            if self._release_version:
                btn.label = t(f"Official Release  ({self._release_version})", f"공식 릴리스  ({self._release_version})")
            elif self._version_retries >= _VERSION_LOOKUP_RETRIES:
                btn.label = t("Official Release  (DockerHub unreachable)",
                              "공식 릴리스  (DockerHub 조회 실패)")
            else:
                btn.label = t("Official Release  (loading...)", "공식 릴리스  (불러오는 중...)")
            btn.disabled = False
        except Exception:
            pass

        nightly_date = await get_dockerhub_nightly_date()
        try:
            btn = radio_set.query_one(f"#{VER_NIGHTLY}", RadioButton)
            if nightly_date == "unknown":
                btn.label = (
                    t("Nightly  (DockerHub unreachable)", "나이틀리  (DockerHub 조회 실패)")
                    if self._version_retries >= _VERSION_LOOKUP_RETRIES
                    else t("Nightly  (loading...)", "나이틀리  (불러오는 중...)")
                )
            elif nightly_date == "available":
                btn.label = t("Nightly", "나이틀리")
            else:
                btn.label = t(f"Nightly  ({nightly_date})", f"나이틀리  ({nightly_date})")
        except Exception:
            pass

        try:
            # Don't steal the selection from a pinned profile — VER_PINNED is
            # its default and stays valid regardless of what's on DockerHub.
            # With no release info there is nothing to auto-select: picking
            # Nightly here would hand the user an unvetted rolling build while
            # they believe they chose a stable one.
            if (
                not self._local_tag
                and not self._profile.image_tag
                and self._release_version
            ):
                radio_set.query_one(f"#{VER_OFFICIAL}", RadioButton).value = True
        except Exception:
            pass

        if not self._release_version or nightly_date == "unknown":
            if self._version_retries < _VERSION_LOOKUP_RETRIES:
                self._version_retries += 1
                self.set_timer(5, self._fetch_version_info)

    @work(exclusive=False)
    async def _fetch_gpu_info(self) -> None:
        try:
            gpus = await get_gpu_info()
        except RuntimeError as exc:
            message = t(f"GPU lookup failed: {exc}", f"GPU 조회 실패: {exc}")
            try:
                self.query_one("#gpu-bar", Static).update(f"[red]{message}[/red]")
            except Exception:
                pass
            if not self._gpu_error_notified:
                self.notify(message, severity="error", timeout=8)
                self._gpu_error_notified = True
            return
        self._gpu_error_notified = False
        try:
            self.query_one("#gpu-bar", Static).update(format_gpu_bar(gpus))
        except Exception:
            pass

    @on(RadioSet.Changed, "#version-radio")
    def _on_version_changed(self, event: RadioSet.Changed) -> None:
        """Show/hide extra inputs based on the selected startup mode."""
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

    @on(Button.Pressed, "#start-btn")
    def _on_start(self) -> None:
        self.action_confirm_start()

    def action_confirm_start(self) -> None:
        """Enter: commit the highlighted version and start.

        Keyboard navigation in a RadioSet only moves the highlight; the option
        isn't actually chosen until it's pressed. So press the highlighted one
        first (no public highlight accessor, hence `_selected`), then start —
        unless Custom/Dev still need their inputs filled, in which case focus
        the input instead.
        """
        try:
            if str(self.query_one("#version-scroll").styles.display) == "none":
                return  # already started — the version UI is hidden
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
                btn.value = True  # fires RadioSet.Changed → _on_version_changed

        pressed = radio_set.pressed_button
        if pressed is None:
            return
        if pressed.id == VER_CUSTOM:
            custom = self.query_one("#custom-tag-input", Input)
            value = custom.value.strip()
            if not value:
                custom.focus()
                return
            error = image_tag_error(resolve_vllm_image_ref(value))
            if error:
                self.notify(error, severity="error", timeout=8)
                custom.focus()
                return
        if pressed.id == VER_DEV and not self.query_one("#dev-repo-input", Input).value.strip():
            self.query_one("#dev-repo-input", Input).focus()
            return
        self._do_start()

    @work(exclusive=True)
    async def _do_start(self) -> None:
        radio_set = self.query_one("#version-radio", RadioSet)
        pressed = radio_set.pressed_button
        default_id = VER_PINNED if self._profile.image_tag else VER_LOCAL
        selected_id = pressed.id if pressed else default_id

        use_dev = False
        use_default_image = False
        tag = ""
        pull = False
        repo_url = ""
        branch = ""

        if selected_id == VER_PINNED:
            # Empty tag + the profile's own image_tag → runtime's pinned branch.
            pass
        elif selected_id == VER_LOCAL:
            if not self._local_tag:
                self.app.notify(t("No local vLLM image is available.", "사용 가능한 로컬 vLLM 이미지가 없습니다."),
                                severity="error")
                return
            # Defeat any pinned image_tag via use_default_image rather than by
            # passing an explicit tag: the runtime keys its pull policy off tag
            # truthiness, and an explicit tag would flip Local Latest from
            # `--pull never` to `--pull missing`. The user picked from images
            # they already have, so a missing image must stay a hard error.
            use_default_image = True
        elif selected_id == VER_OFFICIAL:
            if not self._release_version:
                try:
                    refreshed = await get_dockerhub_release_version()
                except RuntimeError as exc:
                    self.notify(str(exc), severity="error", timeout=8)
                    return
                if refreshed == "unknown":
                    self.app.notify(
                        t(
                            "Could not determine the latest stable release tag from Docker Hub.",
                            "Docker Hub 에서 최신 안정 릴리스 태그를 확인할 수 없습니다.",
                        ),
                        severity="error",
                    )
                    return
                self._release_version = refreshed
            tag = self._release_version
            # Do NOT force pull: "Official Release" means "use the DockerHub
            # latest-stable tag". If that exact version is already local
            # there is no reason to re-pull. backend_runtime's `--pull
            # missing` policy reuses the local image and only fetches when
            # it's genuinely absent.
            pull = False
        elif selected_id == VER_NIGHTLY:
            tag = "nightly"
            # Nightly is intentionally rolling — always re-check upstream.
            pull = True
        elif selected_id == VER_DEV:
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

        # Always keep the runtime bind check enabled right before compose up.
        try:
            conflict = await check_port_conflict(self._profile)
        except (OSError, RuntimeError, ValueError) as exc:
            self.app.notify(
                t(f"Port conflict probe failed: {exc}", f"포트 충돌 조회 실패: {exc}"),
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
        except Exception:  # Screen may already be dismissed
            return

        # Stream backend startup output in real-time
        rc = -1
        async with aclosing(stream_container_up(self.profile_name, use_dev=use_dev, use_default_image=use_default_image, tag=tag, pull=pull, repo_url=repo_url, branch=branch)) as _owned_stream:
            async for (msg_type, data) in _owned_stream:
                if msg_type == "log":
                    try:
                        log_widget.write(data)
                    except Exception:
                        pass
                elif msg_type == "rc":
                    rc = data

        try:
            if rc == 0:
                status.update(t("[green bold]Container started. Logs: (Esc/q to close)[/green bold]", "[green bold]컨테이너 시작됨. 로그: (Esc/q 로 닫기)[/green bold]"))
                try:
                    async with aclosing(stream_container_logs(self._profile.container_name)) as _owned_stream:
                        async for line in _owned_stream:
                            log_widget.write(line)
                except Exception as exc:
                    log_widget.write(t(f"Log stream error: {exc}", f"로그 스트림 오류: {exc}"))
            else:
                status.update(t(f"[red bold]Failed to start (rc={rc})[/red bold]", f"[red bold]시작 실패 (rc={rc})[/red bold]"))
        except Exception:  # Screen may already be dismissed
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


class LogScreen(Screen):
    """Full-screen log viewer that streams container logs in real-time."""

    DEFAULT_CSS = """
    LogScreen {
        layout: vertical;
    }

    LogScreen #log-header {
        dock: top;
        height: 1;
        color: $text-muted;
        text-style: bold;
        padding: 0 2;
        margin: 1 0;
    }

    LogScreen RichLog {
        height: 1fr;
        margin: 0;
    }
    """

    BINDINGS = [
        Binding("q", "go_back", t("Back", "뒤로")),
        Binding("escape", "go_back", t("Back", "뒤로")),
        Binding("f", "toggle_follow", t("Follow on/off", "자동 스크롤")),
    ]

    def __init__(self, container_name: str) -> None:
        super().__init__()
        self.container_name = container_name

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            t(
                f"Logs: [b]{self.container_name}[/b]  "
                "[dim](q/Esc:back  f:auto-follow  ↑↓/PgUp/PgDn:scroll)[/dim]",
                f"로그: [b]{self.container_name}[/b]  "
                "[dim](q/Esc:뒤로  f:자동 추적  ↑↓/PgUp/PgDn:스크롤)[/dim]",
            ),
            id="log-header",
        )
        yield RichLog(
            highlight=False,
            markup=False,
            wrap=False,
            auto_scroll=True,
            max_lines=5000,
            id="log-view",
        )

    def on_mount(self) -> None:
        self._stream_logs()

    @work(exclusive=True)
    async def _stream_logs(self) -> None:
        """Stream container logs into the RichLog widget."""
        log_widget = self.query_one(RichLog)
        try:
            async with aclosing(stream_container_logs(self.container_name)) as _owned_stream:
                async for line in _owned_stream:
                    log_widget.write(line)
        except Exception as exc:
            log_widget.write(t(f"\nLog stream error: {exc}", f"\n로그 스트림 오류: {exc}"))

    def action_toggle_follow(self) -> None:
        log_widget = self.query_one(RichLog)
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

    def action_go_back(self) -> None:
        self.workers.cancel_node(self)
        self.app.pop_screen()
