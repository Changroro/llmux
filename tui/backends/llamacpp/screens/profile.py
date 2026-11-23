"""Profile CRUD — .env 파일 편집 modal."""

from __future__ import annotations

import re

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, Switch, TextArea

from tui.backends.llamacpp.backend import (
    Profile,
    delete_profile,
    list_config_names,
    resolve_gguf_source,
    load_profile,
    save_profile,
    validate_name,
)
from tui.common import profile_store
from tui.common.i18n import t


def _default_port() -> str:
    """Backend default port, honoring a `defaults:` override in profiles.yaml."""
    from tui.common import profile_store

    return str(profile_store.effective_defaults("llamacpp")["port"])


class ProfileFormScreen(ModalScreen[str | None]):
    """Profile 생성/편집 modal."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("pageup", "scroll_form('up')", "Scroll up", show=False),
        Binding("pagedown", "scroll_form('down')", "Scroll down", show=False),
        Binding("home", "scroll_form('home')", "Scroll to top", show=False),
        Binding("end", "scroll_form('end')", "Scroll to bottom", show=False),
    ]

    DEFAULT_CSS = """
    ProfileFormScreen { align: center middle; }
    ProfileFormScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 78;
        min-width: 55;
        /* No max-height — see vLLM ProfileFormScreen for rationale. */
        height: 95%;
        min-height: 12;
    }
    ProfileFormScreen VerticalScroll { height: 1fr; min-height: 5; }
    ProfileFormScreen #form-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ProfileFormScreen .form-row { height: auto; margin-bottom: 1; }
    ProfileFormScreen .form-row Label {
        width: 22;
        padding: 1 1 0 0;
        color: $text-muted;
    }
    ProfileFormScreen #section-title {
        margin-top: 1;
        text-style: bold;
        color: $text;
        border-top: solid $primary 40%;
        padding-top: 1;
    }
    ProfileFormScreen #section-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    ProfileFormScreen #env-vars-input {
        height: 6;
        margin-bottom: 1;
    }
    ProfileFormScreen .form-buttons {
        height: auto;
        min-height: 3;
        margin-top: 1;
        padding-top: 1;
        align: center middle;
        background: $surface;
        border-top: solid $primary 30%;
    }
    """

    def __init__(self, profile: Profile | None = None) -> None:
        super().__init__()
        self._profile = profile
        self._edit_mode = profile is not None
        self._saved_name: str | None = None
        # In edit mode the name field stays editable: a changed name means a
        # rename, and the original is what tells the two apart at save time.
        self._original_name = profile.name if profile is not None else ""

    def compose(self) -> ComposeResult:
        p = self._profile
        title = (
            t(f"Edit Profile: {p.name}", f"프로필 편집: {p.name}")
            if p
            else t("New Profile", "새 프로필")
        )

        configs = list_config_names()
        config_options: list[tuple[str, str]] = [(name, name) for name in configs]

        # A dangling link (config YAML deleted out from under the profile) has
        # no matching option, so the Select would fall back to BLANK and saving
        # would silently rewrite config_name. Surface it instead.
        if p and p.config_name and p.config_name not in configs:
            config_options.insert(
                0, (t(f"{p.config_name} (missing)", f"{p.config_name} (없음)"), p.config_name)
            )
            configs = [*configs, p.config_name]

        with Vertical():
            yield Static(f"[b]{title}[/b]", id="form-title")

            with VerticalScroll():
                with Horizontal(classes="form-row"):
                    yield Label(t("Profile Name", "프로필 이름"))
                    yield Input(
                        value=p.name if p else "",
                        placeholder="my-profile",
                        id="name-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label(t("Container Name", "컨테이너 이름"))
                    yield Input(
                        value=p.container_name if p else "",
                        placeholder=t("(default: profile name)", "(기본: profile name)"),
                        id="container-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Port")
                    _port = _default_port()
                    yield Input(
                        value=str(p.port) if p else _port,
                        placeholder=_port,
                        id="port-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("GPU ID")
                    yield Input(
                        value=p.gpu_id if p else "0",
                        placeholder="0",
                        id="gpu-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Config")
                    select_kwargs: dict = dict(
                        prompt=t("Select config", "config 선택"),
                        allow_blank=True,
                        id="config-select",
                    )
                    if p and p.config_name and p.config_name in configs:
                        select_kwargs["value"] = p.config_name
                    yield Select(config_options, **select_kwargs)

                with Horizontal(classes="form-row"):
                    yield Label("HF Repo")
                    yield Input(
                        value=p.hf_repo if p else "",
                        placeholder=t(
                            "org/Model-GGUF — required; org/Model-GGUF:Q4_K_M picks the file",
                            "org/Model-GGUF — 필수; org/Model-GGUF:Q4_K_M 이면 파일 자동 선택",
                        ),
                        id="hf-repo-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("HF File")
                    yield Input(
                        value=p.hf_file if p else "",
                        placeholder=t(
                            "model-Q4_K_M.gguf (first shard if split) — blank with :QUANT",
                            "model-Q4_K_M.gguf (분할이면 첫 shard) — :QUANT 쓰면 비움",
                        ),
                        id="hf-file-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label(t("Image Tag", "이미지 태그"))
                    yield Input(
                        value=p.image_tag if p else "",
                        placeholder=t(
                            "(blank = default image) e.g. llamacpp-dev:mtp-clean",
                            "(비우면 기본 이미지) 예: llamacpp-dev:mtp-clean",
                        ),
                        id="image-tag-input",
                    )

                yield Label(
                    t("Environment Variables (one KEY=VALUE per line)", "환경 변수 (한 줄에 KEY=VALUE 하나)"),
                    id="section-title",
                )
                yield TextArea(
                    profile_store.format_env_vars_text(p.env_vars if p else {}),
                    id="env-vars-input",
                )

            with Horizontal(classes="form-buttons"):
                yield Button(t("Save", "저장"), id="save-btn", variant="primary")
                yield Button(t("Close", "닫기"), id="cancel-btn", variant="default")

    @on(Button.Pressed, "#save-btn")
    async def _on_save(self, event: Button.Pressed) -> None:
        name = self.query_one("#name-input", Input).value.strip()
        container = self.query_one("#container-input", Input).value.strip()
        port = self.query_one("#port-input", Input).value.strip()
        gpu_id = self.query_one("#gpu-input", Input).value.strip()
        image_tag = self.query_one("#image-tag-input", Input).value.strip()
        try:
            env_vars = profile_store.parse_env_vars_text(
                self.query_one("#env-vars-input", TextArea).text,
                "llamacpp",
                existing=self._profile.env_vars if self._profile is not None else None,
            )
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        hf_repo = self.query_one("#hf-repo-input", Input).value.strip()
        hf_file = self.query_one("#hf-file-input", Input).value.strip()

        config_select = self.query_one("#config-select", Select)
        config_name = (
            str(config_select.value) if config_select.value != Select.BLANK else ""
        )

        if not name:
            self.notify(t("Profile name is required", "Profile 이름 필수"), severity="error")
            return
        if not validate_name(name):
            self.notify(
                t("Name must be lowercase letters/digits/dashes/underscores", "이름은 소문자/숫자/대시/언더스코어"),
                severity="error",
            )
            return
        # Profile names must be globally unique across both backends (see
        # profile_store.find_name_owner) — container_name defaults to the name.
        renaming = self._edit_mode and name != self._original_name
        if not self._edit_mode or renaming:
            owner = profile_store.find_name_owner(name)
            if owner is not None:
                self.notify(
                    t(
                        f"Profile '{name}' already exists (backend {owner})",
                        f"Profile '{name}' 이미 존재 (backend {owner})",
                    ),
                    severity="error",
                )
                return

        if (not self._edit_mode or renaming) and name == "example":
            # config 링크 기본값이 프로필 이름이라, 'example' 프로필은 tracked
            # example.yaml 에 파라미터를 써버린다.
            self.notify(
                t(
                    "'example' is the tracked template config name — use another name",
                    "'example' 은 tracked 템플릿 config 이름 — 다른 이름을 쓸 것",
                ),
                severity="error",
            )
            return
        if container and not validate_name(container):
            self.notify(t("Container name violates the naming rule", "컨테이너 이름 규칙 위반"), severity="error")
            return
        try:
            port_int = int(port or _default_port())
            if not (1024 <= port_int <= 65535):
                raise ValueError
        except ValueError:
            self.notify(t("Port must be an integer 1024–65535", "Port 는 1024–65535 정수"), severity="error")
            return
        if gpu_id and not re.match(r"^[0-9]+(,[0-9]+)*$", gpu_id):
            self.notify(t("GPU ID must be digits/commas (e.g. 0 or 0,1)", "GPU ID 는 숫자/콤마 (예: 0 또는 0,1)"), severity="error")
            return
        unchanged = (
            self._profile is not None and hf_repo and hf_file
            and (hf_repo, hf_file) == (self._profile.hf_repo, self._profile.hf_file)
        )
        if not unchanged:
            try:
                hf_repo, hf_file, gguf_warning = await resolve_gguf_source(hf_repo, hf_file)
            except ValueError as exc:
                self.notify(str(exc), severity="error", timeout=10)
                return
            if gguf_warning:
                self.notify(gguf_warning, severity="warning", timeout=8)
        from tui.common.dev_build import image_tag_error

        tag_err = image_tag_error(image_tag)
        if tag_err:
            self.notify(tag_err, severity="error")
            return

        if renaming and not await self._rename_allowed():
            return

        p = Profile(
            name=name,
            container_name=container or name,
            port=port_int,
            gpu_id=gpu_id or "0",
            config_name=config_name or name,
            image_tag=image_tag,
            hf_repo=hf_repo,
            hf_file=hf_file,
            env_vars=env_vars,
        )

        try:
            if self._profile is not None and hasattr(
                self._profile, "_stored_snapshot"
            ):
                p._stored_snapshot = self._profile._stored_snapshot
            save_profile(p)
        except (OSError, RuntimeError, ValueError) as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(t(f"Saved: {name}", f"저장: {name}"), severity="information")
        self._saved_name = name
        self._profile = p
        self._original_name = name
        self.query_one("#form-title", Static).update(
            t(f"[b]Edit Profile: {name}[/b]", f"[b]프로필 편집: {name}[/b]")
        )

        if not self._edit_mode:
            self._edit_mode = True

    async def _rename_allowed(self) -> bool:
        from tui.common import docker as common_docker

        old_name = self._original_name
        container = self._profile.container_name or old_name
        try:
            running = await common_docker.running_container_names()
        except Exception as exc:
            self.notify(
                t(
                    f"Could not check running containers ({exc}); rename aborted.",
                    f"실행 중 컨테이너를 확인할 수 없습니다 ({exc}). 이름 변경을 중단합니다.",
                ),
                severity="error",
            )
            return False
        if container in running:
            self.notify(
                t(
                    f"Container '{container}' is running; stop it before renaming.",
                    f"컨테이너 '{container}' 가 실행 중입니다. 중지 후 이름을 변경하세요.",
                ),
                severity="error",
            )
            return False
        return True

    @on(Button.Pressed, "#cancel-btn")
    def _on_close(self, event: Button.Pressed) -> None:
        self.dismiss(self._saved_name)

    def action_cancel(self) -> None:
        self.dismiss(self._saved_name)

    def action_scroll_form(self, direction: str) -> None:
        try:
            scroll = self.query_one(VerticalScroll)
        except Exception:
            return
        if direction == "up":
            scroll.scroll_page_up()
        elif direction == "down":
            scroll.scroll_page_down()
        elif direction == "home":
            scroll.scroll_home()
        elif direction == "end":
            scroll.scroll_end()


class ProfileDeleteScreen(ModalScreen[bool]):
    """Profile 삭제 확인."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ProfileDeleteScreen { align: center middle; }
    ProfileDeleteScreen > Vertical {
        background: $surface;
        border: round $error;
        padding: 1 2;
        width: 60;
        height: auto;
    }
    ProfileDeleteScreen #delete-message {
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ProfileDeleteScreen .form-buttons {
        height: auto;
        margin-top: 1;
        align: center middle;
    }
    ProfileDeleteScreen .form-buttons Button { margin: 0 1; }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self._profile_name = profile_name
        self._profile = load_profile(profile_name)
        self._config_name = self._profile.config_name or profile_name

    def compose(self) -> ComposeResult:
        cfg = self._profile.config_name

        with Vertical():
            if cfg:
                yield Static(
                    t(
                        f"Delete [b]{self._profile_name}[/b]?\n"
                        f"[dim](linked config: '{cfg}')[/dim]",
                        f"[b]{self._profile_name}[/b] 삭제?\n"
                        f"[dim](연결된 config: '{cfg}')[/dim]",
                    ),
                    id="delete-message",
                )
            else:
                yield Static(
                    t(f"Delete [b]{self._profile_name}[/b]?", f"[b]{self._profile_name}[/b] 삭제?"),
                    id="delete-message",
                )
            with Horizontal(classes="form-row"):
                yield Label(
                    t(
                        f"Also delete linked config if unused: {self._config_name}",
                        f"사용 중이 아니면 연결 config도 삭제: {self._config_name}",
                    )
                )
                yield Switch(value=False, id="delete-config-switch")
            with Horizontal(classes="form-buttons"):
                yield Button(t("Delete", "삭제"), id="delete-btn", variant="error")
                yield Button(t("Cancel", "취소"), id="cancel-btn", variant="default")

    @on(Button.Pressed, "#delete-btn")
    async def _on_delete(self, event: Button.Pressed) -> None:
        from tui.common import docker as common_docker

        container = self._profile.container_name or self._profile_name
        try:
            running = await common_docker.running_container_names()
        except Exception as exc:
            self.app.notify(
                t(
                    f"Could not verify container state ({exc}); deletion aborted.",
                    f"컨테이너 상태를 확인할 수 없습니다 ({exc}). 삭제를 중단합니다.",
                ),
                severity="error",
            )
            return
        if container in running:
            self.app.notify(
                t(
                    f"Container '{container}' is running; stop it before deleting.",
                    f"컨테이너 '{container}' 가 실행 중입니다. 중지 후 삭제하세요.",
                ),
                severity="error",
            )
            return
        delete_config = self.query_one("#delete-config-switch", Switch).value
        delete_profile(self._profile_name, delete_config_too=delete_config)
        suffix = (
            t(
                f"; config '{self._config_name}' kept",
                f"; config '{self._config_name}' 유지",
            )
            if not delete_config
            else ""
        )
        self.app.notify(
            t(f"Deleted: {self._profile_name}", f"삭제: {self._profile_name}") + suffix
        )
        self.dismiss(True)

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)
