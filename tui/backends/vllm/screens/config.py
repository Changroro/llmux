"""Config management screens - form for create/edit and list screen."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.screen import Screen, ModalScreen
from textual.widgets import Button, Static, Label, Input, DataTable, Footer, Header, Switch
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.binding import Binding
from textual.suggester import SuggestFromList
from textual import on

from textual import work

from tui.backends.vllm.backend_common import CONFIG_DIR
from tui.backends.vllm.backend import (
    Config,
    load_config,
    save_config,
    list_config_names,
    list_profile_names,
    load_profile,
    validate_name as _validate_name,
    extract_vllm_params,
    format_config_param_value,
    parse_config_param_value,
)
from tui.common import profile_store
from tui.common.flag_discovery import config_target
from tui.common.i18n import t
from tui.common.widgets import TextPromptModal



class ConfigFormScreen(ModalScreen[str | None]):
    """Modal form for creating or editing a config."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        # Scroll the modal's content from anywhere — Input widgets capture
        # the arrow keys, so PgUp/PgDn (which they don't capture) are the
        # only reliable way to reach a row that's hidden below the fold on
        # a short terminal. Home/End jump to the form's top/bottom.
        Binding("pageup", "scroll_form('up')", "Scroll up", show=False),
        Binding("pagedown", "scroll_form('down')", "Scroll down", show=False),
        Binding("home", "scroll_form('home')", "Scroll to top", show=False),
        Binding("end", "scroll_form('end')", "Scroll to bottom", show=False),
    ]

    DEFAULT_CSS = """
    ConfigFormScreen {
        align: center middle;
    }
    ConfigFormScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 80;
        min-width: 50;
        /* No `max-height` — modal fills the terminal up to 95% so the
           inner VerticalScroll always has room to actually scroll on
           short terminals. The previous cap left the outer modal smaller
           than the content while the inner scroller had no slack. */
        height: 95%;
        min-height: 12;
    }
    ConfigFormScreen #form-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        width: 100%;
        height: 1;
        margin-bottom: 1;
    }
    ConfigFormScreen .form-row {
        height: auto;
        margin-bottom: 1;
    }
    ConfigFormScreen .form-row Label {
        width: 24;
        padding: 1 1 0 0;
        color: $text-muted;
    }
    ConfigFormScreen #params-title {
        margin-top: 1;
        text-style: bold;
        color: $text;
        border-top: solid $primary 40%;
        padding-top: 1;
    }
    ConfigFormScreen #params-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    ConfigFormScreen #params-container {
        height: auto;
    }
    ConfigFormScreen .param-row {
        height: auto;
        margin-bottom: 0;
    }
    ConfigFormScreen .param-row .param-switch {
        width: 8;
        margin-right: 1;
    }
    ConfigFormScreen .param-row .param-key {
        width: 28;
        margin-right: 1;
    }
    ConfigFormScreen .param-row .param-value {
        width: 1fr;
        margin-right: 1;
    }
    ConfigFormScreen .param-row.-disabled .param-key,
    ConfigFormScreen .param-row.-disabled .param-value {
        color: $text-muted;
    }
    ConfigFormScreen .param-row .param-remove {
        min-width: 5;
        width: 5;
        background: $error 20%;
        border: none;
        color: $error;
    }
    ConfigFormScreen VerticalScroll {
        height: 1fr;
        min-height: 3;
    }
    ConfigFormScreen .form-buttons {
        height: auto;
        min-height: 3;
        margin-top: 1;
        padding-top: 1;
        align: center middle;
        background: $surface;
        border-top: solid $primary 30%;
    }
    """

    def __init__(self, config_name: str = "", *, profile_name: str = "") -> None:
        super().__init__()
        self._config_name = config_name
        self._profile_name = profile_name
        self._edit_mode = bool(config_name)
        # The name field stays editable in edit mode: a changed name is a
        # rename, and _original_name is what tells the two apart at save time.
        self._original_name = config_name
        self._param_counter = 0
        self._initial_config: Config | None = None
        self._initial_text: str | None = None
        self._saved_name: str | None = None
        self._known_params: set[str] = set()
        self._param_suggester = SuggestFromList(
            sorted(self._known_params), case_sensitive=False
        )

    def compose(self) -> ComposeResult:
        if self._edit_mode:
            path = CONFIG_DIR / f"{self._config_name}.yaml"
            with profile_store.storage_transaction():
                if not path.exists():
                    raise ValueError(f"config not found: {path}")
                self._initial_text = path.read_text()
                self._initial_config = load_config(self._config_name)
        cfg = self._initial_config

        title = (
            t(f"Edit Config: {self._config_name}", f"Config 편집: {self._config_name}")
            if self._edit_mode
            else t("New Config", "새 Config")
        )

        with Vertical():
            yield Static(f"[b]{title}[/b]", id="form-title")

            with VerticalScroll():
                with Horizontal(classes="form-row"):
                    yield Label(t("Config Name", "Config 이름"))
                    yield Input(
                        value=cfg.name if cfg else "",
                        placeholder="my-config",
                        id="name-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label(t("Model", "모델"))
                    yield Input(
                        value=cfg.model if cfg else "",
                        placeholder="org/model-name",
                        id="model-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label(t("Recipe source (optional)", "레시피 출처 (선택)"))
                    yield Input(
                        placeholder=t("blank = the model above", "비우면 위 모델 사용"),
                        id="recipe-model-input",
                    )
                yield Button(
                    t("📋 Fetch vLLM recipe", "📋 vLLM 레시피 받아오기"),
                    id="fetch-recipe-btn",
                    variant="default",
                )
                yield Static("", id="recipe-status")

                with Horizontal(classes="form-row"):
                    yield Label(t("GPU Memory Utilization", "GPU 메모리 사용률"))
                    yield Input(
                        value=str(cfg.gpu_memory_utilization) if cfg else "",
                        placeholder="0.9",
                        id="gpu-mem-input",
                    )

                yield Static(t("vLLM Parameters", "vLLM 파라미터"), id="params-title")
                yield Static(
                    "[dim]max-model-len, dtype, trust-remote-code, ...[/dim]",
                    id="params-hint",
                )

                yield Vertical(id="params-container")
                yield Button(
                    t("+ Add Parameter", "+ 파라미터 추가"),
                    id="add-param-btn",
                    variant="default",
                )

            with Horizontal(classes="form-buttons"):
                yield Button(t("Save", "저장"), id="save-btn", variant="primary")
                yield Button(t("Close", "닫기"), id="cancel-btn", variant="default")

    def on_mount(self) -> None:
        if self._edit_mode and self._initial_config:
            for key, value in self._initial_config.extra_params.items():
                self._add_param_row(key, format_config_param_value(value))
            for key, value in self._initial_config.disabled_params.items():
                self._add_param_row(key, format_config_param_value(value), enabled=False)
        self._load_vllm_params()

    @work(exclusive=False)
    async def _load_vllm_params(self) -> None:
        try:
            image, container = await config_target("vllm", self._config_name, self._profile_name)
            extracted = await extract_vllm_params(image, **({"container_name": container} if container else {}))
            if not extracted:
                raise RuntimeError("image returned no supported parameters")
        except (OSError, RuntimeError, ValueError) as exc:
            self.query_one("#params-hint", Static).update(t("Flag lookup failed; autocomplete unavailable.", "인자 조회 실패: 자동완성을 사용할 수 없습니다."))
            self.notify(
                t(
                    f"Could not inspect vLLM flags: {exc}",
                    f"vLLM 플래그를 확인할 수 없습니다: {exc}",
                ),
                severity="error",
            )
            return
        if extracted:
            self.query_one("#params-hint", Static).update(t(f"Parameters: {container or image or 'next startup image'}", f"인자 기준: {container or image or '다음 기동 이미지'}"))
            self._known_params = set(extracted)
            self._param_suggester = SuggestFromList(
                sorted(self._known_params), case_sensitive=False
            )
            for inp in self.query(".param-key"):
                inp.suggester = self._param_suggester

    def _add_param_row(self, key: str = "", value: str = "", enabled: bool = True) -> None:
        container = self.query_one("#params-container")
        row_id = f"param-row-{self._param_counter}"
        self._param_counter += 1
        # A row's Switch decides active (params) vs disabled (comment marker) at
        # save time; a disabled row is dimmed via the `-disabled` class.
        row = Horizontal(
            Switch(value=enabled, classes="param-switch"),
            Input(
                value=key,
                placeholder=t("param-name (Tab: autocomplete)", "param-name (Tab: 자동완성)"),
                suggester=self._param_suggester,
                classes="param-key",
            ),
            Input(
                value=value,
                placeholder=t("value", "값"),
                classes="param-value",
            ),
            Button("x", classes="param-remove"),
            id=row_id,
            classes="param-row" if enabled else "param-row -disabled",
        )
        container.mount(row)
        self.call_after_refresh(self._scroll_to_bottom)

    @on(Switch.Changed, ".param-switch")
    def _on_switch(self, event: Switch.Changed) -> None:
        row = event.switch.parent
        if row is not None:
            row.set_class(not event.value, "-disabled")

    def _scroll_to_bottom(self) -> None:
        try:
            scroll = self.query_one(VerticalScroll)
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    @on(Button.Pressed, "#fetch-recipe-btn")
    def _on_fetch_recipe(self, event: Button.Pressed) -> None:
        event.stop()
        model = self.query_one("#model-input", Input).value.strip()
        source = self.query_one("#recipe-model-input", Input).value.strip() or model
        if not model or "/" not in model:
            self.notify(
                t("Enter a HuggingFace model id (org/name) first",
                  "먼저 HuggingFace 모델 id (org/name) 를 입력하세요"),
                severity="warning",
            )
            return
        if "/" not in source:
            self.notify(
                t("The recipe source model must be an org/name id",
                  "레시피 출처 모델은 org/name 형식이어야 합니다"),
                severity="warning",
            )
            return
        self._fetch_recipe(source, model)

    @work(exclusive=True, group="recipe-fetch")
    async def _fetch_recipe(self, source_model: str, target_model: str) -> None:
        from tui.common.docker import get_gpu_info
        from tui.common.recipes import RecipeUnavailable, fetch_recipe

        status = self.query_one("#recipe-status", Static)
        status.update(t(f"[dim]Fetching recipe for {source_model}…[/dim]",
                        f"[dim]{source_model} 레시피 받는 중…[/dim]"))
        try:
            recipe = await fetch_recipe(source_model)
        except RecipeUnavailable as exc:
            status.update(t(f"[red]Could not reach the recipe index: {exc}[/red]",
                            f"[red]레시피 조회 실패: {exc}[/red]"))
            return
        if recipe is None:
            status.update(t(f"[yellow]No vLLM recipe found for {source_model}[/yellow]",
                            f"[yellow]{source_model} 의 vLLM 레시피 없음[/yellow]"))
            return
        status.update("")

        try:
            gpus = await get_gpu_info()
            gpu_total_gb = None
            if gpus:
                gpu_total_gb = max(int(g.memory_total) / 1024 for g in gpus)
        except (RuntimeError, TypeError, ValueError) as exc:
            status.update(
                t(
                    f"[red]Could not read GPU memory: {exc}[/red]",
                    f"[red]GPU 메모리 조회 실패: {exc}[/red]",
                )
            )
            return

        from tui.backends.vllm.backend import get_local_latest_tag

        local_tag = await get_local_latest_tag()
        local_ver = "" if local_tag == "none" else local_tag

        from tui.backends.vllm.screens.recipe import RecipeReviewScreen

        def after(result: dict | None) -> None:
            if result:
                self._apply_recipe(result["model_id"], dict(result["params"]))

        self.app.push_screen(
            RecipeReviewScreen(
                recipe,
                gpu_total_gb=gpu_total_gb,
                local_vllm_version=local_ver,
                target_model=target_model if target_model != source_model else "",
            ),
            after,
        )

    def _apply_recipe(self, model_id: str, params: dict[str, Any]) -> None:
        """Merge a fetched recipe into the open form (nothing is saved yet).

        An existing row for the same key is updated and re-enabled so a recipe
        can't leave a stale value behind as a disabled marker.
        """
        self.query_one("#model-input", Input).value = model_id

        rows: dict[str, Any] = {}
        for row in self.query(".param-row"):
            key = row.query_one(".param-key", Input).value.strip()
            if key:
                rows[key] = row

        updated, added = 0, 0
        for key, value in params.items():
            text = format_config_param_value(value)
            row = rows.get(key)
            if row is not None:
                row.query_one(".param-value", Input).value = text
                row.query_one(".param-switch", Switch).value = True
                updated += 1
            else:
                self._add_param_row(key, text)
                added += 1

        self.notify(
            t(
                f"Recipe applied: {added} added, {updated} updated — review, then Save.",
                f"레시피 적용: {added}개 추가, {updated}개 갱신 — 확인 후 저장하세요.",
            ),
            severity="information",
            timeout=6,
        )

    @on(Button.Pressed, "#add-param-btn")
    def _on_add_param(self, event: Button.Pressed) -> None:
        self._add_param_row()

    @on(Button.Pressed, ".param-remove")
    def _on_remove_param(self, event: Button.Pressed) -> None:
        widget = event.button.parent
        while widget is not None:
            if hasattr(widget, "classes") and "param-row" in widget.classes:
                widget.remove()
                return
            widget = widget.parent

    @on(Button.Pressed, "#save-btn")
    async def _on_save(self, event: Button.Pressed) -> None:
        name = self.query_one("#name-input", Input).value.strip()
        model = self.query_one("#model-input", Input).value.strip()
        gpu_mem = self.query_one("#gpu-mem-input", Input).value.strip()

        if not name:
            self.notify(t("Config name is required.", "Config 이름은 필수입니다."),
                        severity="error")
            return
        if not _validate_name(name):
            self.notify(
                t(
                    "Name must be lowercase: start with [a-z0-9], then lowercase letters, digits, dashes, or underscores only.",
                    "이름은 소문자여야 합니다: [a-z0-9] 로 시작하고 소문자·숫자·대시·언더스코어만 사용하세요.",
                ),
                severity="error",
            )
            return
        if gpu_mem:
            try:
                gpu_mem_val = float(gpu_mem)
                if not (0.0 < gpu_mem_val <= 1.0):
                    raise ValueError
            except ValueError:
                self.notify(
                    t(
                        "GPU Memory Utilization must be between 0.0 and 1.0",
                        "GPU 메모리 사용률은 0.0 ~ 1.0 사이여야 합니다.",
                    ),
                    severity="error",
                )
                return

        extra_params: dict[str, Any] = {}
        disabled_params: dict[str, Any] = {}
        seen_keys: set[str] = set()
        for row in self.query(".param-row"):
            key_input = row.query_one(".param-key", Input)
            value_input = row.query_one(".param-value", Input)
            switch = row.query_one(".param-switch", Switch)
            k = key_input.value.strip()
            v = value_input.value.strip()
            if k:
                # Duplicate check spans active + disabled — the same key can't
                # exist in both, and 'disabled' wins nothing at save time.
                if k in seen_keys:
                    self.notify(t(f"Duplicate parameter: {k}", f"중복 파라미터: {k}"),
                                severity="error")
                    return
                seen_keys.add(k)
                try:
                    parsed = parse_config_param_value(v)
                except Exception as exc:
                    self.notify(t(f"Invalid value for {k}: {exc}", f"'{k}' 값이 올바르지 않습니다: {exc}"),
                                severity="error")
                    return
                (extra_params if switch.value else disabled_params)[k] = parsed

        unknown = [k for k in extra_params if k not in self._known_params] if self._known_params else []
        if unknown:
            self.notify(
                t(
                    f"Unknown params (may be valid for your vLLM version): {', '.join(unknown)}",
                    f"알 수 없는 파라미터 (vLLM 버전에 따라 유효할 수 있음): {', '.join(unknown)}",
                ),
                severity="warning",
                timeout=6,
            )

        cfg = Config(
            name=name,
            model=model,
            gpu_memory_utilization=gpu_mem or "0.9",
            extra_params=extra_params,
            disabled_params=disabled_params,
        )

        renaming = self._edit_mode and name != self._original_name
        if renaming:
            if not await self._rename_to(name, cfg):
                return
        else:
            try:
                path = CONFIG_DIR / f"{name}.yaml"
                with profile_store.storage_transaction():
                    if self._edit_mode:
                        if not path.exists():
                            raise ValueError(f"config not found: {path}")
                        if path.read_text() != self._initial_text:
                            raise ValueError(
                                f"config changed since it was opened: {path}; "
                                "reload before saving"
                            )
                    elif path.exists():
                        raise ValueError(f"config already exists: {path}")
                    save_config(cfg)
                    self._initial_text = path.read_text()
            except (OSError, RuntimeError, ValueError) as exc:
                self.notify(str(exc), severity="error", timeout=8)
                return
        self.notify(t(f"Saved: {name}", f"저장됨: {name}"), severity="information")
        self._saved_name = name

        if not self._edit_mode:
            self._edit_mode = True
            self._config_name = name
            self._original_name = name
            self.query_one("#form-title", Static).update(
                t(f"[b]Edit Config: {name}[/b]", f"[b]Config 편집: {name}[/b]")
            )

    async def _rename_to(self, new_name: str, config: Config) -> bool:
        from tui.common import config_store, docker as common_docker
        from tui.backends.vllm.backend_storage import serialize_config

        old_name = self._original_name
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
        for prof in config_store.referencing_profiles("vllm", old_name):
            container = prof.container_name or prof.name
            if container in running:
                self.notify(
                    t(
                        f"Container '{container}' is running; stop it before renaming.",
                        f"컨테이너 '{container}' 가 실행 중입니다. 중지 후 이름을 변경하세요.",
                    ),
                    severity="error",
                )
                return False
        try:
            with profile_store.storage_transaction():
                config_store.rename_config(
                    "vllm",
                    old_name,
                    new_name,
                    replacement=lambda source: serialize_config(config, source),
                    expected_text=self._initial_text,
                )
                self._initial_text = (CONFIG_DIR / f"{new_name}.yaml").read_text()
        except (OSError, RuntimeError, ValueError) as exc:
            self.notify(str(exc), severity="error")
            return False

        self._original_name = new_name
        self._config_name = new_name
        self.query_one("#form-title", Static).update(
            t(f"[b]Edit Config: {new_name}[/b]", f"[b]Config 편집: {new_name}[/b]")
        )
        return True

    @on(Button.Pressed, "#cancel-btn")
    def _on_close(self, event: Button.Pressed) -> None:
        self.dismiss(self._saved_name)

    def action_cancel(self) -> None:
        self.dismiss(self._saved_name)

    def action_scroll_form(self, direction: str) -> None:
        """Scroll the inner VerticalScroll. PgUp/PgDn/Home/End route here
        because the bindings on the screen don't reach the scroller when
        an Input child has focus and swallows arrow keys."""
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


class ConfirmDeleteConfigScreen(ModalScreen[bool]):
    """Confirmation modal before deleting a config."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ConfirmDeleteConfigScreen {
        align: center middle;
    }
    ConfirmDeleteConfigScreen > Vertical {
        background: $surface;
        border: round $error;
        padding: 1 2;
        width: 50;
        height: auto;
    }
    ConfirmDeleteConfigScreen #confirm-title {
        text-style: bold;
        color: $error;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ConfirmDeleteConfigScreen #confirm-msg {
        text-align: center;
        margin-bottom: 1;
    }
    ConfirmDeleteConfigScreen #confirm-warn {
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }
    ConfirmDeleteConfigScreen .confirm-buttons {
        height: auto;
        align: center middle;
    }
    ConfirmDeleteConfigScreen .confirm-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, config_name: str, referencing_profiles: list[str]) -> None:
        super().__init__()
        self._config_name = config_name
        self._referencing = referencing_profiles

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(t("Delete Config", "Config 삭제"), id="confirm-title")
            yield Static(
                t(
                    f"Are you sure you want to delete [b]{self._config_name}[/b]?",
                    f"[b]{self._config_name}[/b] 을(를) 삭제할까요?",
                ),
                id="confirm-msg",
            )
            if self._referencing:
                names = ", ".join(self._referencing)
                yield Static(
                    t(
                        f"[b]Warning:[/b] Used by profiles: {names}",
                        f"[b]경고:[/b] 사용 중인 프로필: {names}",
                    ),
                    id="confirm-warn",
                )
            with Horizontal(classes="confirm-buttons"):
                yield Button(t("Delete", "삭제"), id="confirm-yes", variant="error")
                yield Button(t("Cancel", "취소"), id="confirm-no", variant="default")

    @on(Button.Pressed, "#confirm-yes")
    def _on_yes(self, event: Button.Pressed) -> None:
        from tui.common import config_store

        try:
            config_store.delete_config("vllm", self._config_name)
        except (OSError, RuntimeError, ValueError) as exc:
            self.app.notify(str(exc), severity="error", timeout=8)
            return
        self.app.notify(t(f"Deleted config: {self._config_name}", f"Config 삭제됨: {self._config_name}"))
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def _on_no(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class ConfigListScreen(Screen):
    """Full screen listing all configs in a DataTable."""

    BINDINGS = [
        Binding("n", "new_config", t("New", "새로"), show=True),
        Binding("e", "edit_config", t("Edit", "편집"), show=True),
        Binding("c", "clone_config", t("Clone", "복제"), show=True),
        Binding("R", "rename_config", t("Rename", "이름변경"), show=True),
        Binding("delete,x", "delete_config", t("Delete", "삭제"), show=True),
        Binding("r", "refresh", t("Refresh", "새로고침"), show=True),
        Binding("escape", "go_back", t("Back", "뒤로"), show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(t("[b]Configs[/b]", "[b]Config 목록[/b]"), id="config-title")
        yield DataTable(id="config-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#config-table", DataTable)
        table.add_columns("Name", "Model", "GPU Mem", "Params")
        self._refresh_table()

    def on_screen_resume(self) -> None:
        # Named SCREENS entries are cached instances — on_mount fires once, so
        # without this the list goes stale after configs change elsewhere.
        self._refresh_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.action_edit_config()

    def _refresh_table(self) -> None:
        table = self.query_one("#config-table", DataTable)
        table.clear()
        for name in list_config_names():
            cfg = load_config(name)
            model_short = cfg.model.split("/")[-1] if cfg.model else ""
            param_count = str(len(cfg.extra_params) + 2)
            table.add_row(cfg.name, model_short, str(cfg.gpu_memory_utilization), param_count, key=cfg.name)

    def _get_selected_config(self) -> str | None:
        table = self.query_one("#config-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            return row_key.value
        except Exception:
            return None


    def action_new_config(self) -> None:
        self.app.push_screen(ConfigFormScreen(), callback=self._on_form_closed)

    def action_edit_config(self) -> None:
        name = self._get_selected_config()
        if name is None:
            self.notify(t("No config selected.", "선택된 config 가 없습니다."), severity="warning")
            return
        self.app.push_screen(ConfigFormScreen(config_name=name), callback=self._on_form_closed)

    def action_clone_config(self) -> None:
        name = self._get_selected_config()
        if name is None:
            self.notify(t("No config selected.", "선택된 config 가 없습니다."), severity="warning")
            return

        existing = set(list_config_names())
        default = f"{name}-copy"
        suffix = 2
        while default in existing:
            default = f"{name}-copy-{suffix}"
            suffix += 1

        def after(new_name: str | None) -> None:
            if not new_name:
                return
            if not _validate_name(new_name):
                self.notify(
                    t(
                        "Name must be lowercase: start with [a-z0-9], then lowercase "
                        "letters, digits, dashes, or underscores only.",
                        "이름은 소문자여야 합니다: [a-z0-9] 로 시작하고 소문자·숫자·대시·언더스코어만 사용하세요.",
                    ),
                    severity="error",
                )
                return
            try:
                with profile_store.storage_transaction():
                    source = CONFIG_DIR / f"{name}.yaml"
                    destination = CONFIG_DIR / f"{new_name}.yaml"
                    if not source.exists():
                        raise ValueError(f"config not found: {source}")
                    if destination.exists():
                        raise ValueError(f"config already exists: {destination}")
                    profile_store._atomic_write(destination, source.read_text())
            except (OSError, RuntimeError, ValueError) as exc:
                self.notify(str(exc), severity="error", timeout=8)
                return
            self._refresh_table()
            self.notify(t(f"Cloned '{name}' → '{new_name}'", f"복제됨: '{name}' → '{new_name}'"))

        self.app.push_screen(
            TextPromptModal(
                t(f"Clone config '{name}' as:", f"Config '{name}' 을(를) 다음 이름으로 복제:"),
                default=default,
            ),
            callback=after,
        )

    def action_rename_config(self) -> None:
        name = self._get_selected_config()
        if name is None:
            self.notify(t("No config selected.", "선택된 config 가 없습니다."), severity="warning")
            return

        def after(new_name: str | None) -> None:
            if not new_name or new_name == name:
                return
            self._rename_config(name, new_name)

        self.app.push_screen(
            TextPromptModal(
                t(f"Rename config '{name}' to:", f"Config '{name}' 의 새 이름:"),
                default=name,
            ),
            callback=after,
        )

    @work(exclusive=True, group="config-rename")
    async def _rename_config(self, old: str, new: str) -> None:
        from tui.common import config_store, docker as common_docker

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
            return
        for p in config_store.referencing_profiles("vllm", old):
            container = p.container_name or p.name
            if container in running:
                self.notify(
                    t(
                        f"Container '{container}' is running; stop it before renaming.",
                        f"컨테이너 '{container}' 가 실행 중입니다. 중지 후 이름을 변경하세요.",
                    ),
                    severity="error",
                )
                return
        try:
            updated = config_store.rename_config("vllm", old, new)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self._refresh_table()
        suffix = f" ({len(updated)} profile(s) repointed)" if updated else ""
        self.notify(
            t(f"Renamed '{old}' → '{new}'{suffix}", f"이름 변경: '{old}' → '{new}'{suffix}")
        )

    def action_delete_config(self) -> None:
        name = self._get_selected_config()
        if name is None:
            self.notify(t("No config selected.", "선택된 config 가 없습니다."), severity="warning")
            return
        referencing = [
            p for p in list_profile_names()
            if load_profile(p).config_name == name
        ]
        self.app.push_screen(
            ConfirmDeleteConfigScreen(name, referencing),
            callback=self._on_delete_confirmed,
        )

    def _on_delete_confirmed(self, result: bool) -> None:
        if result:
            self._refresh_table()

    def action_refresh(self) -> None:
        self._refresh_table()

    def action_go_back(self) -> None:
        # pop, not switch_screen("dashboard") — this screen is pushed on top of
        # the dashboard, so switching would stack a *second* dashboard instance.
        self.app.pop_screen()

    def _on_form_closed(self, result: str | None = None) -> None:
        self._refresh_table()
