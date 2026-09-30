"""Config 편집 화면 — YAML config (llama-server 플래그) CRUD."""

from __future__ import annotations

from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.suggester import SuggestFromList
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Static, Switch

from tui.backends.llamacpp.backend import (
    CONFIG_DIR,
    Config,
    extract_llama_server_flags,
    format_config_param_value,
    list_config_names,
    list_profile_names,
    load_config,
    load_profile,
    parse_config_param_value,
    save_config,
    validate_name,
)
from tui.common import profile_store
from tui.common.flag_discovery import config_target
from tui.common.i18n import t
from tui.common.widgets import TextPromptModal


# llama-server 주요 플래그 참조표 (자동추출 실패/보완 시 fallback)
# key → (한줄설명, 예시값)

LLAMA_SERVER_FLAGS: dict[str, tuple[str, str]] = {
    # 필수/핵심
    "model-file":       ("GGUF 파일명 (hf_file 미설정 시 -hff 로 사용)", "my-model-Q4_K_M.gguf"),
    "alias":            ("/v1/models 에 노출될 모델 이름",       "my-model"),
    # 컨텍스트 / 레이어
    "ctx-size":         ("컨텍스트 길이 (tokens)",                "32768"),
    "n-gpu-layers":     ("GPU 에 올릴 레이어 수 (99=전체)",       "99"),
    "n-predict":        ("기본 최대 생성 토큰",                   "-1"),
    "parallel":         ("동시 디코딩 슬롯 수",                   "1"),
    "batch-size":       ("physical batch (prompt eval)",         "2048"),
    "ubatch-size":      ("micro-batch size",                     "512"),
    "threads":          ("CPU 스레드 수",                        "8"),
    "threads-batch":    ("배치용 CPU 스레드 수",                  "8"),
    # KV 캐시
    "cache-type-k":     ("KV 캐시 K 정밀도 (f16/bf16/q8_0/q4_0)", "bf16"),
    "cache-type-v":     ("KV 캐시 V 정밀도",                     "bf16"),
    "no-kv-offload":    ("KV 캐시 GPU offload 끄기 (boolean)",    ""),
    # MoE / tensor 관리
    "override-tensors": ("tensor→device 라우팅 (-ot 리스트)",     ".ffn_.*_exps.=CPU"),
    "split-mode":       ("multi-GPU split 전략 (none/layer/row)", "layer"),
    "tensor-split":     ("GPU 간 tensor 비율 (예: 0.5,0.5)",      "0.5,0.5"),
    "main-gpu":         ("primary GPU index",                    "0"),
    # 최적화
    "flash-attn":       ("Flash Attention 켜기 (boolean)",         ""),
    "no-mmap":          ("mmap 비활성 (boolean)",                  ""),
    "mlock":            ("메모리 lock (boolean)",                  ""),
    "cont-batching":    ("continuous batching (boolean, 기본 on)", ""),
    "rope-scaling":     ("RoPE scaling type (none/linear/yarn)",  "yarn"),
    "rope-scale":       ("RoPE 스케일 factor",                    "2.0"),
    "rope-freq-base":   ("RoPE base freq override",               "10000"),
    # 샘플링 기본값
    "temp":             ("샘플링 temperature",                    "0.7"),
    "top-p":            ("top-p nucleus",                        "0.95"),
    "top-k":            ("top-k",                                "40"),
    "min-p":            ("min-p cutoff",                         "0.05"),
    "repeat-penalty":   ("repetition penalty",                   "1.1"),
    "seed":             ("RNG seed (-1=random)",                 "-1"),
    # Chat template / reasoning
    "jinja":            ("jinja chat-template 사용 (boolean)",     ""),
    "chat-template":    ("chat template 이름 오버라이드",          "chatml"),
    "chat-template-file": ("chat template 파일 경로",              "/models/tpl.jinja"),
    "reasoning-format": ("reasoning 출력 포맷 (none/deepseek)",    "deepseek"),
    "reasoning-budget": ("reasoning token budget",                "-1"),
    # 네트워크 / 보안
    "api-key":          ("API key (요청 헤더 인증)",               "sk-secret"),
    "api-key-file":     ("API key 파일 경로",                      "/run/api-keys"),
    "timeout":          ("요청 timeout 초",                        "600"),
    "metrics":          ("/metrics Prometheus 노출 (boolean)",      ""),
    "slots":            ("/slots endpoint (boolean)",              ""),
    "props":            ("/props endpoint (boolean)",              ""),
    "verbose":          ("verbose 로그 (boolean)",                 ""),
    "log-disable":      ("로그 파일 출력 끄기 (boolean)",           ""),
    # 멀티모달 / 프로젝터
    "mmproj":           ("multimodal projector GGUF 경로",         "/models/mmproj.gguf"),
    # 임베딩
    "embedding":        ("embedding 모드 (boolean)",               ""),
    "pooling":          ("embedding pooling (none/mean/cls/last)", "mean"),
    # 기타 고급
    "host":             ("bind host (docker override 에 의해 강제됨)", "0.0.0.0"),
    "port":             ("bind port (docker override 에 의해 강제됨)", "8080"),
}


class ConfigFormScreen(ModalScreen[str | None]):
    """config 생성/편집 modal."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("pageup", "scroll_form('up')", "Scroll up", show=False),
        Binding("pagedown", "scroll_form('down')", "Scroll down", show=False),
        Binding("home", "scroll_form('home')", "Scroll to top", show=False),
        Binding("end", "scroll_form('end')", "Scroll to bottom", show=False),
    ]

    DEFAULT_CSS = """
    ConfigFormScreen { align: center middle; }
    ConfigFormScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 90;
        min-width: 60;
        /* No max-height — see vLLM ConfigFormScreen for rationale. */
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
        width: 20;
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
    ConfigFormScreen #flag-help {
        color: $accent;
        margin-bottom: 1;
        padding: 0 1;
        background: $boost;
        min-height: 1;
    }
    ConfigFormScreen #add-param-row {
        height: 3;
        margin-bottom: 1;
        align-horizontal: left;
    }
    ConfigFormScreen #add-param-row Button {
        width: auto;
        min-width: 20;
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
        min-height: 5;
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
        self._known_flags: set[str] = set()
        self._flag_suggester = SuggestFromList(
            sorted(self._known_flags), case_sensitive=False
        )

    def compose(self) -> ComposeResult:
        if self._edit_mode:
            path = CONFIG_DIR / f"{self._config_name}.yaml"
            with profile_store.storage_transaction():
                if not path.exists():
                    raise ValueError(f"config not found: {path}")
                self._initial_text = path.read_text()
                self._initial_config = load_config(self._config_name)
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
                        value=self._config_name,
                        placeholder="my-model",
                        id="name-input",
                    )
                yield Static(t("llama-server Parameters", "llama-server 파라미터"), id="params-title")
                yield Static(
                    t(
                        "[dim]Tab/→: autocomplete  ·  boolean flags default to true when left blank[/dim]",
                        "[dim]Tab/→: 자동완성  ·  boolean 플래그는 값 비우면 true[/dim]",
                    ),
                    id="params-hint",
                )
                yield Static("", id="flag-help")
                with Horizontal(id="add-param-row"):
                    yield Button(t("+ Add Parameter", "+ 파라미터 추가"), id="add-param-btn")
                yield Vertical(id="params-container")
            with Horizontal(classes="form-buttons"):
                yield Button(t("Save", "저장"), id="save-btn", variant="primary")
                yield Button(t("Close", "닫기"), id="cancel-btn", variant="default")

    def on_mount(self) -> None:
        if self._edit_mode and self._initial_config:
            for key, value in self._initial_config.params.items():
                self._add_param_row(key, format_config_param_value(value))
            for key, value in self._initial_config.disabled_params.items():
                self._add_param_row(key, format_config_param_value(value), enabled=False)
        else:
            for key in ("model-file", "ctx-size", "n-gpu-layers"):
                ex = LLAMA_SERVER_FLAGS.get(key, ("", ""))[1]
                self._add_param_row(key, ex)
        self._load_server_flags()

    @work(exclusive=False)
    async def _load_server_flags(self) -> None:
        try:
            image, container = await config_target("llamacpp", self._config_name, self._profile_name)
            extracted = await extract_llama_server_flags(image, **({"container_name": container} if container else {}))
            if not extracted:
                raise RuntimeError("image returned no supported parameters")
        except (OSError, RuntimeError, ValueError) as exc:
            self.query_one("#params-hint", Static).update(t("Flag lookup failed; autocomplete unavailable.", "인자 조회 실패: 자동완성을 사용할 수 없습니다."))
            self.notify(
                t(
                    f"Could not inspect llama.cpp flags: {exc}",
                    f"llama.cpp 플래그를 확인할 수 없습니다: {exc}",
                ),
                severity="error",
            )
            return
        if extracted:
            self.query_one("#params-hint", Static).update(t(f"Parameters: {container or image or 'next startup image'}", f"인자 기준: {container or image or '다음 기동 이미지'}"))
            self._known_flags = set(extracted)
            self._flag_suggester = SuggestFromList(
                sorted(self._known_flags), case_sensitive=False
            )
            for inp in self.query(".param-key"):
                inp.suggester = self._flag_suggester

    def _add_param_row(
        self, key: str = "", value: str = "", *, focus: bool = False, enabled: bool = True
    ) -> None:
        container = self.query_one("#params-container", Vertical)
        row_id = f"param-row-{self._param_counter}"
        self._param_counter += 1
        key_input = Input(
            value=key,
            placeholder=t("flag-name (Tab: autocomplete)", "flag-name (Tab: 자동완성)"),
            suggester=self._flag_suggester,
            classes="param-key",
        )
        # Switch off → saved as a disabled comment marker (kept, not served).
        row = Horizontal(
            Switch(value=enabled, classes="param-switch"),
            key_input,
            Input(value=value, placeholder=t("value (blank = true)", "value (비우면 true)"), classes="param-value"),
            Button("x", classes="param-remove"),
            id=row_id,
            classes="param-row" if enabled else "param-row -disabled",
        )
        container.mount(row)

        def _after() -> None:
            try:
                row.scroll_visible(animate=False)
                if focus:
                    key_input.focus()
            except Exception:
                pass

        self.call_after_refresh(_after)

    @on(Input.Changed, ".param-key")
    def _on_key_changed(self, event: Input.Changed) -> None:
        key = event.value.strip()
        help_widget = self.query_one("#flag-help", Static)
        info = LLAMA_SERVER_FLAGS.get(key)
        if info:
            desc, example = info
            ex_suffix = f"  [dim](예: {example})[/dim]" if example else ""
            help_widget.update(f"[b]{key}[/b]: {desc}{ex_suffix}")
        elif key in self._known_flags:
            help_widget.update(f"[b]{key}[/b]: [dim](llama-server --help 에 존재)[/dim]")
        else:
            help_widget.update("")

    @on(Switch.Changed, ".param-switch")
    def _on_switch(self, event: Switch.Changed) -> None:
        row = event.switch.parent
        if row is not None:
            row.set_class(not event.value, "-disabled")

    @on(Button.Pressed, "#add-param-btn")
    def _on_add_param(self, event: Button.Pressed) -> None:
        event.stop()
        self._add_param_row(focus=True)
        self.notify(t("Parameter added — enter a flag name (Tab to autocomplete)", "파라미터 추가됨 — flag 이름 입력 (Tab 자동완성)"), timeout=3)

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

        if not name:
            self.notify(t("Config name is required", "Config 이름 필수"), severity="error")
            return
        if not validate_name(name):
            self.notify(
                t(
                    "Name may contain only letters, digits, dashes, and underscores (must not start with '-')",
                    "이름은 영숫자/대시/언더스코어만 가능 ('-' 시작 금지)",
                ),
                severity="error",
            )
            return
        params: dict[str, Any] = {}
        disabled_params: dict[str, Any] = {}
        seen: set[str] = set()
        for row in self.query(".param-row"):
            key = row.query_one(".param-key", Input).value.strip()
            val = row.query_one(".param-value", Input).value.strip()
            switch = row.query_one(".param-switch", Switch)
            if not key:
                continue
            # Duplicate check spans active + disabled.
            if key in seen:
                self.notify(t(f"Duplicate flag: {key}", f"중복 플래그: {key}"), severity="error")
                return
            seen.add(key)
            try:
                parsed = parse_config_param_value(val)
            except Exception as exc:
                self.notify(t(f"Failed to parse value for '{key}': {exc}", f"'{key}' 값 파싱 실패: {exc}"), severity="error")
                return
            (params if switch.value else disabled_params)[key] = parsed

        unknown = [k for k in params if k not in self._known_flags and k not in {"model-file", "extra-args"}] if self._known_flags else []
        if unknown:
            self.notify(
                t(
                    f"Unknown flags (may be valid for your llama-server version): {', '.join(unknown)}",
                    f"알 수 없는 플래그 (llama-server 버전에 따라 유효할 수 있음): {', '.join(unknown)}",
                ),
                severity="warning",
                timeout=6,
            )

        cfg = Config(name=name, params=params, disabled_params=disabled_params)
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
        self.notify(t(f"Saved: {name}", f"저장: {name}"), severity="information")
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
        from tui.common.config_markers import dump_active_config, render_disabled_markers

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
        for prof in config_store.referencing_profiles("llamacpp", old_name):
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
            def serialize(source: str) -> str:
                replacement = dump_active_config(source, config.params)
                return replacement + render_disabled_markers(config.disabled_params)

            with profile_store.storage_transaction():
                config_store.rename_config(
                    "llamacpp",
                    old_name,
                    new_name,
                    replacement=serialize,
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
    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ConfirmDeleteConfigScreen { align: center middle; }
    ConfirmDeleteConfigScreen > Vertical {
        background: $surface;
        border: round $error;
        padding: 1 2;
        width: 60;
        height: auto;
    }
    ConfirmDeleteConfigScreen #confirm-title {
        text-style: bold;
        color: $error;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ConfirmDeleteConfigScreen #confirm-msg,
    ConfirmDeleteConfigScreen #confirm-warn {
        text-align: center;
        margin-bottom: 1;
    }
    ConfirmDeleteConfigScreen #confirm-warn { color: $warning; }
    ConfirmDeleteConfigScreen .confirm-buttons {
        height: auto;
        align: center middle;
    }
    ConfirmDeleteConfigScreen .confirm-buttons Button { margin: 0 1; }
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
                    f"Delete [b]{self._config_name}[/b]?",
                    f"[b]{self._config_name}[/b] 를 삭제할까요?",
                ),
                id="confirm-msg",
            )
            if self._referencing:
                yield Static(
                    t(
                        f"[b]Warning:[/b] Used by profiles: {', '.join(self._referencing)}",
                        f"[b]경고:[/b] 사용 중인 프로필: {', '.join(self._referencing)}",
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
            config_store.delete_config("llamacpp", self._config_name)
        except (OSError, RuntimeError, ValueError) as exc:
            self.app.notify(str(exc), severity="error", timeout=8)
            return
        self.app.notify(t(f"Deleted: {self._config_name}", f"삭제됨: {self._config_name}"))
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def _on_no(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class ConfigListScreen(Screen):
    """config/*.yaml 전체 목록."""

    BINDINGS = [
        Binding("n", "new_config", t("New", "새로")),
        Binding("e,enter", "edit_config", t("Edit", "편집")),
        Binding("c", "clone_config", t("Clone", "복제")),
        Binding("R", "rename_config", t("Rename", "이름변경")),
        Binding("delete,x", "delete_config", t("Delete", "삭제")),
        Binding("escape,backspace", "go_back", t("Back", "뒤로")),
        Binding("r", "refresh", t("Refresh", "새로고침")),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(t("[b]Configs (llama-server YAML)[/b]", "[b]Config 목록 (llama-server YAML)[/b]"), id="config-title")
        yield DataTable(id="config-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#config-table", DataTable)
        table.add_columns("Name", "Model File", "Ctx", "N-GPU-Layers", "Params")
        self._refresh_table()

    def on_screen_resume(self) -> None:
        # Named SCREENS entries are cached instances — on_mount fires once, so
        # without this the list goes stale after configs change elsewhere.
        self._refresh_table()

    def _refresh_table(self) -> None:
        table = self.query_one("#config-table", DataTable)
        table.clear()
        for name in list_config_names():
            cfg = load_config(name)
            model_file = str(cfg.params.get("model-file", "—"))
            ctx = str(cfg.params.get("ctx-size", "—"))
            ngl = str(cfg.params.get("n-gpu-layers", "—"))
            table.add_row(
                cfg.name, model_file, ctx, ngl, str(len(cfg.params)), key=cfg.name
            )

    def _get_selected(self) -> str | None:
        table = self.query_one("#config-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            return row_key.value if row_key else None
        except Exception:
            return None

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.action_edit_config()

    def action_new_config(self) -> None:
        self.app.push_screen(ConfigFormScreen(), self._on_form_closed)

    def action_edit_config(self) -> None:
        name = self._get_selected()
        if not name:
            self.notify(t("No config selected", "선택된 config 없음"), severity="warning")
            return
        self.app.push_screen(ConfigFormScreen(name), self._on_form_closed)

    def action_clone_config(self) -> None:
        name = self._get_selected()
        if not name:
            self.notify(t("No config selected", "선택된 config 없음"), severity="warning")
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
            if not validate_name(new_name):
                self.notify(t("Name must be lowercase letters/digits/dashes/underscores", "이름은 소문자/숫자/대시/언더스코어"), severity="error")
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
            self.notify(t(f"Cloned: '{name}' → '{new_name}'", f"복제: '{name}' → '{new_name}'"))

        self.app.push_screen(
            TextPromptModal(
                t(f"Clone config '{name}' as:", f"Config '{name}' 을(를) 다음 이름으로 복제:"),
                default=default,
            ),
            after,
        )

    def action_rename_config(self) -> None:
        name = self._get_selected()
        if not name:
            self.notify(t("No config selected", "선택된 config 없음"), severity="warning")
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
            after,
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
        for p in config_store.referencing_profiles("llamacpp", old):
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
            updated = config_store.rename_config("llamacpp", old, new)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self._refresh_table()
        suffix = f" ({len(updated)} profile(s) repointed)" if updated else ""
        self.notify(
            t(f"Renamed '{old}' → '{new}'{suffix}", f"이름 변경: '{old}' → '{new}'{suffix}")
        )

    def action_delete_config(self) -> None:
        name = self._get_selected()
        if not name:
            self.notify(t("No config selected", "선택된 config 없음"), severity="warning")
            return
        referencing = [
            p for p in list_profile_names() if load_profile(p).config_name == name
        ]
        self.app.push_screen(
            ConfirmDeleteConfigScreen(name, referencing), self._on_delete_confirmed
        )

    def action_refresh(self) -> None:
        self._refresh_table()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def _on_form_closed(self, result: str | None = None) -> None:
        self._refresh_table()

    def _on_delete_confirmed(self, result: bool) -> None:
        if result:
            self._refresh_table()
