"""Quick Setup screen - create profile + config in one step."""

from __future__ import annotations

import re
from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static, Switch, Select

from tui.common import profile_store
from tui.common.i18n import t


class QuickSetupScreen(ModalScreen[str]):
    """Quick setup: create a profile + config from a model name."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("pageup", "scroll_form('up')", "Scroll up", show=False),
        Binding("pagedown", "scroll_form('down')", "Scroll down", show=False),
        Binding("home", "scroll_form('home')", "Scroll to top", show=False),
        Binding("end", "scroll_form('end')", "Scroll to bottom", show=False),
    ]

    DEFAULT_CSS = """
    QuickSetupScreen {
        align: center middle;
    }
    QuickSetupScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 70;
        min-width: 45;
        /* No max-height — see ProfileFormScreen for rationale. */
        height: 95%;
        min-height: 12;
    }
    QuickSetupScreen VerticalScroll {
        height: 1fr;
        min-height: 3;
    }
    QuickSetupScreen .title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        text-align: center;
        width: 100%;
    }
    QuickSetupScreen Label {
        margin-top: 1;
        color: $text-muted;
    }
    QuickSetupScreen Input {
        margin-bottom: 0;
    }
    QuickSetupScreen #mem-estimate {
        height: 1;
        color: $text-muted;
        margin-top: 0;
    }
    QuickSetupScreen .buttons {
        height: auto;
        min-height: 3;
        margin-top: 1;
        padding-top: 1;
        align: center middle;
        background: $surface;
        border-top: solid $primary 30%;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_estimated_model = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(t("Quick Setup", "빠른 설정"), classes="title")
            with VerticalScroll():
                yield Label(t("HuggingFace Model (e.g., meta-llama/Llama-3-8B)", "HuggingFace 모델 (예: meta-llama/Llama-3-8B)"))
                yield Input(placeholder="org/model-name", id="model-input")
                yield Static("", id="mem-estimate")
                yield Label(t("Profile Name", "프로필 이름"))
                yield Input(
                    placeholder=t("auto-derived from model", "모델에서 자동 생성"),
                    id="name-input",
                )
                yield Label(t("Recipe source model (optional)", "레시피 출처 모델 (선택)"))
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
                yield Label("GPU ID")
                yield Input(placeholder="0", value="0", id="gpu-input")
                yield Label("Port")
                # Prefill from the effective default (profiles.yaml `defaults:`
                # can override the built-in 8000) so the form agrees with what
                # `profile new --port 0` / quick-setup would pick.
                _port = str(profile_store.effective_defaults("vllm")["port"])
                yield Input(placeholder=_port, value=_port, id="port-input")
                yield Label(t("GPU Memory Utilization", "GPU 메모리 사용률"))
                yield Input(placeholder="0.9", value="0.9", id="gpu-mem-input")
                yield Label(t("Copy params from (optional)", "파라미터 복사 원본 (선택)"))
                yield Select(
                    self._build_config_options(),
                    prompt=t("None", "없음"),
                    allow_blank=True,
                    id="copy-config-select",
                )
                with Horizontal():
                    yield Label(t("Enable LoRA", "LoRA 사용"))
                    yield Switch(id="lora-switch")
            with Horizontal(classes="buttons"):
                yield Button(t("Create", "생성"), variant="primary", id="create-btn")
                yield Button(t("Cancel", "취소"), id="cancel-btn")

    def _build_config_options(self) -> list[tuple[str, str]]:
        from tui.backends.vllm.backend import list_config_names, load_config

        return [
            (t(f"{name} ({len(load_config(name).extra_params)} params)", f"{name} ({len(load_config(name).extra_params)} 파라미터)"), name)
            for name in list_config_names()
        ]

    @on(Input.Blurred, "#model-input")
    def _on_model_blur(self, event: Input.Blurred) -> None:
        model = event.input.value.strip()
        if model and model != self._last_estimated_model:
            self._last_estimated_model = model
            self._estimate_memory(model)

    @work(exclusive=True, group="mem-estimate")
    async def _estimate_memory(self, model_id: str) -> None:
        try:
            label = self.query_one("#mem-estimate", Static)
            label.update(t(f"[dim]Estimating memory for {model_id}...[/dim]", f"[dim]{model_id} 메모리 추정 중...[/dim]"))
        except Exception:
            return
        from tui.backends.vllm.backend import estimate_model_memory
        result = await estimate_model_memory(model_id)
        try:
            self.query_one("#mem-estimate", Static).update(
                t(f"[bold]Est. Memory:[/bold] {result}", f"[bold]예상 메모리:[/bold] {result}")
            )
        except Exception:
            pass


    @on(Button.Pressed, "#fetch-recipe-btn")
    def _on_fetch_recipe(self, event: Button.Pressed) -> None:
        event.stop()
        copy_from = self.query_one("#copy-config-select", Select).value
        if copy_from not in (None, Select.NULL, Select.BLANK):
            self.notify(
                t(
                    "A recipe cannot be combined with Copy config",
                    "레시피와 설정 복사를 함께 사용할 수 없습니다",
                ),
                severity="error",
            )
            return
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
    async def _fetch_recipe(self, model_id: str, target_model: str = "") -> None:
        from tui.common.docker import get_gpu_info
        from tui.common.recipes import RecipeUnavailable, fetch_recipe

        status = self.query_one("#recipe-status", Static)
        status.update(t(f"[dim]Fetching recipe for {model_id}…[/dim]",
                        f"[dim]{model_id} 레시피 받는 중…[/dim]"))
        try:
            recipe = await fetch_recipe(model_id)
        except RecipeUnavailable as exc:
            status.update(
                t(f"[red]Could not reach the recipe index: {exc}[/red]",
                  f"[red]레시피 조회 실패: {exc}[/red]"),
            )
            return
        if recipe is None:
            status.update(
                t(f"[yellow]No vLLM recipe found for {model_id}[/yellow]",
                  f"[yellow]{model_id} 의 vLLM 레시피 없음[/yellow]"),
            )
            return
        status.update("")

        try:
            gpus = await get_gpu_info()
        except Exception as exc:
            status.update(
                t(
                    f"[red]Could not query GPUs: {exc}[/red]",
                    f"[red]GPU 조회 실패: {exc}[/red]",
                )
            )
            return
        gpu_total_gb = None
        if gpus:
            try:
                gpu_total_gb = max(int(g.memory_total) / 1024 for g in gpus)
            except (AttributeError, TypeError, ValueError) as exc:
                status.update(
                    t(
                        f"[red]Invalid GPU query result: {exc}[/red]",
                        f"[red]잘못된 GPU 조회 결과: {exc}[/red]",
                    )
                )
                return

        from tui.backends.vllm.backend import get_local_latest_tag
        local_tag = await get_local_latest_tag()
        local_ver = "" if local_tag == "none" else local_tag

        from tui.backends.vllm.screens.recipe import RecipeReviewScreen

        def after(result: dict | None) -> None:
            if result:
                self._persist(result["model_id"], dict(result["params"]))

        self.app.push_screen(
            RecipeReviewScreen(
                recipe,
                gpu_total_gb=gpu_total_gb,
                local_vllm_version=local_ver,
                target_model=target_model if target_model != model_id else "",
            ),
            after,
        )

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")

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

    @on(Button.Pressed, "#create-btn")
    def on_create(self) -> None:
        model = self.query_one("#model-input", Input).value.strip()
        if not model:
            self.notify(t("Model name is required", "모델 이름은 필수입니다"), severity="error")
            return
        copy_select = self.query_one("#copy-config-select", Select)
        copy_from = (
            str(copy_select.value)
            if copy_select.value not in (Select.NULL, Select.BLANK, None)
            else ""
        )
        self._persist(model, {}, copy_from=copy_from)

    def _persist(
        self,
        model: str,
        extra_params: dict[str, Any],
        disabled_params: dict[str, Any] | None = None,
        copy_from: str = "",
    ) -> None:
        """Validate the form fields and create profile + config for `model`,
        seeding the config with `extra_params` (copy-from config, or a fetched
        recipe's flags) and `disabled_params` (copy-from only; recipes have
        none). Shared by manual create and the recipe importer."""
        gpu = self.query_one("#gpu-input", Input).value.strip()
        port = self.query_one("#port-input", Input).value.strip()
        gpu_mem = self.query_one("#gpu-mem-input", Input).value.strip()
        lora = self.query_one("#lora-switch", Switch).value

        requested_name = self.query_one("#name-input", Input).value.strip()
        name_part = model.rsplit("/", 1)[-1]
        safe_name = requested_name or re.sub(
            r"[^a-zA-Z0-9-]", "-", name_part
        ).lower().strip("-")

        if not safe_name:
            self.notify(t("Could not derive a valid name from model", "모델에서 유효한 이름을 만들 수 없습니다"),
                        severity="error")
            return

        from tui.backends.vllm.backend import validate_name

        if not validate_name(safe_name):
            self.notify(
                t(
                    "Profile name must use lowercase letters, digits, dashes, or underscores",
                    "프로필 이름은 소문자·숫자·대시·언더스코어만 사용할 수 있습니다",
                ),
                severity="error",
            )
            return

        try:
            port_num = int(port)
            if not 1024 <= port_num <= 65535:
                raise ValueError
        except ValueError:
            self.notify(t("Port must be between 1024 and 65535", "Port 는 1024 ~ 65535 사이여야 합니다"),
                        severity="error")
            return

        if gpu_mem:
            try:
                gpu_mem_val = float(gpu_mem)
                if not (0.0 < gpu_mem_val <= 1.0):
                    raise ValueError
            except ValueError:
                self.notify(t("GPU Memory Utilization must be between 0.0 and 1.0", "GPU 메모리 사용률은 0.0 ~ 1.0 사이여야 합니다"),
                            severity="error")
                return

        if not gpu or not re.match(r"^[0-9]+(,[0-9]+)*$", gpu):
            self.notify(t("GPU ID is required (e.g., 0 or 0,1)", "GPU ID 는 필수입니다 (예: 0 또는 0,1)"),
                        severity="error")
            return

        gpu_count = len(gpu.split(","))

        from tui.backends.vllm.backend import (
            CONFIG_DIR, Profile, Config, load_config, save_profile, save_config,
        )
        try:
            with profile_store.quick_setup_transaction(
                safe_name,
                "vllm",
                CONFIG_DIR,
            ) as final_name:
                if copy_from:
                    source = CONFIG_DIR / f"{copy_from}.yaml"
                    if not source.exists():
                        raise ValueError(f"config not found: {source}")
                    copied = load_config(copy_from)
                    extra_params = dict(copied.extra_params)
                    disabled_params = dict(copied.disabled_params)
                config = Config(
                    name=final_name,
                    model=model,
                    gpu_memory_utilization=gpu_mem or "0.9",
                    extra_params=extra_params,
                    disabled_params=dict(disabled_params or {}),
                )
                profile = Profile(
                    name=final_name,
                    container_name=final_name,
                    port=port,
                    gpu_id=gpu,
                    tensor_parallel=str(gpu_count),
                    config_name=final_name,
                    model_id=model,
                    enable_lora="true" if lora else "false",
                )
                save_config(config, template=source.read_text() if copy_from else None)
                save_profile(profile)
        except (OSError, RuntimeError, ValueError) as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return

        self.notify(
            t(
                f"✓ Created: {final_name}  (next: press 'u' to start)",
                f"✓ 생성: {final_name}  (다음: 'u' 로 시작)",
            ),
            severity="information",
        )
        self.dismiss(final_name)
