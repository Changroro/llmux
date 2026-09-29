"""Quick Setup — HF repo URL → GGUF 선택 → profile + config 자동 생성."""

from __future__ import annotations

import re
from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    Label,
    Select,
    Static,
    Switch,
)

from tui.backends.llamacpp.backend import (
    CONFIG_DIR,
    Config,
    Profile,
    list_config_names,
    HfListingUnavailable,
    list_hf_repo_files,
    list_profile_names as list_profile_names,
    load_config,
    save_config,
    save_profile,
    validate_name,
)
from tui.backends.llamacpp.defaults import (
    QUICK_SETUP_CACHE_TYPE_K,
    QUICK_SETUP_CACHE_TYPE_V,
    QUICK_SETUP_CTX_SIZE,
    QUICK_SETUP_FLASH_ATTN,
    QUICK_SETUP_JINJA,
    QUICK_SETUP_N_GPU_LAYERS,
)
from tui.common import profile_store
from tui.common.i18n import t
from tui.common.prepare import hf_file_error


_REPO_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
_MOE_PATTERN = re.compile(r"[Aa][0-9]+B")
_DEFAULT_OT = ".ffn_.*_exps.=CPU"


def _normalize_repo(raw: str) -> str:
    """huggingface.co URL 이면 repo 경로만 추출."""
    s = raw.strip()
    if not s:
        return ""
    s = s.rstrip("/")
    if "huggingface.co/" in s:
        s = s.split("huggingface.co/", 1)[1]
        if s.startswith("api/models/"):
            s = s[len("api/models/"):]
    parts = s.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return s


class QuickSetupScreen(ModalScreen[str]):
    """HF repo + GGUF 파일로 profile/config 자동 생성 modal."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("pageup", "scroll_form('up')", "Scroll up", show=False),
        Binding("pagedown", "scroll_form('down')", "Scroll down", show=False),
        Binding("home", "scroll_form('home')", "Scroll to top", show=False),
        Binding("end", "scroll_form('end')", "Scroll to bottom", show=False),
    ]

    DEFAULT_CSS = """
    QuickSetupScreen { align: center middle; }
    QuickSetupScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 82;
        min-width: 60;
        /* No max-height — see vLLM QuickSetupScreen for rationale. */
        height: 95%;
        min-height: 12;
    }
    QuickSetupScreen .title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        text-align: center;
        width: 100%;
    }
    QuickSetupScreen VerticalScroll { height: 1fr; min-height: 5; }
    QuickSetupScreen Label { margin-top: 1; color: $text-muted; }
    QuickSetupScreen #gguf-info {
        height: auto;
        min-height: 1;
        color: $text-muted;
        margin-top: 0;
    }
    QuickSetupScreen #moe-hint {
        height: auto;
        min-height: 1;
        color: $accent;
        margin-top: 1;
    }
    QuickSetupScreen #fetch-btn {
        width: 100%;
        margin-top: 1;
    }
    QuickSetupScreen .switch-row {
        height: 3;
        margin-top: 1;
        padding: 0;
    }
    QuickSetupScreen .switch-row Label {
        width: 1fr;
        margin-top: 1;
    }
    QuickSetupScreen .switch-row Switch {
        width: auto;
    }
    QuickSetupScreen Collapsible {
        margin-top: 1;
        border-top: solid $primary 30%;
    }
    QuickSetupScreen Collapsible CollapsibleTitle {
        color: $text;
    }
    QuickSetupScreen .section-help {
        color: $text-muted;
        margin: 0 0 1 0;
        height: auto;
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
        self._last_repo = ""
        self._listing_failed = False

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(t("Quick Setup — HF repo → profile + config", "빠른 설정 — HF repo → profile + config"), classes="title")
            with VerticalScroll():
                yield Label(t("HuggingFace repo (e.g. unsloth/Qwen3-30B-A3B-GGUF)", "HuggingFace repo (예: unsloth/Qwen3-30B-A3B-GGUF)"))
                yield Input(placeholder=t("org/repo-GGUF or URL", "org/repo-GGUF 또는 URL"), id="repo-input")
                yield Button(t("Fetch files", "파일 가져오기"), id="fetch-btn")
                yield Static("", id="gguf-info")

                yield Label(t("GGUF file", "GGUF 파일"))
                yield Select(
                    [(t("(enter repo then Fetch)", "(repo 입력 후 Fetch)"), "__none__")],
                    allow_blank=False,
                    id="gguf-select",
                )
                yield Input(
                    placeholder=t(
                        "manual filename (enabled if lookup fails)",
                        "수동 파일명 (조회 실패 시 활성화)",
                    ),
                    id="manual-file-input",
                    disabled=True,
                )
                yield Static("", id="moe-hint")

                yield Label(t("Profile name (blank = auto from filename)", "Profile 이름 (비우면 파일명에서 자동 생성)"))
                yield Input(placeholder="auto", id="name-input")

                yield Label("Port")
                # Prefill from the effective default (profiles.yaml `defaults:`
                # can override the built-in 8080).
                _port = str(profile_store.effective_defaults("llamacpp")["port"])
                yield Input(placeholder=_port, value=_port, id="port-input")

                yield Label(t("GPU ID (e.g. 0 or 0,1)", "GPU ID (예: 0 또는 0,1)"))
                yield Input(placeholder="0", value="0", id="gpu-input")

                with Collapsible(
                    title=t("llama.cpp common parameters", "llama.cpp 공통 파라미터"),
                    collapsed=False,
                    id="common-params",
                ):
                    yield Static(
                        t(
                            "[dim]Blank fields are not written to the config and use llama.cpp defaults. "
                            "Fields with a value are saved to the config YAML.[/dim]",
                            "[dim]빈 칸은 config 에 기록되지 않고 llama.cpp 기본값을 사용합니다. "
                            "값이 있으면 config YAML 에 저장됩니다.[/dim]",
                        ),
                        classes="section-help",
                    )

                    yield Label(t("Ctx size (context length, tokens)", "Ctx size (컨텍스트 길이, tokens)"))
                    yield Input(
                        placeholder=QUICK_SETUP_CTX_SIZE,
                        value=QUICK_SETUP_CTX_SIZE,
                        id="ctx-input",
                    )

                    yield Label(
                        t(
                            f"N-GPU-Layers (layers on GPU, {QUICK_SETUP_N_GPU_LAYERS}=all)",
                            f"N-GPU-Layers (GPU 에 올릴 레이어, {QUICK_SETUP_N_GPU_LAYERS}=전체)",
                        )
                    )
                    yield Input(
                        placeholder=QUICK_SETUP_N_GPU_LAYERS,
                        value=QUICK_SETUP_N_GPU_LAYERS,
                        id="ngl-input",
                    )

                    yield Label(
                        t(
                            "KV cache K precision (f16 / bf16 / q8_0 / q4_0 — lower = less VRAM)",
                            "KV cache K 정밀도 (f16 / bf16 / q8_0 / q4_0 — 낮출수록 VRAM ↓)",
                        )
                    )
                    yield Input(
                        placeholder=QUICK_SETUP_CACHE_TYPE_K,
                        value=QUICK_SETUP_CACHE_TYPE_K,
                        id="ctk-input",
                    )

                    yield Label(t("KV cache V precision", "KV cache V 정밀도"))
                    yield Input(
                        placeholder=QUICK_SETUP_CACHE_TYPE_V,
                        value=QUICK_SETUP_CACHE_TYPE_V,
                        id="ctv-input",
                    )

                    yield Label(t("Batch size (prompt eval unit, blank = default)", "Batch size (prompt eval 단위, 비우면 기본)"))
                    yield Input(placeholder=t("use default", "기본값 사용"), id="batch-input")

                    with Horizontal(classes="switch-row"):
                        yield Label(t("Flash Attention (faster, same accuracy)", "Flash Attention (속도↑, 정확도는 동일)"))
                        yield Switch(
                            value=QUICK_SETUP_FLASH_ATTN,
                            id="flash-attn-switch",
                        )

                    with Horizontal(classes="switch-row"):
                        yield Label(t("Jinja chat template (needed for /v1/chat/completions)", "Jinja chat template (/v1/chat/completions 에 필요)"))
                        yield Switch(value=QUICK_SETUP_JINJA, id="jinja-switch")

                with Collapsible(
                    title=t("MoE Expert Offload (optional)", "MoE Expert Offload (선택)"),
                    collapsed=True,
                    id="moe-collapsible",
                ):
                    yield Static(
                        t(
                            "[dim]MoE models (Qwen3-A3B, Mixtral, etc.) activate only a subset of "
                            "the total parameters per token. [b]Keeping expert weights in CPU RAM and "
                            "streaming only the active experts to the GPU[/b] can fit a 35B MoE on a 16GB GPU.\n"
                            "e.g. [b]qwen3.6-35b-a3b UD-Q4_K_XL[/b] → VRAM 7.5GB + RAM 15GB; throughput "
                            "depends heavily on RAM bandwidth (~36 tok/s measured on an RTX 4080 SUPER).\n\n"
                            "If the filename has an 'A3B' / 'A7B' pattern it is auto-detected and the default regex is filled in. "
                            "Leave blank for no expert offload (dense models usually don't need it).[/dim]",
                            "[dim]MoE 모델 (Qwen3-A3B, Mixtral 등) 은 전체 파라미터 중 일부만 "
                            "매 토큰 활성화됩니다. [b]Expert 가중치를 CPU RAM 에 두고 "
                            "활성 expert 만 GPU 로 스트리밍[/b] 하면 16GB GPU 에 35B MoE 도 올라갑니다.\n"
                            "예: [b]qwen3.6-35b-a3b UD-Q4_K_XL[/b] → VRAM 7.5GB + RAM 15GB. 속도는 RAM 대역폭에 "
                            "크게 좌우됩니다 (RTX 4080 SUPER 기준 약 36 tok/s).\n\n"
                            "파일명에 'A3B', 'A7B' 패턴이 있으면 자동 감지 후 기본 정규식을 채워드립니다. "
                            "비우면 expert offload 안 함 (Dense 모델은 보통 필요 없음).[/dim]",
                        ),
                        classes="section-help",
                    )
                    yield Label(t("override-tensors regex (blank = not applied)", "override-tensors 정규식 (비우면 미적용)"))
                    yield Input(
                        placeholder=_DEFAULT_OT,
                        id="ot-input",
                    )

                yield Label(t("Copy extra params from an existing config (optional)", "기존 config 에서 추가 파라미터 복사 (선택)"))
                yield Select(
                    self._build_config_options(),
                    prompt=t("None", "없음"),
                    allow_blank=True,
                    id="copy-config-select",
                )

            with Horizontal(classes="buttons"):
                yield Button(t("Create", "생성"), variant="primary", id="create-btn")
                yield Button(t("Cancel", "취소"), id="cancel-btn")

    def _build_config_options(self) -> list[tuple[str, str]]:
        return [
            (t(f"{name} ({len(load_config(name).params)} params)", f"{name} ({len(load_config(name).params)} 파라미터)"), name)
            for name in list_config_names()
        ]


    @on(Button.Pressed, "#fetch-btn")
    def _on_fetch(self) -> None:
        repo_raw = self.query_one("#repo-input", Input).value
        repo = _normalize_repo(repo_raw)
        if not repo or not _REPO_RE.match(repo):
            self.notify(t("Not a valid HF repo path (org/name)", "유효한 HF repo 경로가 아님 (org/name)"), severity="error")
            return
        self._last_repo = repo
        info = self.query_one("#gguf-info", Static)
        info.update(t(f"[dim]Fetching file list for {repo}...[/dim]", f"[dim]{repo} 파일 목록 가져오는 중...[/dim]"))
        self._fetch_files(repo)

    @work(exclusive=True, group="hf-fetch")
    async def _fetch_files(self, repo: str) -> None:
        info = self.query_one("#gguf-info", Static)
        try:
            files = await list_hf_repo_files(repo)
        except HfListingUnavailable as exc:
            self._listing_failed = True
            info.update(t(
                f"[red]Could not reach huggingface.co: {exc}. Enter the GGUF filename manually.[/red]",
                f"[red]huggingface.co 조회 실패: {exc}. GGUF 파일명을 직접 입력하세요.[/red]",
            ))
            self.query_one("#gguf-select", Select).set_options(
                [(t("(lookup failed)", "(조회 실패)"), "__none__")]
            )
            manual = self.query_one("#manual-file-input", Input)
            manual.disabled = False
            manual.focus()
            return
        self._listing_failed = False
        manual = self.query_one("#manual-file-input", Input)
        manual.value = ""
        manual.disabled = True
        gguf_items = [
            f for f in files
            if isinstance(f, dict)
            and f.get("type") == "file"
            and str(f.get("path", "")).lower().endswith(".gguf")
        ]
        select = self.query_one("#gguf-select", Select)
        if not gguf_items:
            info.update(t("[red]No GGUF files (or private repo — check HF_TOKEN)[/red]", "[red]GGUF 파일 없음 (또는 private repo — HF_TOKEN 확인)[/red]"))
            select.set_options([(t("(none)", "(없음)"), "__none__")])
            return

        opts: list[tuple[str, str]] = []
        for f in sorted(gguf_items, key=lambda x: str(x.get("path", ""))):
            path = str(f.get("path", ""))
            size = f.get("size") or 0
            size_gb = size / 1024**3 if isinstance(size, (int, float)) else 0
            label = f"{path}  ({size_gb:.1f} GB)" if size_gb else path
            opts.append((label, path))
        select.set_options(opts)
        select.value = opts[0][1]
        info.update(t(f"[green]{len(opts)} GGUF files[/green]", f"[green]{len(opts)} 개 GGUF 파일[/green]"))
        self._update_moe_hint(opts[0][1])


    @on(Select.Changed, "#gguf-select")
    def _on_gguf_changed(self, event: Select.Changed) -> None:
        if event.value in (Select.NULL, Select.BLANK, "__none__", None):
            return
        self.query_one("#manual-file-input", Input).value = str(event.value)
        self._update_moe_hint(str(event.value))

    def _update_moe_hint(self, gguf_file: str) -> None:
        hint = self.query_one("#moe-hint", Static)
        ot_input = self.query_one("#ot-input", Input)
        moe_collapsible = self.query_one("#moe-collapsible", Collapsible)

        if _MOE_PATTERN.search(gguf_file):
            hint.update(
                t(
                    f"[accent]⚠ MoE detected[/accent]: [dim]'{gguf_file}' → "
                    f"expert offload recommended (see the 'MoE Expert Offload' section below)[/dim]",
                    f"[accent]⚠ MoE 감지[/accent]: [dim]'{gguf_file}' → "
                    f"expert offload 권장 (아래 'MoE Expert Offload' 섹션 참고)[/dim]",
                )
            )
            if not ot_input.value.strip():
                ot_input.value = _DEFAULT_OT
            moe_collapsible.collapsed = False
        else:
            hint.update(
                t(
                    "[dim]Dense model — no expert offload needed (loads fully into VRAM)[/dim]",
                    "[dim]Dense 모델 — expert offload 불필요 (전체 VRAM 적재)[/dim]",
                )
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

    def _get(self, wid: str) -> str:
        return self.query_one(f"#{wid}", Input).value.strip()

    @on(Button.Pressed, "#create-btn")
    def on_create(self) -> None:
        repo = _normalize_repo(self.query_one("#repo-input", Input).value)
        gguf_select = self.query_one("#gguf-select", Select)
        gguf_file = (
            str(gguf_select.value)
            if gguf_select.value
            not in (Select.NULL, Select.BLANK, "__none__", None)
            else ""
        )
        if self._listing_failed:
            gguf_file = self._get("manual-file-input")
        name_raw = self._get("name-input")
        port_raw = self._get("port-input")
        gpu = self._get("gpu-input") or "0"
        ctx = self._get("ctx-input")
        ngl = self._get("ngl-input")
        ctk = self._get("ctk-input")
        ctv = self._get("ctv-input")
        batch = self._get("batch-input")
        ot = self._get("ot-input")
        flash_attn = self.query_one("#flash-attn-switch", Switch).value
        jinja = self.query_one("#jinja-switch", Switch).value

        if not repo or not _REPO_RE.match(repo):
            self.notify(t("A valid HF repo is required", "유효한 HF repo 필요"), severity="error")
            return
        if self._last_repo and repo != self._last_repo:
            self.notify(
                t(
                    "The repo changed after Fetch; fetch its GGUF files again",
                    "Fetch 후 repo 가 변경되었습니다. GGUF 파일을 다시 가져오세요",
                ),
                severity="error",
            )
            return
        if not gguf_file:
            self.notify(t("Select a GGUF file (after Fetch)", "GGUF 파일 선택 필요 (Fetch 후)"), severity="error")
            return
        file_error = hf_file_error(gguf_file)
        if file_error:
            self.notify(file_error, severity="error")
            return
        try:
            port_num = int(
                port_raw or profile_store.effective_defaults("llamacpp")["port"]
            )
            if not 1024 <= port_num <= 65535:
                raise ValueError
        except ValueError:
            self.notify(t("Port must be 1024–65535", "Port 는 1024–65535"), severity="error")
            return

        if not name_raw:
            base = repo.rsplit("/", 1)[-1]
            base = re.sub(r"[-_]?GGUF$", "", base, flags=re.I)
            name_raw = re.sub(r"[^a-z0-9_-]+", "-", base.lower()).strip("-")
        if not name_raw:
            self.notify(t("Failed to generate a name", "이름 생성 실패"), severity="error")
            return
        # docker compose project name 으로도 쓰이므로 편집 폼과 동일한 규칙 적용.
        if not validate_name(name_raw):
            self.notify(
                t("Name must be lowercase letters/digits/dashes/underscores", "이름은 소문자/숫자/대시/언더스코어만 가능"),
                severity="error",
            )
            return
        if not re.fullmatch(r"[0-9]+(,[0-9]+)*", gpu):
            self.notify(t("GPU ID must be digits/commas (e.g. 0 or 0,1)", "GPU ID 는 숫자/콤마 (예: 0 또는 0,1)"), severity="error")
            return

        copy_sel = self.query_one("#copy-config-select", Select)
        copy_from = (
            str(copy_sel.value)
            if copy_sel.value not in (Select.NULL, Select.BLANK, None)
            else ""
        )
        updates: dict[str, Any] = {"model-file": gguf_file}
        removals: set[str] = set()
        for key, raw, label in (
            ("ctx-size", ctx, "Ctx size"),
            ("n-gpu-layers", ngl, "N-GPU-Layers"),
            ("batch-size", batch, "Batch size"),
        ):
            if not raw:
                removals.add(key)
                continue
            try:
                updates[key] = int(raw)
            except ValueError:
                self.notify(
                    t(f"{label} must be an integer", f"{label} 은(는) 정수여야 합니다"),
                    severity="error",
                )
                return
        if ctk:
            updates["cache-type-k"] = ctk
        else:
            removals.add("cache-type-k")
        if ctv:
            updates["cache-type-v"] = ctv
        else:
            removals.add("cache-type-v")
        if flash_attn:
            updates["flash-attn"] = True
        else:
            removals.add("flash-attn")
        if jinja:
            updates["jinja"] = True
        else:
            removals.add("jinja")
        if ot:
            updates["override-tensors"] = [ot]
        else:
            removals.add("override-tensors")

        try:
            with profile_store.quick_setup_transaction(
                name_raw,
                "llamacpp",
                CONFIG_DIR,
            ) as final_name:
                params: dict[str, Any] = {}
                disabled_params: dict[str, Any] = {}
                if copy_from:
                    source = CONFIG_DIR / f"{copy_from}.yaml"
                    if not source.exists():
                        raise ValueError(f"config not found: {source}")
                    copied = load_config(copy_from)
                    params.update(copied.params)
                    disabled_params = dict(copied.disabled_params)
                for key in removals:
                    params.pop(key, None)
                params.update(updates)
                params.setdefault("alias", final_name)
                config = Config(
                    name=final_name,
                    params=params,
                    disabled_params=disabled_params,
                )
                profile = Profile(
                    name=final_name,
                    container_name=final_name,
                    port=port_num,
                    gpu_id=gpu,
                    config_name=final_name,
                    model_file=gguf_file,
                    hf_repo=repo,
                    hf_file=gguf_file,
                )
                save_config(config, template=source.read_text() if copy_from else None)
                save_profile(profile)
        except (OSError, RuntimeError, ValueError) as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return

        self.notify(
            t(
                f"✓ Created: {final_name}  (next: press 'u' to start — first run auto-downloads the GGUF)",
                f"✓ 생성: {final_name}  (다음: 'u' 로 시작 — 처음이면 GGUF 자동 다운로드)",
            ),
            severity="information",
            timeout=8,
        )
        self.dismiss(final_name)
