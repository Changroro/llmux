"""Interactive first-run `.env.common` setup."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tui.common.env import validate_common_env
from tui.common.profile_store import PROJECT_ROOT

COMMON_ENV = PROJECT_ROOT / ".env.common"
COMMON_ENV_EXAMPLE = PROJECT_ROOT / ".env.common.example"


def needs_onboarding() -> bool:
    if not COMMON_ENV.exists():
        return True
    ok, _messages = validate_common_env(COMMON_ENV)
    return not ok


def _render_env(overrides: dict[str, str]) -> str:
    lines: list[str] = []
    for line in COMMON_ENV_EXAMPLE.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in overrides:
                lines.append(f"{key}={overrides[key]}")
                continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def _write_common_env(content: str) -> None:
    COMMON_ENV.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{COMMON_ENV.name}.", dir=COMMON_ENV.parent
    )
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, COMMON_ENV)
        COMMON_ENV.chmod(0o600)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        tmp_path.unlink(missing_ok=True)
        raise


def run_onboarding() -> bool:
    """Write and validate `.env.common` from interactive answers."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()

    if COMMON_ENV.exists():
        ok, messages = validate_common_env(COMMON_ENV)
        if not ok:
            console.print("\n".join(messages), markup=False)
            console.print(f"Fix {COMMON_ENV}; existing settings were preserved.", markup=False)
        return ok

    if not COMMON_ENV_EXAMPLE.exists():
        return False

    console.print(
        Panel(
            "No [bold].env.common[/bold] yet — let's set up llmux.\n"
            "Press Enter to accept each default shown in [dim][brackets][/dim].",
            title="llmux onboarding",
            border_style="cyan",
        )
    )

    try:
        hf_cache = Prompt.ask(
            "HuggingFace cache directory",
            default=str(Path.home() / ".cache" / "huggingface"),
            console=console,
        )
        model_dir = Prompt.ask(
            "llama.cpp GGUF model directory (legacy detection / disk view only — "
            "no longer mounted into the container)",
            default=str(PROJECT_ROOT / "models"),
            console=console,
        )
        hf_token = Prompt.ask(
            "HuggingFace token (optional — needed for gated models)",
            default="",
            console=console,
            password=True,
        )
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Onboarding cancelled.[/yellow]")
        return False

    hf_cache_path = str(Path(hf_cache).expanduser())
    model_dir_path = str(Path(model_dir).expanduser())

    if not os.path.isabs(hf_cache_path):
        console.print(
            f"[red]HF_CACHE_PATH must be an absolute path:[/red] {hf_cache_path}"
        )
        return False
    if not os.path.isabs(model_dir_path):
        console.print(
            f"[red]MODEL_DIR must be an absolute path:[/red] {model_dir_path}"
        )
        return False

    try:
        _write_common_env(
            _render_env(
                {
                    "HF_CACHE_PATH": hf_cache_path,
                    "MODEL_DIR": model_dir_path,
                    "HF_TOKEN": hf_token.strip(),
                }
            )
        )
    except OSError as exc:
        console.print(f"[red]Could not write {COMMON_ENV}:[/red] {exc}")
        return False

    # Pre-create bind targets before Docker can create root-owned directories.
    for path in (hf_cache_path, model_dir_path):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            console.print(
                f"[yellow]⚠ Could not create {path}: {exc}[/yellow]\n"
                "[yellow]  Create it yourself before starting a container.[/yellow]"
            )

    ok, messages = validate_common_env(COMMON_ENV)
    if not ok:
        console.print("[red]" + "\n".join(messages) + "[/red]")
        try:
            COMMON_ENV.unlink()
        except OSError as exc:
            console.print(
                f"[red]Could not remove invalid {COMMON_ENV}:[/red] {exc}\n"
                "[yellow]Onboarding will retry because the file remains invalid.[/yellow]"
            )
        return False

    console.print(
        Panel(
            f"[green]✓[/green] Wrote [bold]{COMMON_ENV}[/bold]\n"
            "Edit it any time to tweak LoRA paths, image tags, or timezone.",
            border_style="green",
        )
    )
    return True
