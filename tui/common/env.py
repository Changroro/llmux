"""Shared parsing + validation for .env-style files.

Single source of truth for what every backend treats as a well-formed
`.env.common` — one parser and one validator for both backends, so
HF_CACHE_PATH is checked up front rather than surfacing as a late compose
interpolation failure.
"""

from __future__ import annotations

import os
from pathlib import Path

# Escapes interpreted inside a double-quoted value (godotenv-compatible subset).
_DQ_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}


def _read_double_quoted(s: str) -> str:
    """Body of a `"..."` value: honor \\n/\\r/\\t/\\"/\\\\, drop the rest."""
    out: list[str] = []
    i = 1  # skip opening quote
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            out.append(_DQ_ESCAPES.get(s[i + 1], "\\" + s[i + 1]))
            i += 2
            continue
        if ch == '"':
            break  # closing quote — anything after (incl. comments) is dropped
        out.append(ch)
        i += 1
    return "".join(out)


def parse_env_file(path: Path | str, *, expand: bool = False) -> dict[str, str]:
    """Parse a .env file the way docker compose (godotenv) does.

    Rules: split on the first `=`; strip surrounding whitespace. A value that
    opens with `"` runs to the matching `"` (with `\\n`/`\\r`/`\\t`/`\\"`/`\\\\`
    escapes) and anything after it — comments included — is discarded; a `'`
    value is literal to the matching `'`. An unquoted value keeps backslashes
    verbatim and has an inline ` #` comment trimmed. This matters because these
    values are also injected into the process env, where they *outrank*
    compose's own `--env-file` read — so a mismatch here silently overrides
    compose. Missing file → empty dict, never raises.
    """
    p = Path(path)
    data: dict[str, str] = {}
    if not p.exists():
        return data
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        # `export KEY=value` — dotenv/godotenv strip the leading keyword; without
        # this the key became `export KEY` and the real var was silently lost.
        if line[:7] in ("export ", "export\t"):
            line = line[7:].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not value:
            data[key] = ""
            continue
        if value[0] == '"':
            data[key] = _read_double_quoted(value)
        elif value[0] == "'":
            end = value.find("'", 1)
            data[key] = value[1:end] if end != -1 else value[1:]
        else:
            if " #" in value:
                value = value[: value.index(" #")]
            data[key] = value.rstrip()
        if expand and not value.startswith("'"):
            data[key] = host_expand(data[key])
    return data


def host_expand(value: str) -> str:
    """Expand `$VAR` and `~` the way a shell would, for host-side paths."""
    return os.path.expanduser(os.path.expandvars(value))


def expand_env_values(env: dict[str, str]) -> dict[str, str]:
    """Expand `$VAR`/`~` in values read from a .env file.

    docker compose expands these itself when it reads an `--env-file`, but we
    also merge the same values into the *process* env — and process env wins
    over --env-file in compose's precedence order. Passing them through raw
    meant `.env.common.example`'s default `HF_CACHE_PATH=/home/$USER/.cache/...`
    reached compose unexpanded and got bind-mounted as a literal `/home/$USER`
    directory. Only apply this to file-derived values; os.environ is already
    expanded by the shell.
    """
    return {k: host_expand(v) for k, v in env.items()}


def required_image_reference_error(key: str, image_ref: str) -> str:
    if not image_ref.strip():
        return f"Error: {key} is not set in .env.common"
    from tui.common.dev_build import image_tag_error

    error = image_tag_error(image_ref)
    return f"Error: invalid {key}: {error}" if error else ""


def validate_common_env(
    common_env_path: Path,
    *,
    require_lora_base_path: bool = False,
    require_downloader_image: bool = False,
) -> tuple[bool, list[str]]:
    """Validate `.env.common` for shared cross-backend invariants.

    Returns `(ok, messages)`. Messages are user-facing lines (already prefixed
    with `Error:` or `Warning:` where appropriate) suitable for yielding to a
    log stream.

    Currently enforced:
      * `.env.common` exists.
      * `HF_CACHE_PATH` is set and absolute (compose mounts it directly into
        `/root/.cache/huggingface`, so a missing or relative value fails late
        with a cryptic docker error).
      * When `require_lora_base_path=True` (vLLM with `ENABLE_LORA=true`),
        `LORA_BASE_PATH` is set and absolute.
    """
    if not common_env_path.exists():
        return False, [
            "Error: .env.common not found.",
            "Create it from .env.common.example before starting containers.",
        ]
    try:
        mode = common_env_path.stat().st_mode & 0o777
    except OSError as exc:
        return False, [f"Error: cannot inspect .env.common permissions: {exc}"]
    if mode & 0o077:
        return False, [
            "Error: .env.common permissions must be owner-only "
            f"(for example 0600). Current mode: {mode:04o}"
        ]
    common = parse_env_file(common_env_path)
    hf_cache_path = common.get("HF_CACHE_PATH", "")
    if not hf_cache_path:
        return False, ["Error: HF_CACHE_PATH is not set in .env.common"]
    if not os.path.isabs(host_expand(hf_cache_path)):
        return False, [
            f"Error: HF_CACHE_PATH must be an absolute path. Current value: {hf_cache_path}"
        ]
    workers = common.get("PREPARE_MAX_WORKERS", "").strip()
    if workers and (not workers.isdigit() or int(workers) < 1):
        return False, [
            "Error: PREPARE_MAX_WORKERS must be empty or a positive integer. "
            f"Current value: {workers}"
        ]
    if require_downloader_image:
        downloader_error = required_image_reference_error(
            "PREPARE_DOWNLOADER_IMAGE",
            common.get("PREPARE_DOWNLOADER_IMAGE", ""),
        )
        if downloader_error:
            return False, [downloader_error]

    from tui.common.dev_build import image_tag_error

    for key in ("VLLM_IMAGE", "LLAMACPP_IMAGE"):
        image_ref = common.get(key, "").strip()
        if not image_ref:
            continue
        error = image_tag_error(image_ref)
        if error:
            return False, [f"Error: invalid {key}: {error}"]
    if require_lora_base_path:
        lora_base_path = common.get("LORA_BASE_PATH", "")
        if not lora_base_path:
            return False, [
                "Error: ENABLE_LORA=true but LORA_BASE_PATH is not set in .env.common"
            ]
        if not os.path.isabs(host_expand(lora_base_path)):
            return False, [
                f"Error: LORA_BASE_PATH must be an absolute path. Current value: {lora_base_path}"
            ]
    messages: list[str] = []
    if "VLLM_USE_V2_MODEL_RUNNER" not in common:
        # compose defaults it to 1, which rejects thinking_token_budget requests
        # with HTTP 400 — indistinguishable from "the model doesn't support it".
        messages.append(
            "Warning: VLLM_USE_V2_MODEL_RUNNER is not set in .env.common; "
            "compose will use 1. Set it to 0 if you need thinking_token_budget."
        )
    return True, messages
