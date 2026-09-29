#!/usr/bin/env python3

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import yaml

_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CHECKOUT_ROOT))
from tui.common import profile_store  # noqa: E402
from tui.common.config_markers import load_yaml_mapping  # noqa: E402
from tui.common.prepare import resolve_gguf_file  # noqa: E402

ROOT = profile_store.PROJECT_ROOT
CONFIG_DIR = ROOT / "config" / "llamacpp"
RUNTIME_DIR = ROOT / ".runtime" / "llamacpp"

_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MANAGED_FLAGS = frozenset(
    {"--host", "--port", "--webui", "--no-webui", "--metrics"}
)


def _validate_name(name: str, kind: str) -> str:
    if not name or not _SAFE_NAME.match(name) or ".." in name:
        raise SystemExit(
            f"잘못된 {kind} 이름: {name!r} (허용: 소문자, 숫자, _, -)"
        )
    return name


def render_command(
    cfg: dict,
    *,
    model_file: str = "",
    hf_repo: str = "",
    hf_file: str = "",
) -> list[str]:
    args: list[str] = [
        "--host", "0.0.0.0", "--port", "8080", "--no-webui", "--metrics",
    ]

    configured_model_file = cfg.pop("model-file", "")
    resolved_file = resolve_gguf_file(configured_model_file, model_file, hf_file)
    resolved_repo = hf_repo.strip()
    if not resolved_repo:
        raise ValueError(
            "hf_repo is empty — set profile.hf_repo so llama-server can `-hf` download"
        )
    if not resolved_file:
        raise ValueError("hf_file / model_file is empty — need a specific .gguf filename")
    args.extend(["-hf", resolved_repo])
    args.extend(["-hff", resolved_file])
    cfg.pop("model-file", None)

    for key in ("no-webui", "webui", "metrics", "host", "port"):
        if key in cfg:
            print(
                f"warning: config key '{key}' is managed by llmux and was ignored",
                file=sys.stderr,
            )
            cfg.pop(key, None)

    override_tensors = _string_list(
        cfg.pop("override-tensors", None), "override-tensors"
    )
    extra_args_value = cfg.pop("extra-args", None)
    if isinstance(extra_args_value, str):
        extra_args = shlex.split(extra_args_value)
    else:
        extra_args = _string_list(extra_args_value, "extra-args")
    _validate_extra_args(extra_args)

    _BOOL_VALUE_FLAGS = {"flash-attn"}

    for key, value in cfg.items():
        if not isinstance(key, str):
            raise ValueError(
                f"config keys must be strings; got {type(key).__name__}"
            )
        flag = f"--{key}"
        if isinstance(value, bool):
            if key in _BOOL_VALUE_FLAGS:
                args.extend([flag, "on" if value else "off"])
            elif value:
                args.append(flag)
        elif isinstance(value, list):
            for item in value:
                args.extend([flag, _scalar_text(item, key)])
        elif value is None:
            continue
        else:
            args.extend([flag, _scalar_text(value, key)])

    for pattern in override_tensors:
        args.extend(["-ot", str(pattern)])

    args.extend(str(a) for a in extra_args)
    return args


def _scalar_text(value: object, key: str) -> str:
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(
            f"config key '{key}' must be a scalar or list of scalar values; "
            f"got {type(value).__name__}"
        )
    return str(value)


def _string_list(value: object, key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a string or a list of strings")
    return value


def _validate_extra_args(extra_args: list[str]) -> None:
    for token in extra_args:
        option = token.split("=", 1)[0]
        if option in _MANAGED_FLAGS:
            raise ValueError(f"extra-args cannot override llmux-managed option {option}")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"사용법: {sys.argv[0]} <profile-name>", file=sys.stderr)
        return 2

    profile = _validate_name(sys.argv[1], "profile")
    with profile_store.storage_transaction():
        stored = profile_store.load_profile(profile, "llamacpp")
        if stored is None:
            print(f"프로필 없음: llamacpp/{profile} (profiles.yaml 확인)", file=sys.stderr)
            return 1

        config_name = _validate_name(stored.config_name or profile, "config")
        config_path = CONFIG_DIR / f"{config_name}.yaml"
        if not config_path.exists():
            print(f"config 파일 없음: {config_path}", file=sys.stderr)
            return 1

        try:
            cfg = load_yaml_mapping(config_path.read_text(), config_path)
        except ValueError as exc:
            print(f"config 형식 오류: {exc}", file=sys.stderr)
            return 1
        try:
            command = render_command(
                cfg,
                model_file=stored.model_file,
                hf_repo=stored.hf_repo,
                hf_file=stored.hf_file,
            )
        except ValueError as exc:
            print(f"config 렌더 실패: {exc}", file=sys.stderr)
            return 1

        override = {
            "services": {
                "llama-server": {
                    "command": command,
                },
            },
        }

        out_path = RUNTIME_DIR / f"override-{profile}.yaml"
        header = (
            "# AUTO-GENERATED by scripts/llamacpp/render-override.py — 직접 수정 금지.\n"
            f"# Profile: {profile}  Config: {config_name}\n"
        )
        profile_store._atomic_write(
            out_path,
            header + yaml.safe_dump(override, sort_keys=False, width=200),
            mode=0o600,
        )
        profile_store.render_env(stored)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
