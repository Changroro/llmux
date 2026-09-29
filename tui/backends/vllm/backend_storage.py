"""vLLM backend profile/config I/O — delegates profiles to the shared YAML store."""

from __future__ import annotations

from typing import Any

import yaml

from tui.common import profile_store
from tui.common.config_markers import (
    dump_active_config,
    load_yaml_mapping,
    parse_disabled_markers,
    render_disabled_markers,
)
from tui.common.env import parse_env_file as _parse_env_file  # noqa: F401 — re-exported for callers

from .backend_common import CONFIG_DIR, Config, Profile


def _to_profile(stored: profile_store.StoredProfile) -> Profile:
    return Profile(
        name=stored.name,
        container_name=stored.container_name or stored.name,
        port=str(stored.port),
        gpu_id=stored.gpu_id,
        tensor_parallel=str(stored.tensor_parallel_size),
        config_name=stored.config_name,
        model_id=stored.model_id,
        enable_lora="true" if stored.enable_lora else "false",
        max_loras=str(stored.max_loras) if stored.max_loras is not None else "",
        max_lora_rank=str(stored.max_lora_rank) if stored.max_lora_rank is not None else "",
        lora_modules=stored.lora_modules,
        extra_pip_packages=stored.extra_pip_packages,
        image_tag=stored.image_tag,
        env_vars=dict(stored.env_vars),
    )


def _to_stored(profile: Profile) -> profile_store.StoredProfile:
    return profile_store.StoredProfile(
        name=profile.name,
        backend="vllm",
        container_name=profile.container_name or profile.name,
        port=int(profile.port or 8000),
        gpu_id=profile.gpu_id or "0",
        config_name=profile.config_name,
        tensor_parallel_size=int(profile.tensor_parallel or 1),
        model_id=profile.model_id,
        enable_lora=(profile.enable_lora or "false").lower() == "true",
        max_loras=int(profile.max_loras) if str(profile.max_loras).strip() else None,
        max_lora_rank=int(profile.max_lora_rank) if str(profile.max_lora_rank).strip() else None,
        lora_modules=profile.lora_modules,
        image_tag=profile.image_tag,
        extra_pip_packages=profile.extra_pip_packages or "",
        env_vars=dict(profile.env_vars),
    )


def load_profile(name: str) -> Profile:
    stored = profile_store.load_profile(name, "vllm")
    if stored is None:
        return Profile(name=name)
    profile = _to_profile(stored)
    profile._stored_snapshot = stored
    return profile


def save_profile(profile: Profile) -> None:
    stored = _to_stored(profile)
    expected = getattr(profile, "_stored_snapshot", None)
    if expected is None:
        profile_store.create_profile(stored)
    else:
        stored = profile_store.replace_profile(
            expected.name,
            stored,
            expected=expected,
        )
    profile.container_name = stored.container_name or stored.name
    profile._stored_snapshot = stored


def delete_profile(name: str, delete_config: bool = False) -> None:
    if delete_config:
        profile_store.delete_profile_with_config(name, "vllm", CONFIG_DIR)
        return
    profile_store.delete_profile(name, "vllm")


def list_profile_names() -> list[str]:
    return profile_store.list_profile_names("vllm")


def load_config(name: str) -> Config:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return Config(name=name)

    text = path.read_text()
    data = load_yaml_mapping(text, path)
    model = data.pop("model", "")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{path}: model must be a non-empty string")
    gpu_mem = data.pop("gpu-memory-utilization", 0.9)
    if isinstance(gpu_mem, str):
        try:
            parsed_gpu_mem = float(gpu_mem)
        except ValueError as exc:
            raise ValueError(
                f"{path}: gpu-memory-utilization must be a number"
            ) from exc
    elif isinstance(gpu_mem, bool) or not isinstance(gpu_mem, (int, float)):
        raise ValueError(f"{path}: gpu-memory-utilization must be a number")
    else:
        parsed_gpu_mem = float(gpu_mem)
    if not 0 < parsed_gpu_mem <= 1:
        raise ValueError(
            f"{path}: gpu-memory-utilization must be greater than 0 and at most 1"
        )
    extra = {str(key): value for key, value in data.items()}
    # A disabled marker whose key is also an active key is ignored — active wins.
    disabled = {
        k: v for k, v in parse_disabled_markers(text).items() if k not in extra
    }
    config = Config(
        name=name,
        model=model,
        gpu_memory_utilization=str(gpu_mem),
        extra_params=extra,
        disabled_params=disabled,
    )
    config._source_text = text
    return config


def serialize_config(config: Config, existing: str | None = None) -> str:
    gpu_mem: Any = config.gpu_memory_utilization
    if existing:
        existing_data = load_yaml_mapping(existing, config.path)
        existing_gpu_mem = existing_data.get("gpu-memory-utilization")
        if str(existing_gpu_mem) == str(gpu_mem):
            gpu_mem = existing_gpu_mem
    data: dict[str, Any] = {
        "model": config.model,
        "gpu-memory-utilization": gpu_mem,
    }
    for key, value in config.extra_params.items():
        data[key] = True if value == "" else value
    text = dump_active_config(existing, data)
    return text + render_disabled_markers(config.disabled_params)


def save_config(config: Config, *, template: str | None = None) -> None:
    if not isinstance(config.model, str) or not config.model.strip():
        raise ValueError("model must be a non-empty string")
    gpu_mem = config.gpu_memory_utilization
    try:
        parsed_gpu_mem = float(gpu_mem)
    except (TypeError, ValueError) as exc:
        raise ValueError("gpu-memory-utilization must be a number") from exc
    if isinstance(gpu_mem, bool) or not 0 < parsed_gpu_mem <= 1:
        raise ValueError(
            "gpu-memory-utilization must be greater than 0 and at most 1"
        )
    with profile_store.storage_transaction():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        existing = config.path.read_text() if config.path.exists() else None
        source_text = getattr(config, "_source_text", None)
        if source_text is not None and existing != source_text:
            raise ValueError(
                f"config {config.name!r} changed since it was loaded; reopen it and retry"
            )
        profile_store._atomic_write(
            config.path,
            serialize_config(config, existing if existing is not None else template),
        )
        config._source_text = config.path.read_text()


def parse_config_param_value(raw_value: str) -> Any:
    if raw_value == "":
        return True
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML value: {exc}") from exc


def format_config_param_value(value: Any) -> str:
    if value is True:
        return ""
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(
            value,
            default_flow_style=True,
            allow_unicode=True,
            sort_keys=False,
        ).strip()
    return str(value)


def delete_config(name: str) -> None:
    with profile_store.storage_transaction():
        path = CONFIG_DIR / f"{name}.yaml"
        if path.exists():
            path.unlink()


def list_config_names() -> list[str]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(
        path.stem for path in CONFIG_DIR.glob("*.yaml") if path.stem != "example"
    )
