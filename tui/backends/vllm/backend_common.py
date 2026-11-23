"""Shared backend constants, validation, and dataclasses."""

from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

def _resolve_project_root() -> Path:
    env_root = os.environ.get("LLMUX_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    cwd = Path.cwd()
    if (cwd / "profiles.example.yaml").exists() and (cwd / "compose").is_dir():
        return cwd
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _resolve_project_root()
SCRIPT_DIR = PROJECT_ROOT
RUNTIME_DIR = PROJECT_ROOT / ".runtime" / "vllm"
CONFIG_DIR = PROJECT_ROOT / "config" / "vllm"
COMPOSE_DIR = PROJECT_ROOT / "compose" / "vllm"
COMMON_ENV = PROJECT_ROOT / ".env.common"
VLLM_SRC_DIR = PROJECT_ROOT / ".vllm-src"
DEFAULT_VLLM_REPO_URL = "https://github.com/vllm-project/vllm.git"


# llmux owns these `vllm serve` flags: the container always listens on 0.0.0.0:8000
# behind the profile port, and --tensor-parallel-size comes from the profile.
RESERVED_CONFIG_KEYS = frozenset({"port", "host", "tensor-parallel-size"})


def reserved_config_key_error(keys) -> str:
    found = sorted({str(k) for k in keys if str(k).replace("_", "-") in RESERVED_CONFIG_KEYS})
    if not found:
        return ""
    return (
        f"{', '.join(found)} cannot be set in a vLLM config — llmux sets them from the "
        "profile (port: `llmux profile edit <p> --port N`, tensor parallel: "
        "`--tensor-parallel N`; host is fixed inside the container)"
    )


def validate_name(name: str) -> bool:
    """compose-safe lowercase name (docker compose project names are
    lowercase-only). Also prevents argv/path injection."""
    return bool(re.match(r"^[a-z0-9][a-z0-9_-]*$", name))


@dataclass
class Profile:
    name: str
    container_name: str = ""
    port: str = "8000"
    gpu_id: str = "0"
    tensor_parallel: str = "1"
    config_name: str = ""
    model_id: str = ""
    enable_lora: str = "false"
    max_loras: str = ""
    max_lora_rank: str = ""
    lora_modules: str = ""
    extra_pip_packages: str = ""
    image_tag: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        """Runtime .env path rendered from profiles.yaml."""
        return RUNTIME_DIR / f"{self.name}.env"


@dataclass
class Config:
    name: str
    model: str = ""
    gpu_memory_utilization: str = "0.9"
    extra_params: dict[str, Any] = field(default_factory=dict)
    # Params kept but not passed to the server — stored as comment markers.
    disabled_params: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return CONFIG_DIR / f"{self.name}.yaml"


@dataclass
class ContainerStatus:
    profile_name: str
    container_name: str
    running: bool = False
    status_text: str = "stopped"
    health: str = ""
    port: str = ""
    gpu_id: str = ""
    image: str = ""
    model: str = ""
    lora: bool = False


from tui.common.docker import GpuInfo  # noqa: F401 — re-exported from common


@dataclass
class DockerImage:
    repository: str
    tag: str
    size: str
    created: str
