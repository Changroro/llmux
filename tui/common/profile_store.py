from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import fcntl
import os
import re
import shlex
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import yaml

from tui.common.config_markers import load_yaml_mapping

def _resolve_project_root() -> Path:
    env_root = os.environ.get("LLMUX_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not (root / "compose").is_dir():
            raise RuntimeError(
                f"LLMUX_ROOT={env_root} does not look like an llmux checkout "
                f"({root / 'compose'} is missing)."
            )
        return root
    cwd = Path.cwd()
    if (cwd / "profiles.example.yaml").exists() and (cwd / "compose").is_dir():
        return cwd
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _resolve_project_root()
PROFILES_YAML = PROJECT_ROOT / "profiles.yaml"
RUNTIME_DIR = PROJECT_ROOT / ".runtime"

Backend = Literal["vllm", "llamacpp"]

DEFAULTS: dict[str, dict[str, Any]] = {
    "vllm": {
        "port": 8000,
        "gpu_id": "0",
        "tensor_parallel_size": 1,
        "enable_lora": False,
    },
    "llamacpp": {
        "port": 8080,
        "gpu_id": "0",
    },
}

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_LEGACY_CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_GPU_ID_RE = re.compile(r"^[0-9]+(,[0-9]+)*$")
_HF_REPO_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_SCHEMA_VERSION = 1
_TOP_LEVEL_KEYS = frozenset({"version", "defaults", "profiles"})
_SHARED_PROFILE_KEYS = frozenset(
    {
        "name",
        "backend",
        "container_name",
        "port",
        "gpu_id",
        "config_name",
        "env_vars",
        "image_tag",
    }
)
_PROFILE_KEYS_BY_BACKEND = {
    "vllm": _SHARED_PROFILE_KEYS
    | {
        "tensor_parallel_size",
        "model_id",
        "enable_lora",
        "max_loras",
        "max_lora_rank",
        "lora_modules",
        "extra_pip_packages",
    },
    "llamacpp": _SHARED_PROFILE_KEYS
    | {"model_file", "hf_repo", "hf_file"},
}
_PROTECTED_ENV_KEYS = frozenset(
    {
        "HF_TOKEN",
        "HF_ENDPOINT",
    }
)
_PROTECTED_ENV_PREFIXES = (
    "DOCKER_",
    "COMPOSE_",
)
REDACTED_ENV_VALUE = "<redacted>"
_SENSITIVE_ENV_KEY_PARTS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "PRIVATE_KEY",
    "CREDENTIAL",
    "AUTH",
)

# env_vars cannot override fields emitted into the same Compose dotenv file.
_RESERVED_ENV_KEYS_BY_BACKEND: dict[str, frozenset[str]] = {
    "vllm": frozenset(
        {
            "CONTAINER_NAME",
            "VLLM_PORT",
            "GPU_ID",
            "TENSOR_PARALLEL_SIZE",
            "CONFIG_NAME",
            "MODEL_ID",
            "ENABLE_LORA",
            "MAX_LORAS",
            "MAX_LORA_RANK",
            "LORA_MODULES",
            "EXTRA_PIP_PACKAGES",
            "VLLM_IMAGE",
        }
    ),
    "llamacpp": frozenset(
        {
            "CONTAINER_NAME",
            "LLAMA_PORT",
            "GPU_ID",
            "CONFIG_NAME",
            "MODEL_FILE",
            "HF_REPO",
            "HF_FILE",
            "LLAMACPP_IMAGE",
            "LLAMACPP_DEV_TAG",
        }
    ),
}


def effective_defaults(backend: str) -> dict[str, Any]:
    return _backend_defaults(_load_yaml(), backend)


def reserved_env_keys(backend: str) -> frozenset[str]:
    return _RESERVED_ENV_KEYS_BY_BACKEND.get(backend, frozenset()) | _PROTECTED_ENV_KEYS


def is_protected_profile_env_key(key: str, backend: str) -> bool:
    return key in reserved_env_keys(backend) or key.startswith(_PROTECTED_ENV_PREFIXES)


def sensitive_env_key(key: str) -> bool:
    normalized = key.upper()
    return any(part in normalized for part in _SENSITIVE_ENV_KEY_PARTS)


def hf_repo_error(value: str) -> str:
    if not value:
        return "HF repo must be a canonical org/name path"
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return "HF repo cannot include control characters"
    parts = value.split("/")
    if (
        len(parts) != 2
        or any(part in {"", ".", ".."} for part in parts)
        or any(_HF_REPO_SEGMENT_RE.fullmatch(part) is None for part in parts)
    ):
        return "HF repo must be a canonical org/name path"
    return ""


def parse_env_vars_text(
    text: str,
    backend: str,
    existing: dict[str, str] | None = None,
) -> dict[str, str]:
    if backend not in DEFAULTS:
        raise ValueError(f"invalid backend {backend!r}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        if "=" not in raw:
            raise ValueError(f"environment line {line_number} must be KEY=VALUE")
        key, _, value = raw.partition("=")
        key = key.strip()
        if not _ENV_KEY_RE.fullmatch(key):
            raise ValueError(f"invalid environment variable name on line {line_number}: {key!r}")
        if is_protected_profile_env_key(key, backend):
            raise ValueError(
                f"environment variable {key!r} is managed by the profile fields"
            )
        if value == REDACTED_ENV_VALUE:
            if existing is None or key not in existing:
                raise ValueError(
                    f"environment value for {key!r} uses the reserved redaction marker"
                )
            result[key] = existing[key]
            continue
        reason = env_value_rejection(value)
        if reason:
            raise ValueError(f"environment value for {key!r} contains a {reason}")
        result[key] = value
    return result


def format_env_vars_text(env_vars: dict[str, str]) -> str:
    return "\n".join(
        f"{key}={REDACTED_ENV_VALUE}" for key in sorted(env_vars)
    )


@dataclass
class StoredProfile:
    name: str
    backend: str
    container_name: str = ""
    port: int = 0
    gpu_id: str = "0"
    config_name: str = ""
    tensor_parallel_size: int = 1
    model_id: str = ""
    enable_lora: bool = False
    max_loras: int | None = None
    max_lora_rank: int | None = None
    lora_modules: str = ""
    extra_pip_packages: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    model_file: str = ""
    hf_repo: str = ""
    hf_file: str = ""
    image_tag: str = ""


def _load_yaml() -> dict:
    if not PROFILES_YAML.exists():
        return {"version": 1, "defaults": DEFAULTS, "profiles": []}
    raw = load_yaml_mapping(PROFILES_YAML.read_text(), PROFILES_YAML)
    if not raw:
        raise ValueError(f"{PROFILES_YAML} is empty")
    unknown_root = set(raw) - _TOP_LEVEL_KEYS
    if unknown_root:
        raise ValueError(
            f"{PROFILES_YAML}: unknown top-level key(s): "
            f"{', '.join(sorted(map(str, unknown_root)))}"
        )
    version = raw.get("version")
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise ValueError(
            f"{PROFILES_YAML}: unsupported version {version!r}; "
            f"expected {_SCHEMA_VERSION}"
        )
    raw.setdefault("defaults", DEFAULTS)
    raw.setdefault("profiles", [])
    defaults = raw["defaults"]
    if not isinstance(defaults, dict):
        raise ValueError(f"{PROFILES_YAML}: defaults must be a mapping")
    unknown = set(defaults) - set(DEFAULTS)
    if unknown:
        raise ValueError(
            f"{PROFILES_YAML}: unknown defaults backend(s): "
            f"{', '.join(sorted(map(str, unknown)))}"
        )
    for backend, values in defaults.items():
        if not isinstance(values, dict):
            raise ValueError(f"{PROFILES_YAML}: defaults.{backend} must be a mapping")
        unknown_fields = set(values) - set(DEFAULTS[backend])
        if unknown_fields:
            raise ValueError(
                f"{PROFILES_YAML}: unknown defaults.{backend} key(s): "
                f"{', '.join(sorted(map(str, unknown_fields)))}"
            )
        for key, value in values.items():
            expected = type(DEFAULTS[backend][key])
            if type(value) is not expected:
                raise ValueError(
                    f"{PROFILES_YAML}: defaults.{backend}.{key} must be "
                    f"{expected.__name__}"
                )
    return raw


def _backend_defaults(data: dict, backend: str) -> dict[str, Any]:
    if backend not in DEFAULTS:
        raise ValueError(f"Invalid backend: {backend!r}")
    defaults = dict(DEFAULTS[backend])
    user_defaults = data.get("defaults", {})
    if backend in user_defaults:
        defaults.update(user_defaults[backend])
    return defaults


def _dump_yaml(data: dict) -> str:
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=120,
    )


def _atomic_write(path: Path, content: str, *, mode: int | None = None) -> None:
    tmp = _stage_bytes(path, content.encode(), mode=mode)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _stage_bytes(path: Path, content: bytes, *, mode: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        target_mode = (
            mode
            if mode is not None
            else path.stat().st_mode & 0o777
            if path.exists()
            else 0o600
        )
        os.fchmod(fd, target_mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return tmp
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        tmp.unlink(missing_ok=True)
        raise


def _replace_files(
    writes: dict[Path, str],
    deletes: tuple[Path, ...] = (),
    *,
    modes: dict[Path, int] | None = None,
) -> None:
    targets = [*writes, *deletes]
    originals = {
        path: (path.read_bytes(), path.stat().st_mode & 0o777)
        if path.exists()
        else None
        for path in targets
    }
    staged: dict[Path, Path] = {}
    try:
        for path, text in writes.items():
            staged[path] = _stage_bytes(
                path,
                text.encode(),
                mode=(modes or {}).get(path),
            )
    except BaseException:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)
        raise
    mutated: list[Path] = []
    try:
        for path, tmp in staged.items():
            mutated.append(path)
            os.replace(tmp, path)
        for path in deletes:
            if path.exists():
                mutated.append(path)
                path.unlink()
    except BaseException as exc:
        rollback_errors: list[str] = []
        for path in reversed(mutated):
            try:
                original = originals[path]
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    data, mode = original
                    restored = _stage_bytes(path, data)
                    os.chmod(restored, mode)
                    os.replace(restored, path)
            except BaseException as rollback_exc:
                rollback_errors.append(f"{path}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                f"storage update failed ({exc}); rollback failed: "
                + "; ".join(rollback_errors)
            ) from exc
        raise
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)


_LOCK_STATE = threading.local()
_PROCESS_LOCK = threading.RLock()


@contextmanager
def _storage_lock() -> Iterator[None]:
    with _PROCESS_LOCK:
        depth = getattr(_LOCK_STATE, "depth", 0)
        if depth:
            _LOCK_STATE.depth = depth + 1
            try:
                yield
            finally:
                _LOCK_STATE.depth -= 1
            return
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        lock_path = RUNTIME_DIR / ".profiles.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(fd, "rb+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            _LOCK_STATE.depth = 1
            try:
                yield
            finally:
                _LOCK_STATE.depth = 0
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def storage_transaction() -> Iterator[None]:
    with _storage_lock():
        yield


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"invalid {field_name} {value!r}: must be a boolean")
    raise ValueError(f"invalid {field_name} {value!r}: must be a boolean")


def _to_profile(entry: dict, defaults: dict[str, Any] | None = None) -> StoredProfile:
    backend = entry.get("backend")
    if backend not in ("vllm", "llamacpp"):
        raise ValueError(f"Invalid backend {backend!r} in profile {entry.get('name')}")
    defaults = defaults or DEFAULTS[backend]
    merged: dict[str, Any] = dict(defaults)
    merged.update(entry)
    name = merged.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("profile entry is missing a non-empty name")

    def _as_int(field_name: str, raw: Any) -> int:
        if type(raw) is not int:
            raise ValueError(
                f"profile {name!r}: invalid {field_name} {raw!r} (must be an integer)"
            )
        return raw

    raw_env_vars = merged.get("env_vars", {})
    if not isinstance(raw_env_vars, dict):
        raise ValueError(f"profile {name!r}: env_vars must be a mapping")

    profile = StoredProfile(
        name=name,
        backend=backend,
        container_name=merged.get("container_name", ""),
        port=_as_int("port", merged.get("port", defaults["port"])),
        gpu_id=merged.get("gpu_id", "0"),
        config_name=merged.get("config_name", name),
        tensor_parallel_size=_as_int(
            "tensor_parallel_size", merged.get("tensor_parallel_size", 1)
        ),
        model_id=merged.get("model_id", ""),
        enable_lora=_parse_bool(merged.get("enable_lora", False), "enable_lora"),
        max_loras=merged.get("max_loras"),
        max_lora_rank=merged.get("max_lora_rank"),
        lora_modules=merged.get("lora_modules", ""),
        extra_pip_packages=merged.get("extra_pip_packages", ""),
        env_vars=dict(raw_env_vars),
        model_file=merged.get("model_file", ""),
        hf_repo=merged.get("hf_repo", ""),
        hf_file=merged.get("hf_file", ""),
        image_tag=merged.get("image_tag", ""),
    )
    _validate_profile(profile)
    return profile


def _validate_profile(profile: StoredProfile) -> None:
    if profile.backend not in DEFAULTS:
        raise ValueError(f"invalid backend {profile.backend!r}")
    if not isinstance(profile.name, str) or not _NAME_RE.fullmatch(profile.name):
        raise ValueError(
            f"invalid profile name {profile.name!r}: must match {_NAME_RE.pattern}"
        )
    container_name = profile.container_name or profile.name
    if not isinstance(container_name, str) or not _NAME_RE.fullmatch(container_name):
        raise ValueError(
            f"invalid container name {container_name!r}: must match {_NAME_RE.pattern}"
        )
    config_name = profile.config_name or profile.name
    if not isinstance(config_name, str) or not _NAME_RE.fullmatch(config_name):
        raise ValueError(
            f"invalid config name {config_name!r}: must match {_NAME_RE.pattern}"
        )
    if type(profile.port) is not int:
        raise ValueError(f"profile {profile.name!r}: port must be an integer")
    effective_port = profile.port or DEFAULTS[profile.backend]["port"]
    if not 1024 <= effective_port <= 65535:
        raise ValueError(f"profile {profile.name!r}: port must be in 1024–65535")
    if not isinstance(profile.gpu_id, str) or not _GPU_ID_RE.fullmatch(profile.gpu_id):
        raise ValueError(
            f"profile {profile.name!r}: invalid GPU id {profile.gpu_id!r}"
        )
    if profile.backend == "vllm" and (
        type(profile.tensor_parallel_size) is not int
        or profile.tensor_parallel_size < 1
    ):
        raise ValueError(
            f"profile {profile.name!r}: tensor_parallel_size must be at least 1"
        )
    for field_name in ("max_loras", "max_lora_rank"):
        value = getattr(profile, field_name)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(
                f"profile {profile.name!r}: {field_name} must be a positive integer"
            )
    if not isinstance(profile.env_vars, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in profile.env_vars.items()
    ):
        raise ValueError(f"profile {profile.name!r}: env_vars must map strings to strings")
    protected = {
        key
        for key in profile.env_vars
        if is_protected_profile_env_key(key, profile.backend)
    }
    if protected:
        raise ValueError(
            f"profile {profile.name!r}: env_vars cannot override protected key(s): "
            f"{', '.join(sorted(protected))}"
        )
    string_fields = (
        "container_name",
        "config_name",
        "model_id",
        "lora_modules",
        "extra_pip_packages",
        "model_file",
        "hf_repo",
        "hf_file",
        "image_tag",
    )
    for field_name in string_fields:
        if not isinstance(getattr(profile, field_name), str):
            raise ValueError(
                f"profile {profile.name!r}: {field_name} must be a string"
            )
    if profile.backend == "llamacpp" and profile.hf_repo:
        error = hf_repo_error(profile.hf_repo)
        if error:
            raise ValueError(f"profile {profile.name!r}: {error}")
    if profile.image_tag:
        from tui.common.dev_build import image_tag_error

        error = image_tag_error(profile.image_tag)
        if error:
            raise ValueError(f"profile {profile.name!r}: {error}")


def _validated_profiles(data: dict) -> list[StoredProfile]:
    entries = data.get("profiles", [])
    if not isinstance(entries, list):
        raise ValueError(f"{PROFILES_YAML}: profiles must be a list")
    profiles: list[StoredProfile] = []
    names: dict[str, str] = {}
    containers: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{PROFILES_YAML}: profile entry {index} must be a mapping"
            )
        backend = entry.get("backend")
        if backend not in DEFAULTS:
            raise ValueError(
                f"{PROFILES_YAML}: profile entry {index} has invalid backend {backend!r}"
            )
        unknown_fields = set(entry) - _PROFILE_KEYS_BY_BACKEND[backend]
        if unknown_fields:
            raise ValueError(
                f"{PROFILES_YAML}: profile entry {index} has unknown key(s): "
                f"{', '.join(sorted(map(str, unknown_fields)))}"
            )
        profile = _to_profile(entry, _backend_defaults(data, backend))
        if profile.name in names:
            raise ValueError(
                f"duplicate profile name {profile.name!r}: already used by {names[profile.name]}"
            )
        container_name = profile.container_name or profile.name
        if container_name in containers:
            raise ValueError(
                f"duplicate container name {container_name!r}: already used by "
                f"{containers[container_name]}"
            )
        names[profile.name] = f"{profile.backend}/{profile.name}"
        containers[container_name] = f"{profile.backend}/{profile.name}"
        profiles.append(profile)
    return profiles


def _profile_to_entry(
    profile: StoredProfile, defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    defaults = defaults or DEFAULTS[profile.backend]
    out: dict[str, Any] = {"name": profile.name, "backend": profile.backend}
    if profile.container_name and profile.container_name != profile.name:
        out["container_name"] = profile.container_name
    if profile.port and profile.port != defaults.get("port"):
        out["port"] = profile.port
    if profile.gpu_id and profile.gpu_id != defaults.get("gpu_id", "0"):
        out["gpu_id"] = profile.gpu_id
    if profile.config_name != (defaults.get("config_name") or profile.name):
        out["config_name"] = profile.config_name

    if profile.backend == "vllm":
        if profile.tensor_parallel_size != defaults.get("tensor_parallel_size", 1):
            out["tensor_parallel_size"] = profile.tensor_parallel_size
        if profile.model_id:
            out["model_id"] = profile.model_id
        if profile.enable_lora != defaults.get("enable_lora", False):
            out["enable_lora"] = profile.enable_lora
        if profile.max_loras is not None:
            out["max_loras"] = profile.max_loras
        if profile.max_lora_rank is not None:
            out["max_lora_rank"] = profile.max_lora_rank
        if profile.lora_modules:
            out["lora_modules"] = profile.lora_modules
        if profile.extra_pip_packages:
            out["extra_pip_packages"] = profile.extra_pip_packages
    else:
        if profile.model_file:
            out["model_file"] = profile.model_file
        if profile.hf_repo:
            out["hf_repo"] = profile.hf_repo
        if profile.hf_file:
            out["hf_file"] = profile.hf_file
    if profile.env_vars:
        out["env_vars"] = dict(profile.env_vars)
    if profile.image_tag:
        out["image_tag"] = profile.image_tag
    return out


def list_profiles(backend: str) -> list[StoredProfile]:
    if backend not in DEFAULTS:
        raise ValueError(f"Invalid backend: {backend!r}")
    data = _load_yaml()
    return [p for p in _validated_profiles(data) if p.backend == backend]


def find_name_owner(
    name: str, *, exclude: tuple[str, str] | None = None
) -> str | None:
    for p in _validated_profiles(_load_yaml()):
        if exclude is not None and exclude == (p.backend, p.name):
            continue
        if p.name == name:
            return p.backend
    return None


def list_profile_names(backend: str) -> list[str]:
    return sorted(p.name for p in list_profiles(backend))


def load_profile(name: str, backend: str) -> StoredProfile | None:
    for p in list_profiles(backend):
        if p.name == name:
            return p
    return None


def effective_config_name(profile: StoredProfile) -> str:
    return profile.config_name or profile.name


def _available_quick_setup_name(
    base_name: str,
    backend: str,
    config_dir: Path,
) -> str:
    if backend not in DEFAULTS:
        raise ValueError(f"invalid backend {backend!r}")
    if not _NAME_RE.fullmatch(base_name):
        raise ValueError(f"invalid profile name {base_name!r}: must match {_NAME_RE.pattern}")
    profiles = _validated_profiles(_load_yaml())
    used = {profile.name for profile in profiles}
    used.update(profile.container_name or profile.name for profile in profiles)
    if config_dir.exists():
        used.update(path.stem for path in config_dir.glob("*.yaml"))
    final_name = base_name
    suffix = 0
    while final_name in used:
        suffix += 1
        final_name = f"{base_name}-{suffix}"
    return final_name


def _rollback_quick_setup_unlocked(
    name: str,
    backend: str,
    config_path: Path,
) -> None:
    data = _load_yaml()
    _validated_profiles(data)
    profiles = data.get("profiles", [])
    remaining = [
        entry
        for entry in profiles
        if not (
            isinstance(entry, dict)
            and entry.get("name") == name
            and entry.get("backend") == backend
        )
    ]
    writes: dict[Path, str] = {}
    if len(remaining) != len(profiles):
        data["profiles"] = remaining
        writes[PROFILES_YAML] = _dump_yaml(data)
    _replace_files(
        writes,
        deletes=(config_path, runtime_env_path(name, backend)),
        modes={PROFILES_YAML: 0o600} if PROFILES_YAML in writes else None,
    )


@contextmanager
def quick_setup_transaction(
    base_name: str,
    backend: str,
    config_dir: Path,
) -> Iterator[str]:
    with _storage_lock():
        final_name = _available_quick_setup_name(base_name, backend, config_dir)
        config_path = config_dir / f"{final_name}.yaml"
        try:
            yield final_name
        except BaseException as exc:
            try:
                _rollback_quick_setup_unlocked(final_name, backend, config_path)
            except BaseException as rollback_exc:
                raise RuntimeError(
                    f"quick setup failed ({exc}); rollback failed: {rollback_exc}"
                ) from exc
            raise


def save_profile(profile: StoredProfile) -> None:
    with _storage_lock():
        _save_profile_unlocked(profile)


def create_profile(profile: StoredProfile) -> None:
    with _storage_lock():
        if load_profile(profile.name, profile.backend) is not None:
            raise ValueError(
                f"profile {profile.name!r} already exists in backend {profile.backend!r}"
            )
        _save_profile_unlocked(profile)


def update_profile(
    name: str,
    backend: str,
    update: Callable[[StoredProfile], None],
) -> StoredProfile:
    with _storage_lock():
        profile = load_profile(name, backend)
        if profile is None:
            raise ValueError(f"profile {name!r} not found in backend {backend!r}")
        update(profile)
        _save_profile_unlocked(profile)
        return profile


def clone_profile(src: str, dst: str, backend: str) -> StoredProfile:
    with _storage_lock():
        source = next(
            (profile for profile in list_profiles(backend) if profile.name == src),
            None,
        )
        if source is None:
            raise ValueError(f"profile {src!r} not found in backend {backend!r}")
        clone = replace(
            source,
            name=dst,
            container_name=dst,
            config_name=source.config_name or src,
            env_vars=dict(source.env_vars),
        )
        create_profile(clone)
        return clone


def _save_profile_unlocked(profile: StoredProfile) -> None:
    _validate_profile(profile)
    data = _load_yaml()
    existing_profiles = _validated_profiles(data)
    effective_container = profile.container_name or profile.name
    for existing in existing_profiles:
        if (existing.backend, existing.name) == (profile.backend, profile.name):
            continue
        if existing.name == profile.name:
            raise ValueError(
                f"profile name {profile.name!r} is already used by "
                f"{existing.backend}/{existing.name}"
            )
        if (existing.container_name or existing.name) == effective_container:
            raise ValueError(
                f"container name {effective_container!r} is already used by "
                f"{existing.backend}/{existing.name}"
            )
    profiles = data.get("profiles", [])
    entry = _profile_to_entry(profile, _backend_defaults(data, profile.backend))
    for idx, existing in enumerate(profiles):
        if not isinstance(existing, dict):
            continue
        if (
            existing.get("name") == profile.name
            and existing.get("backend") == profile.backend
        ):
            profiles[idx] = entry
            break
    else:
        profiles.append(entry)
    data["profiles"] = profiles

    env_path = runtime_env_path(profile.name, profile.backend)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_lines = _render_env_lines(profile)
    env_text = "\n".join(env_lines)
    yaml_text = _dump_yaml(data)

    _replace_files(
        {env_path: env_text, PROFILES_YAML: yaml_text},
        modes={env_path: 0o600, PROFILES_YAML: 0o600},
    )


def replace_profile(
    old_name: str,
    profile: StoredProfile,
    *,
    expected: StoredProfile | None = None,
) -> StoredProfile:
    with _storage_lock():
        return _replace_profile_unlocked(old_name, profile, expected=expected)


def _replace_profile_unlocked(
    old_name: str,
    profile: StoredProfile,
    *,
    expected: StoredProfile | None = None,
) -> StoredProfile:
    _validate_profile(profile)
    data = _load_yaml()
    profiles = data.get("profiles", [])
    for index, entry in enumerate(profiles):
        if not isinstance(entry, dict):
            continue
        if (
            entry.get("name") == old_name
            and entry.get("backend") == profile.backend
        ):
            break
    else:
        raise ValueError(
            f"profile {old_name!r} not found in backend {profile.backend!r}"
        )

    defaults = _backend_defaults(data, profile.backend)
    current = _to_profile(entry, defaults)
    if expected is not None and _profile_to_entry(
        current, defaults
    ) != _profile_to_entry(expected, defaults):
        raise ValueError(
            f"profile {old_name!r} changed since it was loaded; reopen it and retry"
        )
    if "container_name" not in entry and profile.container_name in {"", old_name}:
        profile = replace(profile, container_name="")

    profiles[index] = _profile_to_entry(
        profile,
        defaults,
    )
    data["profiles"] = profiles
    _validated_profiles(data)

    new_env = runtime_env_path(profile.name, profile.backend)
    writes = {
        new_env: "\n".join(_render_env_lines(profile)),
        PROFILES_YAML: _dump_yaml(data),
    }
    old_env = runtime_env_path(old_name, profile.backend)
    deletes = (old_env,) if old_env != new_env else ()
    _replace_files(
        writes,
        deletes=deletes,
        modes={new_env: 0o600, PROFILES_YAML: 0o600},
    )
    return profile


def rename_profile(old: str, new: str, backend: str) -> StoredProfile:
    with _storage_lock():
        return _rename_profile_unlocked(old, new, backend)


def _rename_profile_unlocked(old: str, new: str, backend: str) -> StoredProfile:
    if old == new:
        raise ValueError(f"profile is already named {new!r}")
    profile = load_profile(old, backend)
    if profile is None:
        raise ValueError(f"profile {old!r} not found in backend {backend!r}")
    data = _load_yaml()
    profiles = data.get("profiles", [])
    for idx, existing in enumerate(profiles):
        if not isinstance(existing, dict):
            continue
        if existing.get("name") == old and existing.get("backend") == backend:
            break
    else:
        raise ValueError(f"profile {old!r} not found in profiles.yaml")

    if not existing.get("container_name"):
        profile.container_name = new
    if not existing.get("config_name"):
        profile.config_name = old
    profile.name = new
    return _replace_profile_unlocked(old, profile)


def delete_profile(name: str, backend: str) -> bool:
    with _storage_lock():
        return _delete_profile_unlocked(name, backend)


def _delete_profile_unlocked(name: str, backend: str) -> bool:
    data = _load_yaml()
    _validated_profiles(data)
    profiles = data.get("profiles", [])
    remaining = [
        p for p in profiles
        if not (
            isinstance(p, dict)
            and p.get("name") == name
            and p.get("backend") == backend
        )
    ]
    if len(remaining) == len(profiles):
        return False
    data["profiles"] = remaining
    rt = runtime_env_path(name, backend)
    _replace_files(
        {PROFILES_YAML: _dump_yaml(data)},
        deletes=(rt,),
        modes={PROFILES_YAML: 0o600},
    )
    return True


def delete_profile_with_config(
    name: str,
    backend: str,
    config_dir: Path,
) -> bool:
    if backend not in DEFAULTS:
        raise ValueError(f"invalid backend {backend!r}")
    with _storage_lock():
        data = _load_yaml()
        profiles = _validated_profiles(data)
        target = next(
            (
                profile
                for profile in profiles
                if (profile.backend, profile.name) == (backend, name)
            ),
            None,
        )
        if target is None:
            return False
        config_name = effective_config_name(target)
        config_path = config_dir / f"{config_name}.yaml"
        shared = any(
            profile.backend == backend
            and profile.name != name
            and effective_config_name(profile) == config_name
            for profile in profiles
        )
        entries = data.get("profiles", [])
        data["profiles"] = [
            entry
            for entry in entries
            if not (
                isinstance(entry, dict)
                and entry.get("name") == name
                and entry.get("backend") == backend
            )
        ]
        deletes = [runtime_env_path(name, backend)]
        if not shared and config_name != "example":
            deletes.append(config_path)
        _replace_files(
            {PROFILES_YAML: _dump_yaml(data)},
            deletes=tuple(deletes),
            modes={PROFILES_YAML: 0o600},
        )
        return True


def repoint_config_references(
    backend: str,
    old_config_name: str,
    new_config_name: str,
    *,
    writes: dict[Path, str] | None = None,
    deletes: tuple[Path, ...] = (),
    moves: tuple[
        tuple[Path, Path] | tuple[Path, Path, str],
        ...,
    ] = (),
) -> list[str]:
    if backend not in DEFAULTS:
        raise ValueError(f"invalid backend {backend!r}")
    if not _LEGACY_CONFIG_NAME_RE.fullmatch(old_config_name):
        raise ValueError(
            f"invalid existing config name {old_config_name!r}: must match "
            f"{_LEGACY_CONFIG_NAME_RE.pattern}"
        )
    if new_config_name and not _NAME_RE.fullmatch(new_config_name):
        raise ValueError(
            f"invalid config name {new_config_name!r}: must match {_NAME_RE.pattern}"
        )

    with _storage_lock():
        file_writes = dict(writes or {})
        file_deletes = list(deletes)
        file_modes: dict[Path, int] = {}
        for move in moves:
            if len(move) == 2:
                source, destination = move
                replacement_text = None
            elif len(move) == 3:
                source, destination, replacement_text = move
            else:
                raise ValueError("config move must contain source, destination, and optional text")
            if source == destination:
                raise ValueError(f"config source and destination are the same: {source}")
            if not source.exists():
                raise ValueError(f"config not found: {source}")
            if destination.exists() or destination in file_writes:
                raise ValueError(f"config already exists: {destination}")
            file_writes[destination] = (
                source.read_text()
                if replacement_text is None
                else replacement_text
            )
            file_modes[destination] = source.stat().st_mode & 0o777
            file_deletes.append(source)

        data = _load_yaml()
        profiles = _validated_profiles(data)
        entries = data.get("profiles", [])
        changed: list[str] = []
        runtime_writes: dict[Path, str] = {}
        runtime_modes: dict[Path, int] = {}
        for index, profile in enumerate(profiles):
            if (
                profile.backend != backend
                or effective_config_name(profile) != old_config_name
            ):
                continue
            updated = replace(profile, config_name=new_config_name)
            entries[index] = _profile_to_entry(
                updated,
                _backend_defaults(data, backend),
            )
            env_path = runtime_env_path(updated.name, backend)
            runtime_writes[env_path] = "\n".join(_render_env_lines(updated))
            runtime_modes[env_path] = 0o600
            changed.append(updated.name)

        data["profiles"] = entries
        _validated_profiles(data)
        internal_paths = {PROFILES_YAML, *runtime_writes}
        overlap = internal_paths.intersection({*file_writes, *file_deletes})
        if overlap:
            listed = ", ".join(str(path) for path in sorted(overlap))
            raise ValueError(f"config transaction targets profile storage path(s): {listed}")
        overlap = set(file_writes).intersection(file_deletes)
        if overlap:
            listed = ", ".join(str(path) for path in sorted(overlap))
            raise ValueError(f"config transaction writes and deletes the same path(s): {listed}")
        if changed:
            file_writes.update(runtime_writes)
            file_writes[PROFILES_YAML] = _dump_yaml(data)
        _replace_files(
            file_writes,
            deletes=tuple(file_deletes),
            modes={
                **file_modes,
                **runtime_modes,
                **({PROFILES_YAML: 0o600} if changed else {}),
            },
        )
        return changed


def runtime_env_path(name: str, backend: str) -> Path:
    if backend not in DEFAULTS:
        raise ValueError(f"invalid backend {backend!r}")
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"invalid profile name {name!r}: must match {_NAME_RE.pattern}")
    return RUNTIME_DIR / backend / f"{name}.env"


def env_value_rejection(value: str) -> str:
    # Compose's dotenv parser cannot consume shlex's quote/control escapes.
    if "'" in value:
        return "single quote (')"
    if '"' in value:
        return 'double quote (")'
    if "\n" in value or "\r" in value:
        return "newline"
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        return "control character"
    return ""


def _env_line(key: str, value: Any) -> str:
    if not _ENV_KEY_RE.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    text = str(value)
    reason = env_value_rejection(text)
    if reason:
        raise ValueError(
            f"env value for {key!r} contains a {reason}, which docker compose's "
            ".env parser cannot read"
        )
    return f"{key}={shlex.quote(text)}"


def _render_env_lines(profile: StoredProfile) -> list[str]:
    lines: list[str] = [
        "# Auto-rendered from profiles.yaml — do not edit directly.",
        f"# Profile: {profile.name} ({profile.backend})",
        "",
    ]
    if profile.backend == "vllm":
        lines += [
            _env_line("CONTAINER_NAME", profile.container_name or profile.name),
            _env_line("VLLM_PORT", profile.port or DEFAULTS["vllm"]["port"]),
            _env_line("GPU_ID", profile.gpu_id),
            _env_line("TENSOR_PARALLEL_SIZE", profile.tensor_parallel_size),
            _env_line("CONFIG_NAME", profile.config_name or profile.name),
            _env_line("MODEL_ID", profile.model_id),
            _env_line("ENABLE_LORA", "true" if profile.enable_lora else "false"),
        ]
        if profile.max_loras is not None:
            lines.append(_env_line("MAX_LORAS", profile.max_loras))
        if profile.max_lora_rank is not None:
            lines.append(_env_line("MAX_LORA_RANK", profile.max_lora_rank))
        if profile.lora_modules:
            lines.append(_env_line("LORA_MODULES", profile.lora_modules))
        if profile.extra_pip_packages:
            lines.append(_env_line("EXTRA_PIP_PACKAGES", profile.extra_pip_packages))
        reserved = _RESERVED_ENV_KEYS_BY_BACKEND["vllm"]
        for k, v in profile.env_vars.items():
            if k in reserved:
                raise ValueError(
                    f"env_vars cannot override reserved profile key {k!r}; "
                    f"set it on the profile itself (gpu_id/port/etc.) instead."
                )
            lines.append(_env_line(k, v))
        if profile.image_tag:
            lines.append(_env_line("VLLM_IMAGE", profile.image_tag))
    else:
        lines += [
            _env_line("CONTAINER_NAME", profile.container_name or profile.name),
            _env_line("LLAMA_PORT", profile.port or DEFAULTS["llamacpp"]["port"]),
            _env_line("GPU_ID", profile.gpu_id),
            _env_line("CONFIG_NAME", profile.config_name or profile.name),
        ]
        if profile.model_file:
            lines.append(_env_line("MODEL_FILE", profile.model_file))
        if profile.hf_repo:
            lines.append(_env_line("HF_REPO", profile.hf_repo))
        if profile.hf_file:
            lines.append(_env_line("HF_FILE", profile.hf_file))
        reserved = _RESERVED_ENV_KEYS_BY_BACKEND["llamacpp"]
        for k, v in profile.env_vars.items():
            if k in reserved:
                raise ValueError(
                    f"env_vars cannot override reserved profile key {k!r}; "
                    f"set it on the profile itself (gpu_id/port/etc.) instead."
                )
            lines.append(_env_line(k, v))
        if profile.image_tag:
            if profile.image_tag.startswith("llamacpp-dev:"):
                lines.append(
                    _env_line(
                        "LLAMACPP_DEV_TAG", profile.image_tag.split(":", 1)[1]
                    )
                )
            else:
                lines.append(_env_line("LLAMACPP_IMAGE", profile.image_tag))
    lines.append("")
    return lines


def _render_env_unlocked(profile: StoredProfile) -> Path:
    _validate_profile(profile)
    out_path = runtime_env_path(profile.name, profile.backend)
    lines = _render_env_lines(profile)
    _atomic_write(out_path, "\n".join(lines), mode=0o600)
    return out_path


def render_env(profile: StoredProfile) -> Path:
    with _storage_lock():
        return _render_env_unlocked(profile)


def render_env_for_profile(name: str, backend: str) -> Path:
    with _storage_lock():
        profile = load_profile(name, backend)
        if profile is None:
            raise ValueError(f"profile {name!r} not found in backend {backend!r}")
        return _render_env_unlocked(profile)


def render_all(backend: str | None = None) -> list[Path]:
    with _storage_lock():
        backends = [backend] if backend else ["vllm", "llamacpp"]
        out: list[Path] = []
        for selected_backend in backends:
            for profile in list_profiles(selected_backend):
                out.append(_render_env_unlocked(profile))
        return out


def _cli() -> int:
    import sys

    argv = sys.argv[1:]
    if len(argv) == 3 and argv[0] == "render":
        backend, name = argv[1], argv[2]
        if backend not in DEFAULTS:
            print(f"Invalid backend: {backend}", file=sys.stderr)
            return 2
        try:
            path = render_env_for_profile(name, backend)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(path)
        return 0
    if len(argv) == 2 and argv[0] == "list":
        backend = argv[1]
        if backend not in DEFAULTS:
            print(f"Invalid backend: {backend}", file=sys.stderr)
            return 2
        for name in list_profile_names(backend):
            print(name)
        return 0
    print(
        "Usage:\n"
        "  python -m tui.common.profile_store render <backend> <name>\n"
        "  python -m tui.common.profile_store list <backend>",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
