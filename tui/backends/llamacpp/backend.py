from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
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
from tui.common.prepare import (
    current_hf_snapshot,
    gguf_shard_names,
    hf_file_error,
    hf_repo_cache_dir,
    resolve_cache_entry,
)
from tui.common.ssl_ctx import open_url, redact_sensitive_text, same_origin

log = logging.getLogger(__name__)


def _resolve_project_root() -> Path:
    env_root = os.environ.get("LLMUX_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    cwd = Path.cwd()
    if (cwd / "profiles.example.yaml").exists() and (cwd / "compose").is_dir():
        return cwd
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _resolve_project_root()
ROOT = PROJECT_ROOT
RUNTIME_DIR = PROJECT_ROOT / ".runtime" / "llamacpp"
CONFIG_DIR = PROJECT_ROOT / "config" / "llamacpp"
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "llamacpp"
COMMON_ENV = PROJECT_ROOT / ".env.common"
LLAMACPP_OFFICIAL_REPO = "ghcr.io/ggml-org/llama.cpp"
LLAMACPP_OFFICIAL_IMAGE = f"{LLAMACPP_OFFICIAL_REPO}:server-cuda"


def validate_name(name: str) -> bool:
    return bool(re.match(r"^[a-z0-9][a-z0-9_-]*$", name))


def _host_expand(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def _get_model_dir() -> Path:
    env_common = ROOT / ".env.common"
    default = ROOT / "models"
    if not env_common.exists():
        return default
    env = _parse_env_file(env_common)
    raw = env.get("MODEL_DIR")
    if not raw:
        return default
    return Path(_host_expand(raw))


def hf_cache_setting() -> str:
    env_common = ROOT / ".env.common"
    if not env_common.exists():
        return ""
    return _parse_env_file(env_common).get("HF_CACHE_PATH", "").strip()


def _get_hf_cache_dir() -> Path:
    raw = hf_cache_setting()
    if not raw:
        raise ValueError(f"HF_CACHE_PATH is not set in {ROOT / '.env.common'}")
    path = Path(_host_expand(raw))
    if not path.is_absolute():
        raise ValueError(f"HF_CACHE_PATH must resolve to an absolute path, got {raw!r}")
    return path


def find_cached_gguf(hf_repo: str, filename: str) -> Path | None:
    if not hf_repo or not filename:
        return None
    repo_dir = hf_repo_cache_dir(_get_hf_cache_dir(), hf_repo)
    shards = gguf_shard_names(filename)
    selected = current_hf_snapshot(repo_dir)
    if selected is None:
        return None
    _, snapshot = selected
    paths = [snapshot / shard for shard in shards]
    if any(resolve_cache_entry(path, repo_dir) is None for path in paths):
        return None
    return paths[0]


def list_cached_gguf() -> list[dict[str, Any]]:
    try:
        cache_root = _get_hf_cache_dir()
        hub = cache_root / "hub"
        if not hub.is_dir():
            return []
        if not hub.resolve(strict=True).is_relative_to(cache_root.resolve(strict=True)):
            raise RuntimeError(
                "Hugging Face hub cache resolves outside the configured cache root"
            )
        out: list[dict[str, Any]] = []
        for repo_dir in sorted(hub.glob("models--*")):
            if not repo_dir.resolve(strict=True).is_relative_to(hub.resolve(strict=True)):
                raise RuntimeError(
                    "Hugging Face repository cache resolves outside the cache root"
                )
            repo = repo_dir.name[len("models--"):].replace("--", "/")
            snapshots = repo_dir / "snapshots"
            if not snapshots.is_dir():
                continue
            seen_blobs: set[tuple[Path, ...]] = set()
            for snapshot in sorted(snapshots.iterdir()):
                if not snapshot.is_dir():
                    continue
                resolved_snapshot = snapshot.resolve(strict=True)
                if not resolved_snapshot.is_relative_to(repo_dir.resolve(strict=True)):
                    raise RuntimeError(
                        "Hugging Face snapshot resolves outside its repository cache root"
                    )
                seen: set[tuple[str, ...]] = set()
                for path in sorted(snapshot.rglob("*.gguf")):
                    relative = path.relative_to(snapshot).as_posix()
                    shard_names = tuple(gguf_shard_names(relative))
                    if shard_names in seen:
                        continue
                    seen.add(shard_names)
                    shard_paths = [snapshot / name for name in shard_names]
                    resolved_shards = [
                        resolve_cache_entry(shard, repo_dir) for shard in shard_paths
                    ]
                    if any(shard is None for shard in resolved_shards):
                        continue
                    blob_group = tuple(
                        shard for shard in resolved_shards if shard is not None
                    )
                    if blob_group in seen_blobs:
                        continue
                    seen_blobs.add(blob_group)
                    size = sum(shard.stat().st_size for shard in blob_group)
                    entry = shard_paths[0]
                    out.append(
                        {
                            "repo": repo,
                            "revision": snapshot.name,
                            "name": entry.name,
                            "path": str(entry),
                            "size_bytes": size,
                            "size_gb": round(size / 1024**3, 1),
                        }
                    )
    except OSError as exc:
        raise RuntimeError(f"llama.cpp cache inventory failed: {exc}") from exc
    out.sort(key=lambda d: d["size_bytes"], reverse=True)
    return out


@dataclass
class Profile:
    name: str
    container_name: str = ""
    port: int = 8080
    gpu_id: str = "0"
    config_name: str = ""
    model_file: str = ""
    hf_repo: str = ""
    hf_file: str = ""
    image_tag: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    downloaded: bool = False
    model_size_gb: float | None = None
    running: bool = False

    @property
    def endpoint(self) -> str:
        return f"http://localhost:{self.port}/v1"

    @property
    def path(self) -> Path:
        return RUNTIME_DIR / f"{self.name}.env"


@dataclass
class Config:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    disabled_params: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return CONFIG_DIR / f"{self.name}.yaml"

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


def _to_profile(stored: profile_store.StoredProfile) -> Profile:
    return Profile(
        name=stored.name,
        container_name=stored.container_name or stored.name,
        port=stored.port,
        gpu_id=stored.gpu_id,
        config_name=stored.config_name,
        model_file=stored.model_file,
        hf_repo=stored.hf_repo,
        hf_file=stored.hf_file,
        image_tag=stored.image_tag,
        env_vars=dict(stored.env_vars),
    )


def _to_stored(profile: Profile) -> profile_store.StoredProfile:
    return profile_store.StoredProfile(
        name=profile.name,
        backend="llamacpp",
        container_name=profile.container_name or profile.name,
        port=int(profile.port),
        gpu_id=profile.gpu_id or "0",
        config_name=profile.config_name,
        model_file=profile.model_file,
        hf_repo=profile.hf_repo,
        hf_file=profile.hf_file,
        image_tag=profile.image_tag,
        env_vars=dict(profile.env_vars),
    )


def list_profile_names() -> list[str]:
    return profile_store.list_profile_names("llamacpp")


def load_profile(name: str) -> Profile:
    stored = profile_store.load_profile(name, "llamacpp")
    if stored is None:
        return Profile(name=name)
    profile = _to_profile(stored)
    profile._stored_snapshot = stored
    return profile


def save_profile(profile: Profile) -> None:
    stored = _to_stored(profile)
    snapshot = getattr(profile, "_stored_snapshot", None)
    if snapshot is None:
        profile_store.create_profile(stored)
    else:
        stored = profile_store.replace_profile(
            snapshot.name,
            stored,
            expected=snapshot,
        )
    profile.container_name = stored.container_name or stored.name
    profile._stored_snapshot = stored


def delete_profile(name: str, delete_config_too: bool = False) -> None:
    if delete_config_too:
        profile_store.delete_profile_with_config(name, "llamacpp", CONFIG_DIR)
        return
    profile_store.delete_profile(name, "llamacpp")


def list_profiles(running: set[str] | None = None) -> list[Profile]:
    model_dir = _get_model_dir()
    running_containers: set[str] = running or set()

    result: list[Profile] = []
    for name in list_profile_names():
        p = load_profile(name)
        from tui.common.prepare import resolve_gguf_file

        filename = resolve_gguf_file(load_config(p.config_name or p.name).get("model-file"), p.model_file, p.hf_file)
        if filename and p.hf_repo:
            model_paths = [model_dir / name for name in gguf_shard_names(filename)]
            if all(path.exists() for path in model_paths):
                p.downloaded = True
                p.model_size_gb = round(
                    sum(path.stat().st_size for path in model_paths) / 1024**3, 1
                )
        if not p.downloaded:
            cached = find_cached_gguf(p.hf_repo, filename)
            if cached is not None:
                relative_parts = Path(gguf_shard_names(filename)[0]).parts
                snapshot = cached
                for _ in relative_parts:
                    snapshot = snapshot.parent
                shard_paths = [snapshot / name for name in gguf_shard_names(filename)]
                p.downloaded = True
                p.model_size_gb = round(
                    sum(path.stat().st_size for path in shard_paths) / 1024**3, 1
                )
        p.running = p.container_name in running_containers
        result.append(p)
    return result


def list_config_names() -> list[str]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(
        path.stem for path in CONFIG_DIR.glob("*.yaml") if path.stem != "example"
    )


def load_config(name: str) -> Config:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return Config(name=name)
    text = path.read_text()
    raw = load_yaml_mapping(text, path)
    params = {str(k): v for k, v in raw.items()}
    disabled = {
        k: v for k, v in parse_disabled_markers(text).items() if k not in params
    }
    return Config(name=name, params=params, disabled_params=disabled)


def save_config(config: Config, *, template: str | None = None) -> None:
    with profile_store.storage_transaction():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        existing = config.path.read_text() if config.path.exists() else None
        text = dump_active_config(existing if existing is not None else template, config.params)
        profile_store._atomic_write(
            config.path,
            text + render_disabled_markers(config.disabled_params),
        )


def delete_config(name: str) -> None:
    with profile_store.storage_transaction():
        path = CONFIG_DIR / f"{name}.yaml"
        if path.exists():
            path.unlink()


def parse_config_param_value(raw: str) -> Any:
    if raw == "":
        return True
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def format_config_param_value(value: Any) -> str:
    if value is True:
        return ""
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(
            value, default_flow_style=True, allow_unicode=True, sort_keys=False
        ).strip()
    return str(value)


async def extract_llama_server_flags(image_ref: str = "") -> set[str]:
    if image_ref:
        image = image_ref
    else:
        env = _parse_env_file(COMMON_ENV) if COMMON_ENV.exists() else {}
        image = env.get("LLAMACPP_IMAGE", "") or LLAMACPP_OFFICIAL_IMAGE
    from tui.common.dev_build import image_reference_credential_error

    error = image_reference_credential_error(image)
    if error:
        raise RuntimeError(error)
    from tui.common.docker import image_identity

    identity = await image_identity(image)
    cache_file = None
    if identity is not None:
        cache_key = hashlib.sha256(f"{image}@{identity}".encode()).hexdigest()[:16]
        cache_file = CONFIG_DIR / f".llamacpp-params-{cache_key}.json"
    if cache_file is not None and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if not isinstance(cached, list) or not cached:
                raise ValueError("expected a non-empty JSON list")
            if any(
                not isinstance(flag, str)
                or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9-]{1,39}", flag)
                for flag in cached
            ):
                raise ValueError("every flag must be a valid string")
            return set(cached)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"invalid llama.cpp flag cache {cache_file}: {exc}") from exc
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "run", "--rm", "--entrypoint", "/app/llama-server",
            image, "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        if proc is not None:
            proc.kill()
            await proc.wait()
        raise RuntimeError(f"timed out inspecting llama.cpp flags from {image}")
    except FileNotFoundError as exc:
        raise RuntimeError("docker is not installed or not available in PATH") from exc
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"could not inspect llama.cpp flags from {image}: docker exited "
            f"with status {proc.returncode}"
        )
    text = stdout.decode("utf-8", errors="replace")
    flags: set[str] = set()
    for match in re.finditer(r"--([a-zA-Z][a-zA-Z0-9-]+)", text):
        flag = match.group(1)
        if 2 <= len(flag) <= 40:
            flags.add(flag)
    if not flags:
        raise RuntimeError(f"could not parse llama.cpp flags from {image}")
    if identity is None:
        identity = await image_identity(image)
        if identity is not None:
            cache_key = hashlib.sha256(f"{image}@{identity}".encode()).hexdigest()[:16]
            cache_file = CONFIG_DIR / f".llamacpp-params-{cache_key}.json"
    if cache_file is not None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        profile_store._atomic_write(cache_file, json.dumps(sorted(flags)))
    return flags


async def stream_logs(container_name: str, *, tail: int = 100):
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "-f", "--tail", str(tail), container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        yield "Error: docker executable not found"
        return
    if proc.stdout is None:
        return
    try:
        while True:
            chunk = await proc.stdout.readline()
            if not chunk:
                break
            yield chunk.decode("utf-8", errors="replace").rstrip()
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


from tui.common.http import chat_completion_bench as chat_completion  # noqa: F401 — re-exported


from tui.common.docker import (  # noqa: F401 — re-exported for backward compat
    GpuInfo,
    format_gpu_bar,
    get_disk_usage,
    get_gpu_info,
    run_command,
)


@dataclass
class DockerImage:
    repository: str
    tag: str
    size: str
    created: str


async def get_docker_images(repo: str = LLAMACPP_OFFICIAL_REPO) -> list[DockerImage]:
    from tui.common.dev_build import image_reference_credential_error

    error = image_reference_credential_error(repo)
    if error:
        raise RuntimeError(error)
    rc, out = await run_command(
        "docker", "images", repo,
        "--format", "{{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}",
        timeout=10,
    )
    if rc != 0:
        raise RuntimeError(
            f"docker images {repo} failed or timed out: {out.strip() or 'no output'}"
        )
    from tui.common.docker import parse_docker_image_rows

    return [
        DockerImage(*row)
        for row in parse_docker_image_rows(out)
        if row[1] != "<none>"
    ]


_HF_TREE_PAGE_CAP = 10


def _parse_link_next(header: str) -> str:
    for part in header.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        for attr in segments[1:]:
            key, _, value = attr.partition("=")
            if key.strip().lower() != "rel" or value.strip().strip('"').lower() != "next":
                continue
            url = segments[0].strip()
            if not (url.startswith("<") and url.endswith(">") and len(url) > 2):
                raise ValueError("malformed HF pagination Link header")
            return url[1:-1]
    return ""


class HfListingUnavailable(RuntimeError):
    pass


async def list_hf_repo_files(repo: str) -> list[dict]:
    import urllib.request
    from urllib.parse import urljoin

    error = profile_store.hf_repo_error(repo)
    if error:
        raise HfListingUnavailable(error)
    loop = asyncio.get_running_loop()

    def _do():
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = f"{endpoint}/api/models/{repo}/tree/main?recursive=true"
        initial_url = url
        token = _parse_env_file(ROOT / ".env.common").get("HF_TOKEN", "").strip()
        try:
            entries: list[dict] = []
            for _ in range(_HF_TREE_PAGE_CAP):
                headers = {"User-Agent": "llmux"}
                if (
                    token
                    and not token.startswith("your_")
                    and same_origin(endpoint, url)
                ):
                    headers["Authorization"] = f"Bearer {token}"
                req = urllib.request.Request(url, headers=headers)
                with open_url(req, timeout=15) as response:
                    page = json.loads(response.read().decode())
                    link = response.headers.get("Link", "") or ""
                if not isinstance(page, list):
                    raise ValueError("HF tree page must be a JSON list")
                if any(
                    not isinstance(entry, dict)
                    or not isinstance(entry.get("type"), str)
                    or not isinstance(entry.get("path"), str)
                    for entry in page
                ):
                    raise ValueError(
                        "HF tree page entries must contain string type and path fields"
                    )
                for entry in page:
                    error = hf_file_error(entry["path"])
                    if error:
                        raise ValueError(f"HF tree entry path is invalid: {error}")
                entries.extend(page)
                next_url = _parse_link_next(link)
                next_url = urljoin(url, next_url) if next_url else ""
                if next_url and not same_origin(initial_url, next_url):
                    raise ValueError("HF pagination refused an off-origin URL")
                url = next_url
                if not url:
                    return entries
            raise RuntimeError(
                f"HF tree listing exceeded the {_HF_TREE_PAGE_CAP}-page limit"
            )
        except Exception as exc:
            safe = redact_sensitive_text(str(exc), (token,))
            raise HfListingUnavailable(f"{repo}: {safe}") from exc

    try:
        return await loop.run_in_executor(None, _do)
    except HfListingUnavailable:
        raise
    except Exception as exc:
        raise HfListingUnavailable(f"{repo}: {exc}") from exc
