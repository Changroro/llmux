"""Container orchestration and compose/runtime helpers."""

from __future__ import annotations

from contextlib import aclosing

import asyncio
import errno
import json
import os
import re
import socket
import urllib.request
from tui.common.ssl_ctx import open_url
from urllib.parse import urlsplit

import yaml

from .backend_common import (
    COMMON_ENV,
    COMPOSE_DIR,
    CONFIG_DIR,
    DEFAULT_VLLM_REPO_URL,
    SCRIPT_DIR,
    VLLM_SRC_DIR,
    Config,
    ContainerStatus,
    Profile,
)
from .backend_inspect import (
    get_dockerhub_release_version,
    get_local_latest_tag,
    resolve_vllm_image_ref,
)
from .backend_process import run_command, run_command_with_options, stream_command
from .backend_storage import (
    _parse_env_file,
    list_profile_names,
    load_config,
    load_profile,
    save_config,
    save_profile,
)
from tui.common import dev_build, prepare, profile_store
from tui.common import docker as common_docker
from tui.common.conflicts import (
    gpu_conflict_messages as _shared_gpu_conflict_messages,
    published_tcp_host_ports,
)
from tui.common.env import validate_common_env


_BUILD_LOCK = asyncio.Lock()
_PACKAGE_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s]+")
_COMPOSE_PROFILE_ENV_KEYS = frozenset(
    {
        "CONTAINER_NAME",
        "VLLM_PORT",
        "GPU_ID",
        "TENSOR_PARALLEL_SIZE",
        "CONFIG_NAME",
        "EXTRA_PIP_PACKAGES",
        "VLLM_IMAGE",
    }
)


def _common_env(*, expand: bool = False) -> dict[str, str]:
    return _parse_env_file(COMMON_ENV, expand=expand)


def get_dev_build_defaults() -> tuple[str, str]:
    common = _common_env()
    repo_url = common.get("VLLM_REPO_URL", "").strip() or VLLM_DEV_SPEC.default_repo_url
    branch = common.get("VLLM_BRANCH", "").strip() or VLLM_DEV_SPEC.default_branch
    return repo_url, branch


def _build_lora_options(profile: Profile) -> str:
    if profile.enable_lora != "true":
        return ""
    parts = ["--enable-lora"]
    if profile.max_loras:
        parts.extend(["--max-loras", profile.max_loras])
    if profile.max_lora_rank:
        parts.extend(["--max-lora-rank", profile.max_lora_rank])
    if profile.lora_modules:
        parts.extend(["--lora-modules", profile.lora_modules.replace(",", " ")])
    return " ".join(parts)


def _extra_pip_packages_error(value: str) -> str:
    for raw_url in _PACKAGE_URL_RE.findall(value):
        parsed = urlsplit(raw_url.rstrip(",;)"))
        if parsed.username is not None or parsed.password is not None:
            return "credential-bearing package URLs are not allowed"
        if parsed.query:
            return "package URLs with query strings are not allowed"
    return ""


def _ensure_common_env(profile: Profile) -> tuple[bool, list[str]]:
    return validate_common_env(
        COMMON_ENV,
        require_lora_base_path=(profile.enable_lora == "true"),
    )


def _validate_core_config_data(data: object) -> None:
    if not isinstance(data, dict):
        raise ValueError(
            f"vLLM config must be a mapping, got {type(data).__name__}"
        )
    model = data.get("model", "")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    gpu_memory = data.get("gpu-memory-utilization", 0.9)
    if isinstance(gpu_memory, bool) or not isinstance(gpu_memory, (int, float)):
        raise ValueError("gpu-memory-utilization must be a number")
    if not 0 < gpu_memory <= 1:
        raise ValueError(
            "gpu-memory-utilization must be greater than 0 and at most 1"
        )


def _validate_core_config_file(path: os.PathLike[str] | str) -> None:
    config_path = os.fspath(path)
    try:
        with open(config_path, encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc
    _validate_core_config_data(data)


def _ensure_profile_config(profile: Profile) -> tuple[bool, list[str]]:
    with profile_store.storage_transaction():
        latest = load_profile(profile.name)
        messages: list[str] = []
        if not latest.config_name:
            latest.config_name = latest.name
            save_profile(latest)
            messages.append(
                f"No config linked for '{latest.name}'. Auto-linked default config '{latest.config_name}'."
            )
        config = load_config(latest.config_name)
        if config.path.exists():
            _validate_core_config_file(config.path)
            model = config.model.strip()
            if model != "your-org/your-model":
                return True, messages
            if latest.model_id:
                config.model = latest.model_id
                save_config(config)
                messages.append(
                    f"Updated config/{latest.config_name}.yaml: model placeholder replaced "
                    f"with profile MODEL_ID '{latest.model_id}'."
                )
                return True, messages
            messages.extend(
                [
                    f"Error: config/{latest.config_name}.yaml does not have a valid model configured yet.",
                    (
                        f"Set the model field in config/{latest.config_name}.yaml or set MODEL_ID "
                        f"in profiles.yaml for '{latest.name}', then start again."
                    ),
                ]
            )
            return False, messages

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if latest.model_id:
            save_config(
                Config(
                    name=latest.config_name,
                    model=latest.model_id,
                    gpu_memory_utilization=0.9,
                )
            )
            messages.append(f"Created default config: config/{latest.config_name}.yaml")
            return True, messages

        profile_store._atomic_write(
            config.path,
            f"# Auto-generated default config for profile: {latest.name}\n"
            "# Set a valid Hugging Face model ID below, then start again.\n"
            "model: your-org/your-model\n"
            "gpu-memory-utilization: 0.9\n",
        )
        messages.extend(
            [
                f"Created config/{latest.config_name}.yaml but MODEL_ID is not set for profile '{latest.name}'.",
                (
                    f"Edit the config model field or set MODEL_ID in profiles.yaml for '{latest.name}', "
                    "then start again."
                ),
            ]
        )
        return False, messages


def _render_profile_snapshot(profile_name: str) -> tuple[Profile, os.PathLike[str]]:
    with profile_store.storage_transaction():
        stored = profile_store.load_profile(profile_name, "vllm")
        if stored is None:
            raise ValueError(f"profile {profile_name!r} not found in backend 'vllm'")
        profile = load_profile(profile_name)
        path = profile_store.render_env(stored)
        return profile, path


def _compose_files(profile: Profile, use_dev: bool) -> list[str]:
    files = ["-f", str(COMPOSE_DIR / "docker-compose.yaml")]
    if use_dev:
        files.extend(["-f", str(COMPOSE_DIR / "docker-compose.dev.yaml")])
    if profile.enable_lora == "true":
        files.extend(["-f", str(COMPOSE_DIR / "docker-compose.lora.yaml")])
    files.extend(["-f", str(COMPOSE_DIR / "docker-compose.overrides.yaml")])
    files.extend(["--project-directory", str(SCRIPT_DIR)])
    return files


def _compose_env(
    profile: Profile,
    *,
    use_dev: bool,
    image_tag: str = "",
    version_tag: str = "",
    vllm_image: str = "",
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(_common_env(expand=True))
    profile_env = _parse_env_file(profile.path)
    env.update(
        {key: profile_env[key] for key in _COMPOSE_PROFILE_ENV_KEYS if key in profile_env}
    )
    env["PROFILE_PATH"] = str(profile.path)
    rendered_profile = Profile(
        name=profile.name,
        enable_lora=profile_env.get("ENABLE_LORA", "false"),
        max_loras=profile_env.get("MAX_LORAS", ""),
        max_lora_rank=profile_env.get("MAX_LORA_RANK", ""),
        lora_modules=profile_env.get("LORA_MODULES", ""),
    )
    env["LORA_OPTIONS"] = _build_lora_options(rendered_profile)
    if use_dev:
        env["VLLM_DEV_TAG"] = image_tag
    else:
        env["VLLM_VERSION"] = version_tag
    env["VLLM_IMAGE"] = vllm_image
    return env


def _env_file_args(profile: Profile) -> list[str]:
    args: list[str] = []
    if COMMON_ENV.exists():
        args.extend(["--env-file", str(COMMON_ENV)])
    if profile.path.exists():
        args.extend(["--env-file", str(profile.path)])
    return args


async def _container_exists(container_name: str) -> bool | None:
    """True/False if the probe succeeded, None if `docker ps` failed/timed out.

    Collapsing a failed probe to False would let a slow daemon read as
    "container gone" and `container_down` claim success while it kept running.
    """
    rc, out = await run_command("docker", "ps", "-a", "--format", "{{.Names}}", timeout=10)
    if rc != 0:
        return None
    return container_name in out.strip().splitlines()


async def _remove_compose_network(project_name: str) -> tuple[int, str]:
    rc, out = await run_command(
        "docker", "network", "rm", f"{project_name}_default", timeout=10
    )
    if rc == 0 or "No such network" in out:
        return 0, ""
    return rc or 1, out.strip() or "docker network rm failed"


async def _gpu_conflict_messages(profile: Profile) -> list[str]:
    """Backend-local thin wrapper over the shared cross-backend helper. Both
    backends call the same `tui.common.conflicts.gpu_conflict_messages` so
    new GPU overlap rules propagate without per-backend drift."""
    return await _shared_gpu_conflict_messages(
        profile_name=profile.name,
        container_name=profile.container_name or profile.name,
        profile_gpu_id=profile.gpu_id,
        backend="vllm",
    )


async def _detect_gpu_arch() -> str:
    """Return PyTorch's TORCH_CUDA_ARCH_LIST format (space-separated dotted caps).

    Thin wrapper around the shared dev_build.detect_local_gpu_caps() helper.
    Kept here as a stable name for the rest of the vllm runtime to call.
    """
    return dev_build.format_arch_torch(await dev_build.detect_local_gpu_caps())


def _force_local_arch_for_deepep(dockerfile: os.PathLike[str] | str) -> tuple[bool, str]:
    """Patch upstream Dockerfile so DeepEP respects TORCH_CUDA_ARCH_LIST.

    Upstream sometimes hard-codes `9.0a 10.0a` for the DeepEP wheel build stage.
    We replace only that step with the already-exported TORCH_CUDA_ARCH_LIST value.
    """
    path = os.fspath(dockerfile)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        return False, f"Error: failed to read upstream Dockerfile: {exc}"

    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if "/tmp/install_python_libraries.sh" not in line:
            continue
        if line.lstrip().startswith("COPY "):
            continue

        # The export is typically the immediately previous non-empty line.
        for prev in range(idx - 1, max(idx - 6, -1), -1):
            if "export TORCH_CUDA_ARCH_LIST=" not in lines[prev]:
                continue

            if (
                "$TORCH_CUDA_ARCH_LIST" in lines[prev]
                or "${TORCH_CUDA_ARCH_LIST}" in lines[prev]
            ):
                return True, "DeepEP already respects local TORCH_CUDA_ARCH_LIST."

            indent = lines[prev].split("export", 1)[0]
            lines[prev] = (
                f'{indent}export TORCH_CUDA_ARCH_LIST="${{TORCH_CUDA_ARCH_LIST}}" && \\\n'
            )
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("".join(lines))
            except OSError as exc:
                return False, f"Error: failed to patch upstream Dockerfile: {exc}"
            return True, "Patched DeepEP stage to use local TORCH_CUDA_ARCH_LIST."

        return False, "Error: could not locate DeepEP arch export line near install step."

    return True, "DeepEP install step not found; skipping DeepEP arch patch."


async def _stream_build_dev_image(
    branch: str,
    *,
    repo_url: str = DEFAULT_VLLM_REPO_URL,
    custom_tag: str = "",
    use_official: bool = False,
):
    if _BUILD_LOCK.locked():
        yield (
            "log",
            "Another dev build is already running. Waiting for it to finish...",
        )
    async with _BUILD_LOCK:
        async with aclosing(_do_build_dev_image(branch, repo_url=repo_url, custom_tag=custom_tag, use_official=use_official)) as _owned_stream:
            async for event in _owned_stream:
                yield event


VLLM_DEV_SPEC = dev_build.DevBuildSpec(
    backend="vllm",
    image_prefix="vllm-dev",
    src_dir=VLLM_SRC_DIR,
    default_repo_url=DEFAULT_VLLM_REPO_URL,
    default_branch="main",
    dockerfile_relpath="docker/Dockerfile",
    target="vllm-openai",
    base_build_args=(("RUN_WHEEL_CHECK", "false"),),
    label_prefix="vllm",
)


async def _do_build_dev_image(
    branch: str,
    *,
    repo_url: str,
    custom_tag: str,
    use_official: bool,
):
    gpu_arch = ""
    gpu_name = ""
    if not use_official:
        gpu_arch = await _detect_gpu_arch()
        if not gpu_arch:
            yield ("log", "Error: Could not detect GPU. Make sure nvidia-smi works.")
            yield ("rc", 1)
            return
        rc, gpu_name = await run_command(
            "nvidia-smi", "--query-gpu=name", "--format=csv,noheader", timeout=10
        )
        gpu_name = gpu_name.splitlines()[0].strip() if rc == 0 and gpu_name.strip() else "unknown"

    extra_log: list[str] = []
    if use_official:
        extra_log.append("Using official vLLM Dockerfile (ALL architectures)")
    else:
        extra_log.append(f"Detected GPU: {gpu_name} (compute: {gpu_arch})")
        extra_log.append("Building with local GPU arch targets.")

    extra_build_args: list[tuple[str, str]] = []
    if not use_official:
        extra_build_args.append(("torch_cuda_arch_list", gpu_arch))

    async def _patch():
        if use_official:
            return True, ""
        ok, msg = _force_local_arch_for_deepep(VLLM_SRC_DIR / "docker/Dockerfile")
        return ok, msg

    async with aclosing(dev_build.stream_build(VLLM_DEV_SPEC, branch, repo_url=repo_url, custom_tag=custom_tag, extra_build_args=tuple(extra_build_args), extra_log_lines=tuple(extra_log), pre_build=_patch, extra_labels=(('vllm.build.type', 'official' if use_official else 'fast'),))) as _owned_stream:
        async for event in _owned_stream:
            yield event


async def _dev_image_matches(image_tag: str, repo_url: str, branch: str) -> bool:
    return await dev_build.image_matches(VLLM_DEV_SPEC, image_tag, repo_url, branch)


async def get_container_statuses() -> list[ContainerStatus]:
    profiles = list_profile_names()
    snapshots = await common_docker.container_snapshots(include_stopped=True)

    statuses = []
    for name in profiles:
        profile = load_profile(name)
        snapshot = snapshots.get(profile.container_name)
        running = snapshot.running if snapshot is not None else False
        health = ""
        status_text = "stopped"
        if snapshot is not None:
            status_text = snapshot.display_status
            if snapshot.health is not common_docker.ContainerHealth.NONE:
                health = snapshot.health.value
        model = ""
        if profile.config_name:
            config = load_config(profile.config_name)
            model = config.model
        statuses.append(
            ContainerStatus(
                profile_name=name,
                container_name=profile.container_name,
                running=running,
                status_text=status_text,
                health=health,
                port=profile.port,
                gpu_id=profile.gpu_id,
                image=profile.image_tag,
                model=model,
                lora=profile.enable_lora == "true",
            )
        )
    return statuses


async def check_port_conflict(profile: Profile) -> str | None:
    """Check whether the profile port is already occupied by a running container
    or local process. Static profile-to-profile overlap (both stopped) is ignored.

    Returns a short human-readable description when a conflict is found.
    """
    rc, out = await run_command(
        "docker", "ps", "--format", "{{.Names}}\t{{.Ports}}", timeout=10
    )
    if rc != 0:
        raise RuntimeError(out.strip() or "docker ps port probe failed")
    container_ports = common_docker.parse_running_container_ports(out)
    for container_name, ports in container_ports.items():
        published_ports = published_tcp_host_ports(ports)
        if container_name == profile.container_name:
            if int(profile.port) in published_ports:
                return None
            continue
        if int(profile.port) in published_ports:
            for name in list_profile_names():
                other = load_profile(name)
                if other.container_name == container_name:
                    return f"profile '{name}'"
            return f"container '{container_name}'"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SO_REUSEADDR: our own readiness probe leaves a TIME_WAIT entry on this
    # port for ~60s, and a plain bind() refuses those. A port a process is
    # actively LISTENING on still fails — that is the conflict we look for.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", int(profile.port)))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            return f"another local process on 127.0.0.1:{profile.port}"
        raise RuntimeError(
            f"local port bind probe failed for 127.0.0.1:{profile.port}: {exc}"
        ) from exc
    finally:
        sock.close()
    return None


async def _models_endpoint_ready(port: str, timeout: int = 3, *, api_key: str = "") -> bool:
    """Return True when /v1/models responds with at least one served model id."""
    loop = asyncio.get_running_loop()

    def _probe() -> bool:
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            )
            with open_url(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            data = payload.get("data", [])
            return any(isinstance(item, dict) and item.get("id") for item in data)
        except Exception:
            return False

    return await loop.run_in_executor(None, _probe)


async def _dir_size_bytes(path: str | None) -> int | None:
    if not path:
        return None
    rc, out = await run_command("du", "-s", "-B1", path, timeout=15)
    if rc != 0:
        raise RuntimeError(out.strip() or f"du exited with status {rc}")
    head = out.strip().split(maxsplit=1)
    if not head or not head[0].isdigit():
        raise RuntimeError(f"du returned malformed output: {out.strip()!r}")
    return int(head[0])


async def _post_start_validation(
    profile: Profile,
    *,
    timeout: float = 45.0,
    poll_interval: float = 3.0,
    max_wait: float = 1800.0,
    hf_cache_path: str | None = None,
):
    loop = asyncio.get_running_loop()
    stall_timeout = max(timeout, poll_interval)
    deadline = loop.time() + stall_timeout
    start = loop.time()

    last_cache_size: int | None = None
    last_log_tail: str | None = None
    last_heartbeat = start
    downloaded_bytes = 0
    from tui.common.http import config_api_key

    try:
        api_key = await config_api_key(load_config(profile.config_name or profile.name).extra_params, profile.container_name, env=profile.env_vars)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("result", False, [f"API authentication configuration failed: {exc}"])
        return

    while True:
        rc, state = await run_command(
            "docker",
            "inspect",
            profile.container_name,
            "--format",
            "{{.State.Status}}\t{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            timeout=10,
        )
        if rc != 0:
            message = state.strip() or "docker inspect failed"
            yield (
                "result",
                False,
                [
                    f"Error: could not inspect container '{profile.container_name}' after startup.",
                    f"  {message}",
                ],
            )
            return
        status = "unknown"
        health = "unknown"
        if state.strip():
            parts = state.strip().split("\t", 1)
            status = parts[0].strip()
            health = parts[1].strip() if len(parts) == 2 else "unknown"

        if status in {"restarting", "exited", "dead"} or health == "unhealthy":
            logs_rc, tail = await run_command(
                "docker",
                "logs",
                "--tail",
                "80",
                profile.container_name,
                timeout=10,
            )
            reason = (
                f"container '{profile.container_name}' exited during startup ({status})"
                if status in {"restarting", "exited", "dead"}
                else f"container '{profile.container_name}' became unhealthy during startup"
            )
            messages = [f"Error: {reason}."]
            if logs_rc != 0:
                messages.append(
                    f"Recent logs unavailable: {tail.strip() or f'docker logs exited with status {logs_rc}'}"
                )
            elif tail.strip():
                messages.append("Recent logs:")
                messages.extend([f"  {line}" for line in tail.strip().splitlines()[-12:]])
            yield ("result", False, messages)
            return

        if status not in {"running"}:
            yield (
                "result",
                False,
                [
                    f"Error: container '{profile.container_name}' is not running after startup ({status})."
                ],
            )
            return

        if await _models_endpoint_ready(profile.port, **({"api_key": api_key} if api_key else {})):
            yield ("result", True, [])
            return

        progressed = False
        downloading = False

        try:
            cache_size = await _dir_size_bytes(hf_cache_path)
        except RuntimeError as exc:
            yield (
                "result",
                False,
                [f"Error: startup progress probe failed: {exc}"],
            )
            return
        if (
            last_cache_size is not None
            and cache_size is not None
            and cache_size > last_cache_size
        ):
            downloading = True
            progressed = True
            downloaded_bytes += cache_size - last_cache_size
        if cache_size is not None:
            last_cache_size = cache_size

        logs_rc, log_tail = await run_command(
            "docker", "logs", "--tail", "15", profile.container_name, timeout=10
        )
        if logs_rc != 0:
            detail = log_tail.strip() or f"docker logs exited with status {logs_rc}"
            yield (
                "result",
                False,
                [f"Error: startup progress probe failed: {detail}"],
            )
            return
        if last_log_tail is not None and log_tail != last_log_tail:
            progressed = True
        last_log_tail = log_tail

        now = loop.time()
        if progressed:
            deadline = now + stall_timeout

        if progressed and now - last_heartbeat >= 12.0:
            if downloading:
                yield (
                    "log",
                    f"  ⏳ 모델 다운로드 중... "
                    f"({int(now - start)}s 경과, HF 캐시 +{downloaded_bytes / 1024**3:.1f} GB)",
                )
            else:
                yield ("log", f"  ⏳ 모델 로딩 중... ({int(now - start)}s 경과)")
            last_heartbeat = now

        if not downloading and now - start >= max_wait:
            yield (
                "result",
                False,
                [
                    f"Error: container '{profile.container_name}' still not ready "
                    f"after {int(max_wait)}s.",
                    "  Model may be stuck — check `docker logs` and retry once ready.",
                ],
            )
            return

        if now >= deadline:
            yield (
                "result",
                False,
                [
                    "Error: container started but /v1/models is not ready within timeout.",
                    "  Model is likely still loading — watch logs and retry once ready.",
                ],
            )
            return

        await asyncio.sleep(poll_interval)


async def stream_container_up(
    profile_name: str,
    use_dev: bool = False,
    use_default_image: bool = False,
    tag: str = "",
    pull: bool = False,
    repo_url: str = "",
    branch: str = "",
):
    """Async generator that streams container startup output line by line.

    `use_default_image=True` mirrors llama.cpp's flag of the same name (and the
    TUI's "Default Image" selection): drop the profile's pinned image_tag for
    this run so compose falls back to its built-in default.

    Priority order (identical to llama.cpp):
      use_default_image > use_dev / explicit tag > profile.image_tag > default.
    """
    # Guard against an unknown name up front — load_profile() below returns a
    # fresh default Profile for a missing name, which would then get persisted
    # as a junk profile. Mirrors llama.cpp's stream_container_up.
    if profile_store.load_profile(profile_name, "vllm") is None:
        yield ("log", f"✗ 프로필 없음: vllm/{profile_name} (profiles.yaml 확인)")
        yield ("rc", 1)
        return

    if tag and not use_dev:
        error = dev_build.image_tag_error(resolve_vllm_image_ref(tag))
        if error:
            yield ("log", f"Error: invalid image reference: {error}")
            yield ("rc", 1)
            return

    if use_dev:
        default_repo_url, _ = get_dev_build_defaults()
        repo_error = dev_build.repo_url_error(repo_url.strip() or default_repo_url)
        if repo_error:
            yield ("log", f"Error: invalid repository URL: {repo_error}")
            yield ("rc", 1)
            return

    profile = load_profile(profile_name)
    image_credential_error = dev_build.image_reference_credential_error(
        profile.image_tag
    )
    if image_credential_error:
        yield ("log", f"Error: invalid profile image reference: {image_credential_error}")
        yield ("rc", 1)
        return

    extra_packages = (profile.extra_pip_packages or "").strip()
    extra_packages_error = _extra_pip_packages_error(extra_packages)
    if extra_packages_error:
        yield ("log", f"Error: EXTRA_PIP_PACKAGES {extra_packages_error}")
        yield ("rc", 1)
        return

    try:
        conflict = await check_port_conflict(profile)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("log", f"Error: port conflict probe failed — {exc}")
        yield ("rc", 1)
        return
    if conflict:
        yield ("log", f"Error: Port {profile.port} is already in use by {conflict}")
        yield ("rc", 1)
        return

    ok, messages = _ensure_common_env(profile)
    for message in messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    try:
        ok, messages = _ensure_profile_config(profile)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("log", f"Error: cannot save {profile_name} — {exc}")
        yield ("rc", 1)
        return
    for message in messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    try:
        profile, _ = _render_profile_snapshot(profile_name)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("log", f"Error: cannot render {profile_name}.env — {exc}")
        yield ("log", "  Fix the value in profiles.yaml (e.g. `llmux profile edit "
                      f"{profile_name} --unset <KEY>`).")
        yield ("rc", 1)
        return

    if use_default_image:
        # Clear the pin *in memory*, before the image branch below reads it.
        # A caller cannot do this by rewriting the .env — load_profile() above
        # re-renders that file from profiles.yaml on every start, restoring the
        # pin. Clearing the field is enough because _compose_env always sets
        # VLLM_IMAGE explicitly, and an empty value makes compose's
        # `${VLLM_IMAGE:-...}` fall through to the default.
        #
        # Must run AFTER _ensure_profile_config: that helper calls
        # save_profile(), so clearing the tag earlier would persist
        # image_tag="" and destroy the user's pin.
        had_pin = bool((profile.image_tag or "").strip())
        profile.image_tag = ""
        if had_pin:
            yield (
                "log",
                "Default Image: ignoring the profile's pinned image_tag for this run.",
            )

    try:
        gpu_conflicts = await _gpu_conflict_messages(profile)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("log", f"Error: GPU conflict probe failed — {exc}")
        yield ("rc", 1)
        return
    for message in gpu_conflicts:
        yield ("log", message)

    if extra_packages:
        yield ("log", "Extra pip packages are configured.")

    compose_files = _compose_files(profile, use_dev)

    if use_dev:
        default_repo_url, default_branch = get_dev_build_defaults()
        resolved_repo_url = repo_url.strip() or default_repo_url
        resolved_branch = branch.strip() or default_branch
        repo_error = dev_build.repo_url_error(resolved_repo_url)
        if repo_error:
            yield ("log", f"Error: invalid repository URL: {repo_error}")
            yield ("rc", 1)
            return
        image_tag = dev_build.sanitize_docker_tag(tag or resolved_branch)
        try:
            exists = await dev_build.image_exists_locally(VLLM_DEV_SPEC, image_tag)
            needs_build = not exists
            if exists:
                needs_build = not await _dev_image_matches(
                    image_tag, resolved_repo_url, resolved_branch
                )
        except RuntimeError as exc:
            yield ("log", f"Error: Docker image probe failed — {exc}")
            yield ("rc", 1)
            return

        if needs_build:
            if exists:
                yield (
                    "log",
                    "Existing dev image metadata does not match the requested repository/branch. Rebuilding...",
                )
            else:
                yield ("log", "Dev image not found. Building first...")
            async with aclosing(_stream_build_dev_image(resolved_branch, repo_url=resolved_repo_url, custom_tag=image_tag if tag else '')) as _owned_stream:
                async for event in _owned_stream:
                    if event[0] == "rc":
                        if int(event[1]) != 0:
                            yield event
                            return
                        continue
                    yield event

        yield ("log", f"Using image: {VLLM_DEV_SPEC.image_prefix}:{image_tag}")
        env = _compose_env(profile, use_dev=True, image_tag=image_tag)
        compose_cmd = [
            "docker",
            "compose",
            *compose_files,
            *_env_file_args(profile),
            "-p",
            profile.name,
            "up",
            "-d",
            # Dev images are locally built — there is no remote to pull
            # them from, so disable compose's default missing-policy
            # fallback.
            "--pull",
            "never",
        ]
    elif not tag and profile.image_tag:
        # Per-profile pinned image. UI overrides (Dev Build / explicit Custom
        # Tag) take precedence and are handled above; absent those, honor the
        # profile's image_tag over the version-tag default. version_tag stays
        # empty so the shared block below skips version verification.
        version_tag = profile.image_tag.rsplit(":", 1)[-1]
        # Mirror llama.cpp: a pinned dev image (vllm-dev:<tag>) that hasn't been
        # built yet would fail compose with an opaque "image not found" error.
        # Pre-check it exists locally and point the user at build-dev. (Non-dev
        # references are pulled/verified by compose itself.)
        if profile.image_tag.startswith(f"{VLLM_DEV_SPEC.image_prefix}:"):
            dev_tag = profile.image_tag.split(":", 1)[1]
            try:
                exists = await dev_build.image_exists_locally(VLLM_DEV_SPEC, dev_tag)
            except RuntimeError as exc:
                yield ("log", f"Error: Docker image probe failed — {exc}")
                yield ("rc", 1)
                return
            if not exists:
                yield ("log", f"Error: Dev image {profile.image_tag} not found locally.")
                yield ("log", "  Build it first:")
                yield ("log", f"  uv run llmux image build-dev --backend vllm --tag {dev_tag}")
                yield ("rc", 1)
                return
        yield ("log", f"Using image: {profile.image_tag}")
        env = _compose_env(profile, use_dev=False, vllm_image=profile.image_tag)
        pull_policy = (
            "never" if profile.image_tag.startswith(f"{VLLM_DEV_SPEC.image_prefix}:")
            else "always" if pull
            else "missing"
        )
        compose_cmd = [
            "docker",
            "compose",
            *compose_files,
            *_env_file_args(profile),
            "-p",
            profile.name,
            "up",
            "-d",
            "--pull",
            pull_policy,
        ]
    else:
        resolved_remote_release = False
        explicit_image_ref = ""
        if tag:
            if resolve_vllm_image_ref(tag) == tag:
                explicit_image_ref = tag
                version_tag = tag.rsplit(":", 1)[-1]
            else:
                version_tag = tag
        else:
            try:
                version_tag = await get_local_latest_tag()
            except RuntimeError as exc:
                yield ("log", f"Error: could not list local images — {exc}")
                yield ("rc", 1)
                return
        if version_tag == "latest":
            # Refuse the `:latest` alias outright. It doesn't describe the image's
            # real contents and clicking "Local Latest" / "Official Release" should
            # always resolve to a specific semver tag before reaching here.
            yield (
                "log",
                "Error: `:latest` is an ambiguous alias and is not allowed. "
                "Pick Local Latest (resolves to your highest local versioned tag) "
                "or Official Release (pulls DockerHub's latest stable by explicit "
                "version), or enter a specific tag in Custom.",
            )
            yield ("rc", 1)
            return
        if version_tag == "none":
            try:
                release_version = await get_dockerhub_release_version()
            except RuntimeError as exc:
                yield ("log", f"Error: could not resolve the release image — {exc}")
                yield ("rc", 1)
                return
            if release_version == "unknown":
                yield (
                    "log",
                    "Error: no local versioned vllm/vllm-openai image and DockerHub "
                    "is unreachable. Pull a specific version first: "
                    "docker pull vllm/vllm-openai:<version>",
                )
                yield ("rc", 1)
                return
            version_tag = release_version
            resolved_remote_release = True

        image_ref = explicit_image_ref or f"vllm/vllm-openai:{version_tag}"
        local_dev = image_ref.startswith(f"{VLLM_DEV_SPEC.image_prefix}:")
        if local_dev:
            dev_tag = image_ref.split(":", 1)[1]
            try:
                exists = await dev_build.image_exists_locally(VLLM_DEV_SPEC, dev_tag)
            except RuntimeError as exc:
                yield ("log", f"Error: Docker image probe failed — {exc}")
                yield ("rc", 1)
                return
            if not exists:
                yield ("log", f"Error: Dev image {image_ref} not found locally. Build it first: llmux image build-dev --backend vllm --tag {dev_tag}")
                yield ("rc", 1)
                return
        yield ("log", f"Using image: {image_ref}")
        if explicit_image_ref:
            env = _compose_env(
                profile, use_dev=False, vllm_image=explicit_image_ref
            )
        else:
            env = _compose_env(profile, use_dev=False, version_tag=version_tag)
        compose_cmd = [
            "docker",
            "compose",
            *compose_files,
            *_env_file_args(profile),
            "-p",
            profile.name,
            "up",
            "-d",
        ]
        if local_dev:
            compose_cmd.extend(["--pull", "never"])
        elif pull or version_tag == "nightly":
            compose_cmd.extend(["--pull", "always"])
        elif tag or resolved_remote_release:
            compose_cmd.extend(["--pull", "missing"])
        else:
            compose_cmd.extend(["--pull", "never"])

    async with aclosing(stream_command(compose_cmd, cwd=SCRIPT_DIR, env=env)) as _owned_stream:
        async for event in _owned_stream:
            if event[0] != "rc":
                yield event
                continue

            rc = int(event[1])
            if rc != 0:
                yield ("rc", rc)
                return

            ok = False
            val_messages: list[str] = []
            async with aclosing(_post_start_validation(profile, hf_cache_path=env.get('HF_CACHE_PATH') or None)) as _owned_stream:
                async for ev in _owned_stream:
                    if ev[0] == "result":
                        ok, val_messages = ev[1], ev[2]
                    else:
                        yield ev
            for message in val_messages:
                yield ("log", message)
            if not ok:
                yield ("rc", 1)
                return

            yield ("log", f"{profile.name} started successfully!")
            if not use_dev:
                async with aclosing(_verify_vllm_version(profile.container_name, version_tag)) as _owned_stream:
                    async for evt in _owned_stream:
                        yield evt
            yield ("rc", 0)
            return


async def _resolve_prepare_image(profile: Profile) -> tuple[str, str]:
    """`(image_ref, error)` — the image `prepare` downloads weights with.

    A pinned image_tag wins; otherwise the newest local versioned image, and
    only when there is none does it fall back to DockerHub's latest stable
    (which prepare is allowed to pull — fetching ahead of time is its job).
    """
    if profile.image_tag:
        error = dev_build.image_reference_credential_error(profile.image_tag)
        if error:
            return "", f"invalid profile image reference: {error}"
        return profile.image_tag, ""
    try:
        version_tag = await get_local_latest_tag()
    except RuntimeError as exc:
        return "", f"could not list local vLLM images — {exc}"
    if version_tag != "none":
        return f"vllm/vllm-openai:{version_tag}", ""
    try:
        release = await get_dockerhub_release_version()
    except RuntimeError as exc:
        return "", f"could not resolve the release image — {exc}"
    if release == "unknown":
        return "", (
            "no local versioned vllm/vllm-openai image and DockerHub is "
            "unreachable — pull one first (docker pull vllm/vllm-openai:<version>)"
        )
    return f"vllm/vllm-openai:{release}", ""


async def stream_container_prepare(profile_name: str, *, max_workers: int | None = None):
    """Render runtime files, make sure the image is local, download the model.

    Everything `up` does before `docker compose up`, plus the weight download —
    and nothing after it. No server is started and no GPU is touched, so the
    next `up` only has to load what is already on disk.
    """
    if profile_store.load_profile(profile_name, "vllm") is None:
        yield ("log", f"✗ profile not found: vllm/{profile_name} (check profiles.yaml)")
        yield ("rc", 1)
        return

    profile = load_profile(profile_name)

    ok, messages = _ensure_common_env(profile)
    for message in messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    try:
        ok, messages = _ensure_profile_config(profile)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("log", f"Error: cannot save {profile_name} — {exc}")
        yield ("rc", 1)
        return
    for message in messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    try:
        profile, path = _render_profile_snapshot(profile_name)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("log", f"Error: cannot render {profile_name}.env — {exc}")
        yield ("rc", 1)
        return
    yield ("log", f"▸ Rendered runtime env: {path}")

    config = load_config(profile.config_name)
    model = config.model.strip()
    if not model:
        yield ("log", f"Error: config/{profile.config_name}.yaml has no model set.")
        yield ("rc", 1)
        return

    image_ref, error = await _resolve_prepare_image(profile)
    if error:
        yield ("log", f"Error: {error}")
        yield ("rc", 1)
        return

    try:
        present = await prepare.image_present(image_ref)
    except RuntimeError as exc:
        yield ("log", f"Error: Docker image probe failed — {exc}")
        yield ("rc", 1)
        return
    if not present:
        if image_ref.startswith(f"{VLLM_DEV_SPEC.image_prefix}:"):
            dev_tag = image_ref.split(":", 1)[1]
            yield ("log", f"Error: Dev image {image_ref} not found locally.")
            yield ("log", "  Build it first:")
            yield ("log", f"  uv run llmux image build-dev --backend vllm --tag {dev_tag}")
            yield ("rc", 1)
            return
        async with aclosing(prepare.stream_pull(image_ref)) as _owned_stream:
            async for event in _owned_stream:
                if event[0] == "rc":
                    if int(event[1]) != 0:
                        yield ("log", f"Error: could not pull {image_ref}")
                        yield ("rc", int(event[1]))
                        return
                else:
                    yield event
    yield ("log", f"▸ Image: {image_ref}")

    if model.startswith("/") or model.startswith("~"):
        yield ("log", f"▸ Model is a container path ({model}) — nothing to download.")
        yield ("log", f"✓ {profile_name} prepared. Start it with: llmux up {profile_name}")
        yield ("rc", 0)
        return

    cache_path = prepare.hf_cache_path()
    if not cache_path:
        yield ("log", "Error: HF_CACHE_PATH is not set in .env.common — "
                      "prepare has nowhere to download to.")
        yield ("rc", 1)
        return

    yield ("log", f"▸ Downloading {model} into {cache_path}")
    rc = -1
    async with aclosing(prepare.stream_vllm_download(image_ref=image_ref, model_id=model, revision=config.extra_params.get('revision') or '', cache_path=cache_path, token=prepare.hf_token(), container_name=prepare.prepare_container_name(profile.container_name), max_workers=max_workers)) as _owned_stream:
        async for event in _owned_stream:
            if event[0] == "rc":
                rc = int(event[1])
            else:
                yield event
    if rc != 0:
        yield ("log", f"✗ Download failed (rc={rc}).")
        yield ("rc", rc if rc != 0 else 1)
        return

    yield ("log", f"✓ {profile_name} prepared. Start it with: llmux up {profile_name}")
    yield ("rc", 0)


async def _verify_vllm_version(container_name: str, expected_tag: str):
    """Compare the tag we told docker to run against the vllm version actually
    running inside the container. Warn (but don't fail) on mismatch — tag names
    can lie about contents if someone ran `docker tag` by hand or pulled the
    same `latest` alias at different times.
    """
    from .backend_inspect import _parse_stable_version_tag

    expected = _parse_stable_version_tag(expected_tag)
    if expected is None:
        # `nightly` reaches here from the Start screen and has no version to
        # compare against.
        return

    rc, out = await run_command(
        "docker",
        "exec",
        container_name,
        "python3",
        "-c",
        "import vllm; print(vllm.__version__)",
        timeout=15,
    )
    if rc != 0 or not out.strip():
        yield (
            "log",
            "Warning: could not verify vLLM version inside the container "
            f"({out.strip() or 'docker exec failed'}).",
        )
        return

    actual_str = out.strip().splitlines()[-1].strip()
    actual = _parse_stable_version_tag("v" + actual_str if not actual_str.startswith("v") else actual_str)
    if actual is None:
        yield (
            "log",
            f"Warning: could not parse vLLM version reported by container: {actual_str}",
        )
        return

    if actual != expected:
        yield (
            "log",
            f"⚠ Warning: image tag says v{expected[0]}.{expected[1]}.{expected[2]}, "
            f"but the container reports vllm {actual_str}. The tag may have been "
            f"retagged or pulled at a different time — consider re-pulling the "
            f"specific version you want.",
        )


async def container_down(profile_name: str) -> tuple[int, str]:
    try:
        stored = profile_store.load_profile(profile_name, "vllm")
    except Exception as exc:
        return 1, f"could not load vllm/{profile_name}: {exc}"
    if stored is None:
        return 1, f"profile not found: vllm/{profile_name}"
    try:
        profile = load_profile(profile_name)
        exists = await _container_exists(profile.container_name)
    except Exception as exc:
        return 1, f"could not determine container state for {profile_name}: {exc}"
    if exists is None:
        return 1, (
            f"could not determine container state for {profile_name} "
            "(docker ps failed or timed out)"
        )
    if not exists:
        return 0, f"{profile_name} is not running"

    compose_err = ""
    try:
        profile_store.render_env_for_profile(profile_name, "vllm")
    except Exception as exc:
        compose_err = f"could not render runtime env: {exc}"

    if not compose_err:
        try:
            image_rc, image = await run_command(
                "docker",
                "inspect",
                profile.container_name,
                "--format={{.Config.Image}}",
                timeout=20,
            )
        except Exception as exc:
            image_rc, image = 1, ""
            compose_err = f"image probe failed: {exc}"
        if image_rc != 0 and not compose_err:
            detail = image.strip() or f"docker inspect exited with status {image_rc}"
            compose_err = f"image probe failed: {detail}"

    if not compose_err:
        use_dev = image.strip().startswith(f"{VLLM_DEV_SPEC.image_prefix}:")
        try:
            env = _compose_env(
                profile,
                use_dev=use_dev,
                image_tag=(
                    image.strip().split(":", 1)[1]
                    if use_dev and ":" in image.strip()
                    else ""
                ),
            )
            compose_cmd = [
                "docker",
                "compose",
                *_compose_files(profile, use_dev),
                *_env_file_args(profile),
                "-p",
                profile.name,
                "down",
            ]
            rc, out = await run_command_with_options(
                *compose_cmd, cwd=SCRIPT_DIR, env=env, timeout=60
            )
            if rc == 0:
                return 0, f"{profile_name} stopped successfully!"
            compose_err = next(
                (line for line in reversed(out.strip().splitlines()) if line.strip()),
                f"rc={rc}",
            )
        except Exception as exc:
            compose_err = f"compose preparation failed: {exc}"

    stop_rc, stop_out = await run_command("docker", "stop", profile.container_name, timeout=30)
    if stop_rc != 0:
        detail = stop_out.strip() or f"docker stop exited with status {stop_rc}"
        return stop_rc if stop_rc > 0 else 1, (
            f"{profile_name} compose down unavailable ({compose_err}); "
            f"manual stop failed: {detail}"
        )
    rm_rc, rm_out = await run_command("docker", "rm", profile.container_name, timeout=30)
    if rm_rc != 0:
        detail = rm_out.strip() or f"docker rm exited with status {rm_rc}"
        return rm_rc if rm_rc > 0 else 1, (
            f"{profile_name} stopped, but manual remove failed: {detail} "
            f"(compose down unavailable: {compose_err})"
        )
    network_rc, network_out = await _remove_compose_network(profile.name)
    if network_rc != 0:
        return network_rc, (
            f"{profile_name} container stopped, but network cleanup failed: "
            f"{network_out}"
        )
    return 0, (
        f"{profile_name} stopped via docker stop/rm "
        f"(compose down failed: {compose_err})"
    )


async def stream_container_logs(container_name: str, *, tail: int = 100):
    """Async generator that yields log lines.

    `tail` controls how many recent lines `docker logs` prints before it
    starts following live output.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "logs",
            "-f",
            "--tail",
            str(tail),
            container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        yield "Error: docker executable not found"
        return
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield line.decode(errors="replace").rstrip("\n")
    finally:
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            await proc.wait()
        except (asyncio.CancelledError, ProcessLookupError, OSError):
            pass
