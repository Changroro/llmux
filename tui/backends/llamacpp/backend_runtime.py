from __future__ import annotations

import asyncio
import errno
import json
import os
import socket
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tui.common import dev_build, prepare, profile_store
from tui.common import docker as common_docker
from tui.common.conflicts import (
    gpu_conflict_messages as _shared_gpu_conflict_messages,
    published_tcp_host_ports,
)
from tui.common.docker import (
    run_command as _docker_run,
)
from tui.common.env import validate_common_env

from .backend import (
    COMMON_ENV,
    LLAMACPP_OFFICIAL_IMAGE,
    CONFIG_DIR,
    PROJECT_ROOT,
    RUNTIME_DIR,
    SCRIPTS_DIR,
    Config,
    Profile,
    _parse_env_file,
    list_profile_names,
    load_config,
    load_profile,
    save_config,
)

COMPOSE_DIR = PROJECT_ROOT / "compose" / "llamacpp"
BASE_COMPOSE = COMPOSE_DIR / "docker-compose.yaml"
DEV_COMPOSE = COMPOSE_DIR / "docker-compose.dev.yaml"

LLAMACPP_SRC_DIR = PROJECT_ROOT / ".llamacpp-src"
DEFAULT_LLAMACPP_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"

LLAMACPP_DEV_SPEC = dev_build.DevBuildSpec(
    backend="llamacpp",
    image_prefix="llamacpp-dev",
    src_dir=LLAMACPP_SRC_DIR,
    default_repo_url=DEFAULT_LLAMACPP_REPO_URL,
    default_branch="master",
    dockerfile_relpath=".devops/cuda.Dockerfile",
    target="server",
    label_prefix="llamacpp",
)


_BUILD_LOCK = asyncio.Lock()
_COMPOSE_PROFILE_ENV_KEYS = frozenset(
    {
        "CONTAINER_NAME",
        "LLAMA_PORT",
        "GPU_ID",
        "CONFIG_NAME",
        "LLAMACPP_IMAGE",
        "LLAMACPP_DEV_TAG",
    }
)


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


def get_dev_build_defaults() -> tuple[str, str]:
    env = _parse_env_file(COMMON_ENV) if COMMON_ENV.exists() else {}
    repo_url = env.get("LLAMACPP_REPO_URL", "").strip() or DEFAULT_LLAMACPP_REPO_URL
    branch = env.get("LLAMACPP_BRANCH", "").strip() or LLAMACPP_DEV_SPEC.default_branch
    return repo_url, branch


async def _stream_build_dev_image(
    branch: str,
    *,
    repo_url: str = "",
    custom_tag: str = "",
    cuda_arch: str = "",
    use_multi_arch: bool = False,
):
    if _BUILD_LOCK.locked():
        yield ("log", "Another llamacpp dev build is already running. Waiting...")
    async with _BUILD_LOCK:
        async for event in _do_build_dev_image(
            branch,
            repo_url=repo_url,
            custom_tag=custom_tag,
            cuda_arch=cuda_arch,
            use_multi_arch=use_multi_arch,
        ):
            yield event


async def _do_build_dev_image(
    branch: str,
    *,
    repo_url: str,
    custom_tag: str,
    cuda_arch: str,
    use_multi_arch: bool = False,
):
    resolved_arch = cuda_arch.strip()
    log_lines: list[str] = []

    if not resolved_arch and not use_multi_arch:
        caps = await dev_build.detect_local_gpu_caps()
        if not caps:
            # A multi-arch build can take hours; only --multi-arch opts into it.
            yield ("log", "Error: could not detect GPU (nvidia-smi unavailable). "
                          "Pass --cuda-arch <arch> or --multi-arch to build anyway.")
            yield ("rc", 1)
            return
        resolved_arch = dev_build.format_arch_cmake(caps)
        log_lines.append(f"Detected GPUs (compute_cap): {', '.join(caps)}")
        log_lines.append(f"Building with CUDA_DOCKER_ARCH={resolved_arch} (fast)")

    extra_build_args: list[tuple[str, str]] = []
    if resolved_arch:
        extra_build_args.append(("CUDA_DOCKER_ARCH", resolved_arch))
    else:
        log_lines.append("Building with CUDA_DOCKER_ARCH=default (multi-arch, slower)")

    async for event in dev_build.stream_build(
        LLAMACPP_DEV_SPEC,
        branch,
        repo_url=repo_url,
        custom_tag=custom_tag,
        extra_build_args=tuple(extra_build_args),
        extra_log_lines=tuple(log_lines),
    ):
        yield event


async def _dev_image_matches(image_tag: str, repo_url: str, branch: str) -> bool:
    return await dev_build.image_matches(LLAMACPP_DEV_SPEC, image_tag, repo_url, branch)


async def _run(*args: str, env: dict[str, str] | None = None, timeout: float = 60) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    except FileNotFoundError:
        return -1, f"Executable not found: {args[0] if args else '<empty>'}"
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        return -1, "Command timed out"
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
    rc = proc.returncode if proc.returncode is not None else -1
    return rc, (stdout or b"").decode(errors="replace")


async def _stream(args: list[str], *, env: dict[str, str] | None = None):
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    except FileNotFoundError:
        yield ("log", f"✗ Executable not found: {args[0] if args else '<empty>'}")
        yield ("rc", -1)
        return
    if proc.stdout is None:
        yield ("rc", 1)
        return
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield ("log", line.decode(errors="replace").rstrip("\n"))
        await proc.wait()
        rc = proc.returncode if proc.returncode is not None else -1
        yield ("rc", rc)
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()


def _override_path(profile_name: str) -> Path:
    return RUNTIME_DIR / f"override-{profile_name}.yaml"


def _compose_files(profile: Profile) -> list[str]:
    override = _override_path(profile.name)
    files = ["-f", str(BASE_COMPOSE)]
    if profile.image_tag.startswith(f"{LLAMACPP_DEV_SPEC.image_prefix}:"):
        files.extend(["-f", str(DEV_COMPOSE)])
    files.extend(["-f", str(override)])
    return files


def _compose_env(profile: Profile) -> dict[str, str]:
    env = os.environ.copy()
    if COMMON_ENV.exists():
        env.update(_parse_env_file(COMMON_ENV, expand=True))
    if profile.path.exists():
        profile_env = _parse_env_file(profile.path)
        env.update(
            {
                key: profile_env[key]
                for key in _COMPOSE_PROFILE_ENV_KEYS
                if key in profile_env
            }
        )
    env["PROFILE_PATH"] = str(profile.path)
    return env


def _resolve_runtime_image(
    profile: Profile, *, use_default_image: bool = False, tag: str = ""
) -> str:
    if use_default_image:
        if COMMON_ENV.exists():
            common_image = _parse_env_file(COMMON_ENV).get("LLAMACPP_IMAGE", "").strip()
            if common_image:
                return common_image
        return LLAMACPP_OFFICIAL_IMAGE
    if tag.strip():
        return tag.strip()
    if profile.image_tag.strip():
        return profile.image_tag.strip()
    if COMMON_ENV.exists():
        common_image = _parse_env_file(COMMON_ENV).get("LLAMACPP_IMAGE", "").strip()
        if common_image:
            return common_image
    return LLAMACPP_OFFICIAL_IMAGE


def _compose_base_args(profile: Profile) -> list[str]:
    args = [
        "docker",
        "compose",
        "-p",
        profile.name,
        *_compose_files(profile),
        "--project-directory",
        str(PROJECT_ROOT),
    ]
    if COMMON_ENV.exists():
        args.extend(["--env-file", str(COMMON_ENV)])
    if profile.path.exists():
        args.extend(["--env-file", str(profile.path)])
    return args


async def _render_override(profile_name: str) -> tuple[int, str]:
    return await _run(
        sys.executable,
        str(SCRIPTS_DIR / "render-override.py"),
        profile_name,
    )


async def _gpu_conflict_messages(profile: Profile) -> list[str]:
    return await _shared_gpu_conflict_messages(
        profile_name=profile.name,
        container_name=profile.container_name or profile.name,
        profile_gpu_id=profile.gpu_id,
        backend="llamacpp",
    )


async def check_port_conflict(profile: Profile) -> str | None:
    port = str(profile.port)
    rc, out = await _docker_run(
        "docker", "ps", "--format", "{{.Names}}\t{{.Ports}}", timeout=10
    )
    if rc != 0:
        detail = out.strip() or f"exit status {rc}"
        raise RuntimeError(f"docker ps port probe failed: {detail}")
    container_ports = common_docker.parse_running_container_ports(out)
    for container_name, ports in container_ports.items():
        published_ports = published_tcp_host_ports(ports)
        if container_name == profile.container_name:
            if int(port) in published_ports:
                return None
            continue
        if int(port) in published_ports:
            for name in list_profile_names():
                other = load_profile(name)
                if other.container_name == container_name:
                    return f"profile '{name}'"
            return f"container '{container_name}'"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SO_REUSEADDR ignores TIME_WAIT but still detects active listeners.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", int(port)))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            return f"another local process on 127.0.0.1:{port}"
        raise RuntimeError(
            f"local port bind probe failed for 127.0.0.1:{port}: {exc}"
        ) from exc
    finally:
        sock.close()
    return None


async def get_container_statuses() -> list[ContainerStatus]:
    names = list_profile_names()
    snapshots = await common_docker.container_snapshots(include_stopped=True)

    statuses: list[ContainerStatus] = []
    for name in names:
        profile = load_profile(name)
        snapshot = snapshots.get(profile.container_name)
        running = snapshot.running if snapshot is not None else False
        health = ""
        status_text = "stopped"
        if snapshot is not None:
            status_text = snapshot.display_status
            if snapshot.health is not common_docker.ContainerHealth.NONE:
                health = snapshot.health.value
        model_label = ""
        if profile.config_name:
            cfg = load_config(profile.config_name)
            alias = cfg.get("alias")
            if isinstance(alias, str) and alias.strip():
                model_label = alias.strip()
        if not model_label:
            model_label = profile.hf_repo or profile.model_file or ""

        statuses.append(
            ContainerStatus(
                profile_name=name,
                container_name=profile.container_name,
                running=running,
                status_text=status_text,
                health=health,
                port=str(profile.port),
                gpu_id=profile.gpu_id,
                image=profile.image_tag,
                model=model_label,
                lora=False,
            )
        )
    return statuses


async def _models_endpoint_ready(port: str | int, timeout: int = 3, *, api_key: str = "") -> bool:
    loop = asyncio.get_running_loop()

    def _probe() -> bool:
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            data = payload.get("data", [])
            return any(isinstance(item, dict) and item.get("id") for item in data)
        except Exception:
            return False

    return await loop.run_in_executor(None, _probe)


async def _dir_size_bytes(path: str | None) -> int | None:
    if not path:
        return None
    rc, out = await _docker_run("du", "-s", "-B1", path, timeout=15)
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
        api_key = await config_api_key(load_config(profile.config_name or profile.name).params, profile.container_name, backend="llamacpp", env=profile.env_vars)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("result", False, [f"API authentication configuration failed: {exc}"])
        return

    while True:
        rc, state = await _docker_run(
            "docker",
            "inspect",
            profile.container_name,
            "--format",
            "{{.State.Status}}\t{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            timeout=10,
        )
        if rc != 0:
            yield (
                "result",
                False,
                [
                    f"Error: could not inspect container '{profile.container_name}' after startup.",
                    f"  {state.strip() or 'docker inspect failed'}",
                ],
            )
            return
        status, _, health = state.strip().partition("\t")
        status = status or "unknown"

        if status in {"restarting", "exited", "dead"} or health == "unhealthy":
            logs_rc, tail = await _docker_run(
                "docker", "logs", "--tail", "80", profile.container_name, timeout=10
            )
            reason = (
                f"container '{profile.container_name}' exited during startup ({status})"
                if status in {"restarting", "exited", "dead"}
                else f"container '{profile.container_name}' became unhealthy during startup"
            )
            msgs = [f"Error: {reason}."]
            if logs_rc != 0:
                detail = tail.strip() or f"docker logs exited with status {logs_rc}"
                msgs.append(f"Recent logs unavailable: {detail}")
            elif tail.strip():
                msgs.append("Recent logs:")
                msgs.extend([f"  {line}" for line in tail.strip().splitlines()[-12:]])
            yield ("result", False, msgs)
            return

        if status != "running":
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

        logs_rc, log_tail = await _docker_run(
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


def _ensure_profile_config(
    stored: profile_store.StoredProfile, profile: Profile
) -> tuple[bool, list[str]]:
    with profile_store.storage_transaction():
        latest_stored = profile_store.load_profile(stored.name, "llamacpp")
        if latest_stored is None:
            raise ValueError(
                f"profile {stored.name!r} not found in backend 'llamacpp'"
            )
        latest_profile = load_profile(stored.name)
        messages: list[str] = []
        if not latest_stored.config_name:
            latest_stored.config_name = latest_profile.name
            latest_profile.config_name = latest_profile.name
            try:
                profile_store.save_profile(latest_stored)
            except Exception as exc:  # noqa: BLE001
                messages.append(f"✗ config 자동 링크 저장 실패: {exc}")
                return False, messages
            messages.append(f"▸ config 미링크 — '{latest_profile.name}' 로 자동 링크")

        config_path = CONFIG_DIR / f"{latest_stored.config_name}.yaml"
        if not config_path.exists():
            try:
                save_config(Config(
                    name=latest_stored.config_name,
                    params={
                        "alias": latest_stored.config_name,
                        "ctx-size": 8192,
                        "n-gpu-layers": 99,
                    },
                ))
            except Exception as exc:  # noqa: BLE001
                messages.append(f"✗ 기본 config 생성 실패: {exc}")
                return False, messages
            messages.append(
                f"▸ 기본 config 생성: {config_path} (ctx-size/n-gpu-layers 기본값 — 확인 후 조정하세요)"
            )
            return True, messages

        existing = load_config(latest_stored.config_name)
        if not existing.params:
            messages.append(
                f"✗ config '{latest_stored.config_name}' 가 비어 있습니다 ({config_path})."
            )
            messages.append("  ctx-size / n-gpu-layers 등을 채운 뒤 다시 시작하세요.")
            return False, messages
        missing = [
            key for key in ("ctx-size", "n-gpu-layers")
            if key not in existing.params
        ]
        if missing:
            messages.append(
                f"⚠ config '{latest_stored.config_name}' 에 {', '.join(missing)} 없음 — llama-server 기본값이 적용됩니다."
            )
        return True, messages


def _render_profile_snapshot(profile_name: str) -> tuple[Profile, Path]:
    with profile_store.storage_transaction():
        stored = profile_store.load_profile(profile_name, "llamacpp")
        if stored is None:
            raise ValueError(
                f"profile {profile_name!r} not found in backend 'llamacpp'"
            )
        profile = load_profile(profile_name)
        path = profile_store.render_env(stored)
        return profile, path


async def stream_container_up(
    profile_name: str,
    *,
    use_dev: bool = False,
    use_default_image: bool = False,
    tag: str = "",
    pull: bool = False,
    repo_url: str = "",
    branch: str = "",
):
    stored = profile_store.load_profile(profile_name, "llamacpp")
    if stored is None:
        yield ("log", f"✗ 프로필 없음: llamacpp/{profile_name} (profiles.yaml 확인)")
        yield ("rc", 1)
        return

    if tag and not use_dev:
        error = dev_build.image_tag_error(tag)
        if error:
            yield ("log", f"✗ 잘못된 이미지 reference: {error}")
            yield ("rc", 1)
            return


    if use_dev:
        default_repo, _ = get_dev_build_defaults()
        repo_error = dev_build.repo_url_error((repo_url or default_repo).strip())
        if repo_error:
            yield ("log", f"Error: invalid repository URL: {repo_error}")
            yield ("rc", 1)
            return

    ok, env_messages = validate_common_env(COMMON_ENV)
    for message in env_messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    profile = load_profile(profile_name)
    image_credential_error = dev_build.image_reference_credential_error(
        profile.image_tag
    )
    if image_credential_error:
        yield ("log", f"Error: invalid runtime image reference: {image_credential_error}")
        yield ("rc", 1)
        return

    try:
        conflict = await check_port_conflict(profile)
    except RuntimeError as exc:
        yield ("log", f"✗ 포트 사전 확인 실패: {exc}")
        yield ("rc", 1)
        return
    if conflict:
        yield ("log", f"✗ 포트 {profile.port} 사용 중 — {conflict}")
        yield ("rc", 1)
        return
    try:
        gpu_messages = await _gpu_conflict_messages(profile)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("log", f"✗ GPU 충돌 사전 확인 실패: {exc}")
        yield ("rc", 1)
        return
    for message in gpu_messages:
        yield ("log", message)

    # Config linking must precede transient image_tag assignment.
    try:
        ok, cfg_messages = _ensure_profile_config(stored, profile)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("log", f"✗ config 저장 실패: {exc}")
        yield ("rc", 1)
        return
    for message in cfg_messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    try:
        profile, _ = _render_profile_snapshot(profile_name)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("log", f"✗ profile env render 실패: {exc}")
        yield ("rc", 1)
        return

    resolved_image_tag = _resolve_runtime_image(
        profile, use_default_image=use_default_image, tag=tag
    )
    if use_dev:
        default_repo, default_branch = get_dev_build_defaults()
        resolved_repo = (repo_url or default_repo).strip()
        resolved_branch = (branch or default_branch).strip()
        repo_error = dev_build.repo_url_error(resolved_repo)
        if repo_error:
            yield ("log", f"Error: invalid repository URL: {repo_error}")
            yield ("rc", 1)
            return
        dev_tag = dev_build.sanitize_docker_tag((tag or resolved_branch).strip())
        resolved_image_tag = f"{LLAMACPP_DEV_SPEC.image_prefix}:{dev_tag}"

        try:
            exists = await dev_build.image_exists_locally(LLAMACPP_DEV_SPEC, dev_tag)
        except RuntimeError as exc:
            yield ("log", f"✗ Docker 이미지 확인 실패: {exc}")
            yield ("rc", 1)
            return

        try:
            matches = exists and (
                await dev_build.image_matches(
                    LLAMACPP_DEV_SPEC, dev_tag, resolved_repo, resolved_branch
                )
            )
        except RuntimeError as exc:
            yield ("log", f"✗ Docker 이미지 metadata 확인 실패: {exc}")
            yield ("rc", 1)
            return
        if not exists:
            yield ("log", f"▸ Dev image {resolved_image_tag} not found — building from {resolved_repo}@{resolved_branch}")
        elif not matches:
            yield ("log", f"▸ Dev image {resolved_image_tag} exists but was built from a different repo/branch — rebuilding from {resolved_repo}@{resolved_branch}")
        if not matches:
            async for event in _stream_build_dev_image(
                resolved_branch, repo_url=resolved_repo, custom_tag=tag
            ):
                if event[0] == "rc":
                    if int(event[1]) != 0:
                        yield event
                        return
                    continue
                yield event
    elif tag:
        resolved_image_tag = tag.strip()

    image_credential_error = dev_build.image_reference_credential_error(
        resolved_image_tag
    )
    if image_credential_error:
        yield ("log", f"Error: invalid runtime image reference: {image_credential_error}")
        yield ("rc", 1)
        return

    profile.image_tag = resolved_image_tag

    if (
        not use_dev
        and resolved_image_tag.startswith(f"{LLAMACPP_DEV_SPEC.image_prefix}:")
    ):
        dev_tag = resolved_image_tag.split(":", 1)[1]
        try:
            exists = await dev_build.image_exists_locally(LLAMACPP_DEV_SPEC, dev_tag)
        except RuntimeError as exc:
            yield ("log", f"✗ Docker 이미지 확인 실패: {exc}")
            yield ("rc", 1)
            return
        if not exists:
            yield ("log", f"✗ Dev image {resolved_image_tag} not found locally.")
            yield ("log", "  Build it first:")
            yield ("log", f"  uv run llmux image build-dev --backend llamacpp --tag {dev_tag}")
            yield ("rc", 1)
            return
        yield ("log", f"▸ Dev image: {resolved_image_tag}")
    elif not use_dev and resolved_image_tag:
        yield ("log", f"▸ Image: {resolved_image_tag}")

    yield ("log", "▸ command 렌더링")
    rc, out = await _render_override(profile_name)
    if rc != 0:
        yield ("log", out.strip() or "✗ override 렌더 실패")
        yield ("rc", rc)
        return
    for line in out.splitlines():
        if line.startswith("warning:"):
            yield ("log", f"⚠ {line[len('warning:'):].strip()}")

    yield ("log", f"▸ '{profile_name}' 프로필로 기동")
    env = _compose_env(profile)
    if resolved_image_tag.startswith(f"{LLAMACPP_DEV_SPEC.image_prefix}:"):
        env["LLAMACPP_DEV_TAG"] = resolved_image_tag.split(":", 1)[1]
        env["LLAMACPP_IMAGE"] = ""
    else:
        env["LLAMACPP_IMAGE"] = resolved_image_tag
        env["LLAMACPP_DEV_TAG"] = ""
    pull_policy = (
        "never" if resolved_image_tag.startswith(f"{LLAMACPP_DEV_SPEC.image_prefix}:")
        else "always" if pull
        else "missing"
    )
    cmd = [*_compose_base_args(profile), "up", "-d", "--pull", pull_policy]

    async for event in _stream(cmd, env=env):
        if event[0] != "rc":
            yield event
            continue
        rc = int(event[1])
        if rc != 0:
            yield ("rc", rc)
            return

        ok = False
        val_messages: list[str] = []
        async for ev in _post_start_validation(
            profile, hf_cache_path=env.get("HF_CACHE_PATH") or None
        ):
            if ev[0] == "result":
                ok, val_messages = ev[1], ev[2]
            else:
                yield ev
        for msg in val_messages:
            yield ("log", msg)
        if not ok:
            yield ("rc", 1)
            return

        yield ("log", f"✓ 프로필 '{profile_name}' 활성화됨")
        yield ("log", f"  Endpoint: http://localhost:{profile.port}/v1")
        yield ("log", f"  Ready:    curl http://localhost:{profile.port}/v1/models")
        yield ("rc", 0)
        return


async def stream_container_prepare(profile_name: str, *, max_workers: int | None = None):
    stored = profile_store.load_profile(profile_name, "llamacpp")
    if stored is None:
        yield ("log", f"✗ 프로필 없음: llamacpp/{profile_name} (profiles.yaml 확인)")
        yield ("rc", 1)
        return

    ok, env_messages = validate_common_env(
        COMMON_ENV, require_downloader_image=True
    )
    for message in env_messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    profile = load_profile(profile_name)

    try:
        ok, cfg_messages = _ensure_profile_config(stored, profile)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("log", f"✗ config 저장 실패: {exc}")
        yield ("rc", 1)
        return
    for message in cfg_messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    try:
        profile, env_path = _render_profile_snapshot(profile_name)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("log", f"✗ profile env render 실패: {exc}")
        yield ("rc", 1)
        return

    try:
        hf_file = prepare.resolve_gguf_file(load_config(profile.config_name or profile.name).get("model-file"), profile.model_file, profile.hf_file)
    except (OSError, RuntimeError, ValueError) as exc:
        yield ("log", f"Error: {exc}")
        yield ("rc", 1)
        return
    if not profile.hf_repo or not hf_file:
        yield ("log", f"✗ '{profile_name}' 에 hf_repo / hf_file 이 없습니다 — 받을 GGUF 를 특정할 수 없습니다.")
        yield ("log", f"  llmux profile edit {profile_name} --hf-repo <org/repo> --hf-file <파일.gguf>")
        yield ("rc", 1)
        return

    yield ("log", f"▸ runtime env 렌더: {env_path}")

    rc, out = await _render_override(profile_name)
    if rc != 0:
        yield ("log", out.strip() or "✗ override 렌더 실패")
        yield ("rc", rc)
        return
    for line in out.splitlines():
        if line.startswith("warning:"):
            yield ("log", f"⚠ {line[len('warning:'):].strip()}")
    yield ("log", f"▸ command 렌더: {_override_path(profile_name)}")

    image_ref = _resolve_runtime_image(profile)
    image_credential_error = dev_build.image_reference_credential_error(image_ref)
    if image_credential_error:
        yield ("log", f"Error: invalid runtime image reference: {image_credential_error}")
        yield ("rc", 1)
        return
    try:
        present = await prepare.image_present(image_ref)
    except RuntimeError as exc:
        yield ("log", f"✗ Docker 이미지 확인 실패: {exc}")
        yield ("rc", 1)
        return
    if not present:
        if image_ref.startswith(f"{LLAMACPP_DEV_SPEC.image_prefix}:"):
            dev_tag = image_ref.split(":", 1)[1]
            yield ("log", f"✗ Dev image {image_ref} 가 로컬에 없습니다.")
            yield ("log", f"  먼저 빌드하세요: uv run llmux image build-dev --backend llamacpp --tag {dev_tag}")
            yield ("rc", 1)
            return
        async for event in prepare.stream_pull(image_ref):
            if event[0] == "rc":
                if int(event[1]) != 0:
                    yield ("log", f"✗ 이미지 pull 실패: {image_ref}")
                    yield ("rc", int(event[1]))
                    return
            else:
                yield event
    yield ("log", f"▸ Image: {image_ref}")

    cache_path = prepare.hf_cache_path()
    if not cache_path:
        yield ("log", "✗ .env.common 의 HF_CACHE_PATH 가 비어 있습니다 — 받을 위치가 없습니다.")
        yield ("rc", 1)
        return

    yield ("log", f"▸ {profile.hf_repo} / {hf_file} 다운로드 → {cache_path}")
    rc = -1
    async for event in prepare.stream_llamacpp_download(
        hf_repo=profile.hf_repo,
        hf_file=hf_file,
        cache_path=cache_path,
        token=prepare.hf_token(),
        container_name=prepare.prepare_container_name(profile.container_name),
        max_workers=max_workers,
    ):
        if event[0] == "rc":
            rc = int(event[1])
        else:
            yield event
    if rc != 0:
        yield ("log", f"✗ 다운로드 실패 (rc={rc}).")
        yield ("rc", rc if rc != 0 else 1)
        return

    yield ("log", f"✓ '{profile_name}' 준비 완료 — 시작: llmux up {profile_name}")
    yield ("rc", 0)


async def _container_exists(container_name: str) -> bool | None:
    rc, out = await _docker_run(
        "docker", "ps", "-a", "--format", "{{.Names}}", timeout=10
    )
    if rc != 0:
        return None
    return container_name in out.strip().splitlines()


async def _remove_compose_network(project_name: str) -> tuple[int, str]:
    rc, out = await _docker_run(
        "docker", "network", "rm", f"{project_name}_default", timeout=10
    )
    if rc == 0 or "No such network" in out:
        return 0, ""
    return rc or 1, out.strip() or "docker network rm failed"


async def container_down(profile_name: str) -> tuple[int, str]:
    try:
        stored = profile_store.load_profile(profile_name, "llamacpp")
    except Exception as exc:
        return 1, f"✗ llamacpp/{profile_name} 로드 실패: {exc}"
    if stored is None:
        return 1, f"✗ 프로필 없음: llamacpp/{profile_name}"

    try:
        profile = load_profile(profile_name)
    except Exception as exc:
        return 1, f"✗ '{profile_name}' runtime profile 로드 실패: {exc}"

    render_error = ""
    try:
        profile_store.render_env_for_profile(profile_name, "llamacpp")
    except Exception as exc:
        render_error = f"env 렌더 실패: {exc}"

    compose_err = render_error
    override = _override_path(profile_name)
    if not compose_err and override.exists():
        try:
            env = _compose_env(profile)
            cmd = [*_compose_base_args(profile), "down"]
            rc, out = await _run(*cmd, env=env, timeout=120)
            if rc == 0:
                return 0, f"✓ '{profile_name}' 중지 완료"
            compose_err = next(
                (line for line in reversed(out.strip().splitlines()) if line.strip()),
                f"rc={rc}",
            )
        except Exception as exc:
            compose_err = f"compose 준비 실패: {exc}"

    try:
        exists = await _container_exists(profile.container_name)
    except Exception as exc:
        return 1, f"✗ '{profile_name}' 컨테이너 상태 확인 실패: {exc}"
    if exists is None:
        return 1, (
            f"✗ '{profile_name}' 컨테이너 상태를 확인할 수 없음 "
            "(docker ps 실패 또는 타임아웃) — 중지 여부 불명"
        )

    if exists:
        rc_rm, out_rm = await _docker_run(
            "docker", "rm", "-f", profile.container_name, timeout=30
        )
        if rc_rm != 0:
            detail = out_rm.strip() or f"docker rm -f exited with status {rc_rm}"
            return rc_rm if rc_rm > 0 else 1, (
                f"✗ '{profile_name}' 수동 정리 실패: {detail}"
            )
        network_rc, network_out = await _remove_compose_network(profile.name)
        if network_rc != 0:
            return network_rc, (
                f"✗ '{profile_name}' 컨테이너는 중지됐지만 네트워크 정리 실패: "
                f"{network_out}"
            )
        detail = f" (compose down 실패: {compose_err})" if compose_err else ""
        return 0, f"✓ '{profile_name}' 중지 완료 (rm -f){detail}"

    if render_error:
        return 0, f"({profile_name}) 컨테이너 없음 — 이미 중지됨 (⚠ {render_error})"
    if compose_err:
        return 1, (
            f"✗ '{profile_name}' compose down 실패 후 컨테이너를 찾을 수 없음: "
            f"{compose_err}"
        )
    return 0, f"({profile_name}) 컨테이너 없음 — 이미 중지됨"
