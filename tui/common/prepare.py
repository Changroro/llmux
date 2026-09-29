"""Model prefetch primitives shared by both backends' `prepare` flow.

`prepare` renders a profile's runtime files, makes sure its image is on disk and
downloads the weights — then stops. The server never starts, so nothing is
loaded onto a GPU.

Downloads run inside a throwaway container that calls
`huggingface_hub.snapshot_download`, so both backends pull over parallel
workers and neither needs an `hf` CLI on the host. llama.cpp's own `-hf` fetch
is a single stream, so its GGUFs come down through the downloader image named
by PREPARE_DOWNLOADER_IMAGE instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import time
from pathlib import Path, PurePosixPath

from tui.common.env import (
    host_expand,
    parse_env_file,
    required_image_reference_error,
)
from tui.common.i18n import t
from tui.common.profile_store import PROJECT_ROOT, hf_repo_error

COMMON_ENV = PROJECT_ROOT / ".env.common"

_PROGRESS_INTERVAL = 1.0

# `original/` holds the Meta-format checkpoint of Llama-style repos — vLLM loads
# the HF-format weights next to it, so pulling both doubles the download.
_VLLM_IGNORE_PATTERNS = ["original/**"]

# HF caps a single connection at a few MB/s, so the worker count is really a
# bandwidth dial — see PREPARE_MAX_WORKERS in .env.common.
_WORKERS_SNIPPET = (
    "workers=os.environ.get('LLMUX_PREPARE_WORKERS');"
    "extra={'max_workers': int(workers)} if workers else {};"
)

_VLLM_DOWNLOAD_SNIPPET = (
    "import os;"
    "from huggingface_hub import snapshot_download;"
    "ignore=os.environ['LLMUX_PREPARE_IGNORE'].split(',');"
    + _WORKERS_SNIPPET +
    "print('skipping (vLLM does not load it):', ignore, extra);"
    "print('snapshot:', snapshot_download("
    "os.environ['LLMUX_PREPARE_MODEL'], revision=os.environ.get('LLMUX_PREPARE_REVISION') or None, ignore_patterns=ignore, **extra))"
)

_GGUF_DOWNLOAD_SNIPPET = (
    "import os;"
    "from huggingface_hub import snapshot_download;"
    "allow=os.environ['LLMUX_PREPARE_ALLOW'].split(',');"
    + _WORKERS_SNIPPET +
    "print('fetching:', allow, extra);"
    "print('snapshot:', snapshot_download("
    "os.environ['LLMUX_PREPARE_MODEL'], allow_patterns=allow, **extra))"
)

_SPLIT_SHARD_RE = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$"
)


def resolve_gguf_file(configured: str | None, model_file: str, hf_file: str) -> str:
    if configured is not None and not isinstance(configured, str):
        raise ValueError("config key 'model-file' must be a string")
    return (hf_file or configured or model_file).strip()


def hf_file_error(hf_file: str) -> str:
    if not hf_file:
        return "HF file must be a non-empty relative POSIX path"
    if any(ord(character) < 32 or ord(character) == 127 for character in hf_file):
        return "HF file cannot include control characters"
    path = PurePosixPath(hf_file)
    if path.is_absolute() or ".." in path.parts:
        return "HF file must be a normalized relative POSIX path"
    if path.as_posix() != hf_file or "." in path.parts:
        return "HF file must be a normalized relative POSIX path"
    return ""


def gguf_shard_names(hf_file: str) -> list[str]:
    """Every shard of a split GGUF, or just the file when it is not split.

    Shards are far from equal — unsloth's first one can be 10MB against a 50GB
    sibling — so a profile's `hf_file` is only ever the entry point.
    """
    error = hf_file_error(hf_file)
    if error:
        raise ValueError(error)
    directory, _, name = hf_file.rpartition("/")
    match = _SPLIT_SHARD_RE.match(name)
    if match is None:
        return [hf_file]
    prefix = f"{directory}/" if directory else ""
    stem = match.group("stem")
    index = int(match.group("index"))
    total = int(match.group("total"))
    if total < 1:
        raise ValueError(
            f"invalid split GGUF shard {hf_file!r}: total must be at least 1"
        )
    if index < 1 or index > total:
        raise ValueError(
            f"invalid split GGUF shard {hf_file!r}: index {index} is outside 1..{total}"
        )
    return [f"{prefix}{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


def prepare_max_workers() -> int | None:
    """Download connections from .env.common (None when unset)."""
    raw = _common().get("PREPARE_MAX_WORKERS", "").strip()
    if not raw:
        return None
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(
            f"PREPARE_MAX_WORKERS must be a positive integer, got {raw!r}"
        )
    return int(raw)


def workers_env(max_workers: int | None) -> list[str]:
    """`docker run` args pinning the worker count, empty when nothing is set."""
    resolved = max_workers if max_workers is not None else prepare_max_workers()
    if resolved is None:
        return []
    return ["-e", f"LLMUX_PREPARE_WORKERS={resolved}"]


def downloader_image() -> str:
    """Image whose python3 runs snapshot_download ("" when unset)."""
    return _common().get("PREPARE_DOWNLOADER_IMAGE", "").strip()


def _common() -> dict[str, str]:
    if not COMMON_ENV.exists():
        return {}
    return parse_env_file(COMMON_ENV)


def hf_cache_path() -> str:
    """Host HF cache directory from .env.common ("" when unset)."""
    raw = _common().get("HF_CACHE_PATH", "").strip()
    return host_expand(raw) if raw else ""


def hf_token() -> str:
    token = _common().get("HF_TOKEN", "").strip()
    return "" if token.startswith("your_") else token


def prepare_container_name(container_name: str) -> str:
    return f"{container_name}-prepare"


async def _run(*args: str, timeout: float = 30) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        return -1, f"Executable not found: {args[0] if args else '<empty>'}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        return -1, "Command timed out"
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
    rc = proc.returncode if proc.returncode is not None else -1
    return rc, (out or b"").decode(errors="replace")


async def _remove_prepare_container(container_name: str) -> tuple[int, str]:
    try:
        rc, output = await _run("docker", "rm", "-f", container_name, timeout=30)
    except OSError as exc:
        return 1, f"docker rm -f could not start: {exc}"
    if rc == 0:
        return 0, ""
    lowered = output.lower()
    if "no such container" in lowered or "no such object" in lowered:
        return 0, ""
    detail = output.strip() or f"docker rm -f exited with status {rc}"
    return rc if rc > 0 else 1, detail


async def stream_lines(args: list[str], *, env: dict[str, str] | None = None):
    """Yield ("log", line) then ("rc", code), splitting on CR as well as LF.

    Download progress is redrawn with a bare CR, so a plain readline() would
    hold the whole transfer in one line; CR-terminated redraws are throttled to
    one per second so a multi-GB pull can't flood the log.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        yield ("log", f"✗ Executable not found: {args[0] if args else '<empty>'}")
        yield ("rc", -1)
        return

    buf = b""
    last_progress = 0.0
    try:
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                match = re.search(rb"\r\n|\r|\n", buf)
                if match is None:
                    break
                line, delim = buf[: match.start()], match.group()
                buf = buf[match.end():]
                text = line.decode(errors="replace").rstrip()
                if not text:
                    continue
                if delim == b"\r":
                    now = time.monotonic()
                    if now - last_progress < _PROGRESS_INTERVAL:
                        continue
                    last_progress = now
                yield ("log", text)
        tail = buf.decode(errors="replace").rstrip()
        if tail:
            yield ("log", tail)
        await proc.wait()
        yield ("rc", proc.returncode if proc.returncode is not None else -1)
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()


async def image_present(image_ref: str) -> bool:
    rc, output = await _run("docker", "image", "inspect", image_ref, timeout=20)
    if rc == 0:
        return True
    lowered = output.lower()
    if "no such image" in lowered or "no such object" in lowered:
        return False
    raise RuntimeError(output.strip() or f"docker image inspect {image_ref} failed")


async def stream_pull(image_ref: str):
    """Pull `image_ref`, streaming docker's own progress output."""
    yield ("log", t(f"▸ Pulling image: {image_ref}", f"▸ 이미지 받는 중: {image_ref}"))
    async for event in stream_lines(["docker", "pull", image_ref]):
        yield event


async def stream_vllm_download(
    *,
    image_ref: str,
    model_id: str,
    cache_path: str,
    token: str,
    container_name: str,
    max_workers: int | None = None,
    revision: str = "",
):
    """Download a HF snapshot with the image's own huggingface_hub."""
    cleanup_rc, cleanup_error = await _remove_prepare_container(container_name)
    if cleanup_rc != 0:
        yield ("log", f"Error: prepare cleanup failed: {cleanup_error}")
        yield ("rc", cleanup_rc)
        return
    args = [
        "docker", "run", "--rm",
        "--name", container_name,
        "-v", f"{cache_path}:/root/.cache/huggingface",
        "-e", f"LLMUX_PREPARE_MODEL={model_id}",
        "-e", f"LLMUX_PREPARE_IGNORE={','.join(_VLLM_IGNORE_PATTERNS)}",
    ] + workers_env(max_workers)
    if revision:
        args += ["-e", f"LLMUX_PREPARE_REVISION={revision}"]
    process_env = None
    if token:
        args += ["-e", "HF_TOKEN", "-e", "HUGGING_FACE_HUB_TOKEN"]
        process_env = os.environ.copy()
        process_env["HF_TOKEN"] = token
        process_env["HUGGING_FACE_HUB_TOKEN"] = token
    args += ["--entrypoint", "python3", image_ref, "-c", _VLLM_DOWNLOAD_SNIPPET]

    try:
        stream = stream_lines(args, env=process_env) if process_env else stream_lines(args)
        async for event in stream:
            yield event
    except (asyncio.CancelledError, GeneratorExit) as exc:
        cleanup_rc, cleanup_error = await _remove_prepare_container(container_name)
        if cleanup_rc != 0:
            raise RuntimeError(
                f"prepare cancelled and container cleanup failed: {cleanup_error}"
            ) from exc
        raise


def hf_repo_cache_dir(cache_path: str | Path, hf_repo: str) -> Path:
    error = hf_repo_error(hf_repo)
    if error:
        raise ValueError(error)
    org, _, name = hf_repo.partition("/")
    cache_root = Path(cache_path)
    repo_dir = cache_root / "hub" / f"models--{org}--{name}"
    if not repo_dir.resolve().is_relative_to(cache_root.resolve()):
        raise RuntimeError(
            "Hugging Face repository cache resolves outside the configured cache root"
        )
    return repo_dir


def current_hf_snapshot(repo_dir: Path) -> tuple[str, Path] | None:
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    ref = repo_dir / "refs" / "main"
    if ref.exists():
        if not ref.resolve(strict=True).is_relative_to(repo_dir.resolve(strict=True)):
            raise RuntimeError("Hugging Face refs/main resolves outside its repository cache root")
        revision = ref.read_text().strip()
        if (
            not revision
            or any(ord(character) < 32 or ord(character) == 127 for character in revision)
            or PurePosixPath(revision).name != revision
        ):
            raise RuntimeError("invalid Hugging Face refs/main revision")
        snapshot = snapshots / revision
        if not snapshot.is_dir():
            return None
        resolved = snapshot.resolve(strict=True)
        if not resolved.is_relative_to(repo_dir.resolve(strict=True)):
            raise RuntimeError("Hugging Face refs/main resolves outside its repository cache root")
        return revision, snapshot

    candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
    if not candidates:
        return None
    if len(candidates) > 1:
        raise RuntimeError(
            "Hugging Face cache has multiple revisions but no refs/main"
        )
    snapshot = candidates[0]
    resolved = snapshot.resolve(strict=True)
    if not resolved.is_relative_to(repo_dir.resolve(strict=True)):
        raise RuntimeError("Hugging Face snapshot resolves outside its repository cache root")
    return snapshot.name, snapshot


def resolve_cache_entry(path: Path, repo_dir: Path) -> Path | None:
    if not path.exists():
        return None
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(repo_dir.resolve(strict=True)):
        raise RuntimeError("cached GGUF resolves outside its repository cache root")
    return resolved


def gguf_in_cache(cache_path: str, hf_repo: str, hf_file: str) -> Path | None:
    """The cached entry point for `hf_repo`/`hf_file`, or None.

    Every shard has to be there, not just the one the profile names: a partial
    set would otherwise read as a hit and skip the rest of the download.
    snapshot_download links a snapshot entry only once its blob is complete, so
    a full set of links means the model is whole.
    """
    repo_dir = hf_repo_cache_dir(cache_path, hf_repo)
    shards = gguf_shard_names(hf_file)
    selected = current_hf_snapshot(repo_dir)
    if selected is None:
        return None
    _, snapshot = selected
    paths = [snapshot / shard for shard in shards]
    if any(resolve_cache_entry(path, repo_dir) is None for path in paths):
        return None
    return paths[0]


async def stream_llamacpp_download(
    *,
    hf_repo: str,
    hf_file: str,
    cache_path: str,
    token: str,
    container_name: str,
    max_workers: int | None = None,
):
    """Fetch a GGUF — every shard of a split one — into the host HF cache.

    The transfer runs in the PREPARE_DOWNLOADER_IMAGE container, not the
    llama.cpp server image: the weights are pulled by huggingface_hub and never
    reach a GPU. HF caps a single connection at a few MB/s, so `max_workers` is
    the knob for how much of the line the download is allowed to take.
    """
    try:
        shards = gguf_shard_names(hf_file)
    except ValueError as exc:
        yield ("log", f"Error: {exc}")
        yield ("rc", 1)
        return

    image_ref = downloader_image()
    image_error = required_image_reference_error(
        "PREPARE_DOWNLOADER_IMAGE", image_ref
    )
    if image_error:
        yield ("log", image_error)
        yield ("rc", 1)
        return

    cached = gguf_in_cache(cache_path, hf_repo, hf_file)
    if cached is not None:
        yield ("log", t(f"▸ Already in the HF cache: {cached}",
                        f"▸ 이미 HF 캐시에 있음: {cached}"))
        yield ("rc", 0)
        return

    try:
        present = await image_present(image_ref)
    except RuntimeError as exc:
        yield ("log", t(f"✗ image probe failed: {exc}", f"✗ 이미지 확인 실패: {exc}"))
        yield ("rc", 1)
        return
    if not present:
        async for event in stream_pull(image_ref):
            if event[0] == "rc":
                if int(event[1]) != 0:
                    yield ("log", t(f"✗ could not pull {image_ref}",
                                    f"✗ downloader 이미지 pull 실패: {image_ref}"))
                    yield ("rc", int(event[1]))
                    return
            else:
                yield event

    cleanup_rc, cleanup_error = await _remove_prepare_container(container_name)
    if cleanup_rc != 0:
        yield ("log", f"Error: prepare cleanup failed: {cleanup_error}")
        yield ("rc", cleanup_rc)
        return
    args = [
        "docker", "run", "--rm",
        "--name", container_name,
        "-v", f"{cache_path}:/root/.cache/huggingface",
        "-e", f"LLMUX_PREPARE_MODEL={hf_repo}",
        "-e", f"LLMUX_PREPARE_ALLOW={','.join(shards)}",
    ]
    args += workers_env(max_workers)
    process_env = None
    if token:
        args += ["-e", "HF_TOKEN", "-e", "HUGGING_FACE_HUB_TOKEN"]
        process_env = os.environ.copy()
        process_env["HF_TOKEN"] = token
        process_env["HUGGING_FACE_HUB_TOKEN"] = token
    args += ["--entrypoint", "python3", image_ref, "-c", _GGUF_DOWNLOAD_SNIPPET]

    rc = -1
    try:
        stream = stream_lines(args, env=process_env) if process_env else stream_lines(args)
        async for event in stream:
            if event[0] == "rc":
                rc = int(event[1])
            else:
                yield event
    except (asyncio.CancelledError, GeneratorExit) as exc:
        cleanup_rc, cleanup_error = await _remove_prepare_container(container_name)
        if cleanup_rc != 0:
            raise RuntimeError(
                f"prepare cancelled and container cleanup failed: {cleanup_error}"
            ) from exc
        raise

    if rc != 0:
        yield ("rc", rc)
        return

    cached = gguf_in_cache(cache_path, hf_repo, hf_file)
    if cached is None:
        yield ("log", t(f"✗ {hf_file} is not in the cache after the download",
                        f"✗ 다운로드 후에도 캐시에 {hf_file} 이 없습니다"))
        yield ("rc", 1)
        return

    yield ("log", t(f"▸ Downloaded: {cached}", f"▸ 다운로드 완료: {cached}"))
    yield ("rc", 0)
