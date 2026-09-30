"""System inspection, version lookup, and vLLM metadata helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

from tui.common import profile_store
from tui.common.docker import (  # noqa: F401 — re-exported for backward compat
    GpuInfo,
    format_gpu_bar,
    get_gpu_info,
)
from tui.common.mem import estimate_model_memory  # noqa: F401 — re-exported

from tui.common.ssl_ctx import open_url, same_origin

from .backend_common import CONFIG_DIR, DockerImage
from .backend_process import run_command

logger = logging.getLogger(__name__)





VLLM_OFFICIAL_REPO = "vllm/vllm-openai"
_DOCKERHUB_TAG_PAGE_CAP = 100


def resolve_vllm_image_ref(image_tag: str) -> str:
    if "/" in image_tag or ":" in image_tag:
        return image_tag
    return f"{VLLM_OFFICIAL_REPO}:{image_tag}"


async def get_docker_images(repo: str = VLLM_OFFICIAL_REPO) -> list[DockerImage]:
    """Local images for `repo`. Raises if the probe fails — an empty list must
    mean "no such image", never "docker could not be reached"."""
    from tui.common.dev_build import image_reference_credential_error

    error = image_reference_credential_error(repo)
    if error:
        raise RuntimeError(error)
    rc, out = await run_command(
        "docker",
        "images",
        repo,
        "--format",
        "{{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}",
        timeout=10,
    )
    if rc != 0:
        raise RuntimeError(
            f"docker images {repo} failed or timed out: {out.strip() or 'no output'}"
        )
    from tui.common.docker import parse_docker_image_rows

    return [DockerImage(*row) for row in parse_docker_image_rows(out)]


async def get_dev_images() -> list[DockerImage]:
    from tui.backends.vllm.backend_runtime import VLLM_DEV_SPEC

    return await get_docker_images(VLLM_DEV_SPEC.image_prefix)


def _parse_stable_version_tag(tag: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _pick_preferred_tag(tags: list[str]) -> str | None:
    """Pick the highest semantic-version tag (e.g. v0.19.1). Returns None if the
    image has only moving tags like `latest` / `nightly` / `<none>`.

    We deliberately ignore `latest` and `nightly` because they don't describe
    the actual image contents — they're just aliases that upstream rewrites.
    """
    stable_tags = [
        (version, tag)
        for tag in tags
        if (version := _parse_stable_version_tag(tag)) is not None
    ]
    if stable_tags:
        return max(stable_tags)[1]
    return None


async def get_local_latest_tag() -> str:
    """Return the highest-version local vllm/vllm-openai tag.

    Only semver-style tags (e.g. `v0.19.1`) are considered. `latest` and
    `nightly` are skipped because they don't self-describe. If no versioned
    tag exists locally, returns "none" so the UI can prompt the user to pull
    a specific version.
    """
    rc, out = await run_command(
        "docker",
        "images",
        VLLM_OFFICIAL_REPO,
        "--format",
        "{{.ID}}\t{{.Tag}}",
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(
            f"docker images failed or timed out: {out.strip() or 'no output'}"
        )
    if not out.strip():
        return "none"

    image_tags: dict[str, list[str]] = {}
    for line in out.strip().splitlines():
        image_id, _, tag = line.partition("\t")
        image_id = image_id.strip()
        tag = tag.strip()
        if not image_id or not tag or tag == "<none>":
            continue
        image_tags.setdefault(image_id, []).append(tag)

    best_version: tuple[int, int, int] | None = None
    best_tag = "none"
    for tags in image_tags.values():
        preferred = _pick_preferred_tag(tags)
        if preferred is None:
            continue
        version = _parse_stable_version_tag(preferred)
        if version is None:
            continue
        if best_version is None or version > best_version:
            best_version = version
            best_tag = preferred

    return best_tag


async def _fetch_json_url(
    url: str,
    timeout: float = 5.0,
    *,
    headers: dict[str, str] | None = None,
) -> dict | None:
    """Fetch JSON from URL in a thread to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()

    def _fetch() -> dict | None:
        request_headers = {
            "Accept": "application/json",
            "User-Agent": "llmux/1.0 (+https://github.com/Changroro/llmux)",
        }
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(url, headers=request_headers)
        try:
            with open_url(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8", errors="replace")
            data = json.loads(payload)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.debug("fetch %s failed: %s", url, exc)
        return None

    return await loop.run_in_executor(None, _fetch)


async def _fetch_docker_registry_tags() -> list[str]:
    token_url = (
        "https://auth.docker.io/token?"
        "service=registry.docker.io&scope=repository:vllm/vllm-openai:pull"
    )
    token_payload = await _fetch_json_url(token_url, timeout=5.0)
    token = str((token_payload or {}).get("token", "")).strip()
    if not token:
        return []

    tags_payload = await _fetch_json_url(
        "https://registry-1.docker.io/v2/vllm/vllm-openai/tags/list?n=1000",
        timeout=5.0,
        headers={"Authorization": f"Bearer {token}"},
    )
    tags = (tags_payload or {}).get("tags", [])
    return [str(tag) for tag in tags if tag]


async def get_dockerhub_release_version() -> str:
    """Get the latest exact stable release version from Docker Hub."""
    base_urls = [
        "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100",
        "https://registry.hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100",
    ]
    for attempt in range(3):
        for base_url in base_urls:
            url = base_url
            pages_checked = 0
            visited: set[str] = set()
            stable_tags: list[tuple[tuple[int, int, int], str]] = []
            failed = False
            while url:
                if url in visited:
                    raise RuntimeError("DockerHub pagination repeated a page URL")
                visited.add(url)
                if pages_checked >= _DOCKERHUB_TAG_PAGE_CAP:
                    raise RuntimeError("DockerHub pagination exceeded the page limit")
                data = await _fetch_json_url(url, timeout=5.0)
                if not data:
                    failed = True
                    break
                stable_tags.extend(
                    (version, name)
                    for result in data.get("results", [])
                    if isinstance(result, dict)
                    if (name := str(result.get("name", "")))
                    if (version := _parse_stable_version_tag(name)) is not None
                )
                next_url = data.get("next", "")
                next_url = str(next_url) if next_url else ""
                next_url = urljoin(url, next_url) if next_url else ""
                if next_url and not same_origin(base_url, next_url):
                    raise RuntimeError(
                        "DockerHub pagination refused an off-origin URL"
                    )
                url = next_url
                pages_checked += 1
            if not failed and stable_tags:
                return max(stable_tags)[1]
        if attempt < 2:
            await asyncio.sleep(0.5)
    registry_preferred = _pick_preferred_tag(await _fetch_docker_registry_tags())
    if registry_preferred:
        return registry_preferred
    return "unknown"


async def get_dockerhub_nightly_date() -> str:
    """Get last updated date of the nightly tag from Docker Hub."""
    urls = [
        "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/nightly",
        "https://registry.hub.docker.com/v2/repositories/vllm/vllm-openai/tags/nightly",
    ]
    for attempt in range(3):
        for url in urls:
            data = await _fetch_json_url(url, timeout=5.0)
            if not data:
                continue
            last_updated = str(data.get("last_updated", "")).strip()
            if last_updated:
                return last_updated.split("T")[0]
        if attempt < 2:
            await asyncio.sleep(0.5)
    registry_tags = await _fetch_docker_registry_tags()
    return "available" if "nightly" in registry_tags else "unknown"


_VLLM_PARAMS_CACHE_DIR = CONFIG_DIR

_FLAG_RE = re.compile(r"(?<![A-Za-z0-9_-])--([A-Za-z][A-Za-z0-9_-]*)")


def _validate_flags(raw: object) -> set[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("root must be a non-empty list")
    if any(
        not isinstance(flag, str) or not _FLAG_RE.fullmatch(f"--{flag}")
        for flag in raw
    ):
        raise ValueError("items must be valid flag names")
    return set(raw)


def _load_flag_cache(path: Path) -> set[str]:
    return _validate_flags(json.loads(path.read_text()))




async def extract_vllm_params(image_tag: str = "", *, container_name: str = "") -> set[str]:
    """Extract valid vllm serve parameters from a docker image."""
    if image_tag:
        image_ref = resolve_vllm_image_ref(image_tag)
    else:
        from .backend_runtime import _resolve_prepare_image
        from .backend_common import Profile

        image_ref, error = await _resolve_prepare_image(Profile(name=""))
        if error:
            raise RuntimeError(error)

    from tui.common.dev_build import image_reference_credential_error

    error = image_reference_credential_error(image_ref)
    if error:
        raise RuntimeError(error)

    from tui.common.docker import image_identity

    identity = await image_identity(image_ref) if not container_name else None
    cache_file = None
    if identity is not None:
        cache_key = hashlib.sha256(f"full-help-v2:{image_ref}@{identity}".encode()).hexdigest()[:16]
        cache_file = _VLLM_PARAMS_CACHE_DIR / f".vllm-params-{cache_key}.json"
    if cache_file is not None and cache_file.exists():
        try:
            return _load_flag_cache(cache_file)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"invalid vLLM flag cache {cache_file}: {exc}") from exc

    command = (
        ["docker", "exec", container_name, "vllm", "serve"]
        if container_name else
        ["docker", "run", "--rm", "--pull=never", "--entrypoint", "vllm", identity or image_ref, "serve"]
    )
    rc, out = await run_command(*command, "--help", timeout=30)
    if rc == 0 and "--help=all" in out:
        rc, out = await run_command(*command, "--help=all", timeout=30)
    if rc != 0 or not out.strip():
        raise RuntimeError(
            f"could not inspect vLLM flags from {image_ref}: "
            f"{out.strip() or 'docker run failed'}"
        )

    params = set(_FLAG_RE.findall(out))
    if not params:
        try:
            params = _validate_flags(json.loads(out))
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"could not parse vLLM flags from {image_ref}: {exc}") from exc
    if identity is None and not container_name:
        identity = await image_identity(image_ref)
        if identity is not None:
            cache_key = hashlib.sha256(f"full-help-v2:{image_ref}@{identity}".encode()).hexdigest()[:16]
            cache_file = _VLLM_PARAMS_CACHE_DIR / f".vllm-params-{cache_key}.json"
    if cache_file is not None:
        _VLLM_PARAMS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        profile_store._atomic_write(cache_file, json.dumps(sorted(params)))
    return params
