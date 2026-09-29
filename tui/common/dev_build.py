"""Backend-agnostic development image builds."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit


def sanitize_repo_url(repo_url: str) -> str:
    parsed = urlsplit(repo_url)
    if not parsed.scheme or parsed.hostname is None:
        return repo_url
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = f"{hostname}:{parsed.port}" if parsed.port is not None else hostname
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def repo_url_error(repo_url: str) -> str:
    if any(ord(character) < 32 or ord(character) == 127 for character in repo_url):
        return "repository URL cannot include control characters"
    try:
        parsed = urlsplit(repo_url)
    except ValueError:
        return "repository URL is invalid"
    if parsed.query or parsed.fragment:
        return "repository URL cannot include a query or fragment"
    if parsed.password is not None or (
        parsed.username is not None and parsed.scheme != "ssh"
    ):
        return "repository URL cannot include inline credentials"
    return ""


_INVALID_GIT_REF_CHAR_RE = re.compile(r"[\x00-\x20\x7f~^:?*\[\\]")


def git_branch_error(branch: str) -> str:
    if not branch:
        return "Git branch must be non-empty"
    if branch.startswith("-"):
        return "Git branch cannot begin with '-'"
    if branch == "@":
        return "Git branch cannot be '@'"
    if _INVALID_GIT_REF_CHAR_RE.search(branch):
        return "Git branch contains a character forbidden by Git ref syntax"
    if (
        branch.startswith("/")
        or branch.endswith("/")
        or "//" in branch
        or branch.endswith(".")
        or ".." in branch
        or "@{" in branch
    ):
        return "Git branch is not a valid Git ref"
    if any(
        component.startswith(".") or component.endswith(".lock")
        for component in branch.split("/")
    ):
        return "Git branch is not a valid Git ref"
    return ""


_HTTP_URL_RE = re.compile(r"https?://[^\s'\"<>]+")


def _redact_git_output(output: str, repo_url: str) -> str:
    redacted = output.replace(repo_url, sanitize_repo_url(repo_url))
    redacted = _HTTP_URL_RE.sub(
        lambda match: sanitize_repo_url(match.group(0)), redacted
    )
    parsed = urlsplit(repo_url)
    for credential in (parsed.username, parsed.password):
        if credential:
            redacted = redacted.replace(unquote(credential), "***")
    return redacted


def _git_transport(
    repo_url: str,
) -> tuple[str, dict[str, str] | None, tempfile.TemporaryDirectory[str] | None]:
    parsed = urlsplit(repo_url)
    if parsed.query or parsed.fragment:
        raise ValueError("repository URLs cannot include a query or fragment")
    has_credentials = parsed.username is not None or parsed.password is not None
    if not has_credentials:
        return repo_url, None, None
    if parsed.scheme == "ssh" and parsed.password is None:
        return repo_url, None, None
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError(
            "repository URL credentials are supported only for HTTP(S) repositories"
        )
    temp_dir = tempfile.TemporaryDirectory(prefix="llmux-git-auth-")
    askpass = Path(temp_dir.name) / "askpass.sh"
    askpass.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s\\n' \"$LLMUX_GIT_AUTH_USERNAME\" ;;\n"
        "  *Password*) printf '%s\\n' \"$LLMUX_GIT_AUTH_PASSWORD\" ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n"
    )
    askpass.chmod(0o700)
    env = os.environ.copy()
    env.update(
        {
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "credential.helper",
            "GIT_CONFIG_VALUE_0": "",
            "LLMUX_GIT_AUTH_USERNAME": unquote(parsed.username or ""),
            "LLMUX_GIT_AUTH_PASSWORD": unquote(parsed.password or ""),
        }
    )
    return sanitize_repo_url(repo_url), env, temp_dir


@dataclass(frozen=True)
class DevBuildSpec:
    backend: str
    image_prefix: str
    src_dir: Path
    default_repo_url: str
    default_branch: str = "main"
    dockerfile_relpath: str = ""
    target: str = ""
    base_build_args: tuple[tuple[str, str], ...] = ()
    label_prefix: str = ""


def _label(spec: DevBuildSpec) -> str:
    return spec.label_prefix or spec.backend


_TAG_INVALID_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_docker_tag(name: str) -> str:
    """Replace Docker-tag-invalid characters with `-`."""
    sanitized = _TAG_INVALID_CHARS.sub("-", name).lstrip(".-")
    return sanitized or "branch"


def _dev_image_prefixes() -> tuple[str, ...]:
    from tui.backends.llamacpp.backend_runtime import LLAMACPP_DEV_SPEC
    from tui.backends.vllm.backend_runtime import VLLM_DEV_SPEC

    return tuple(
        f"{spec.image_prefix}:" for spec in (VLLM_DEV_SPEC, LLAMACPP_DEV_SPEC)
    )


_TAG_PART_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
_DIGEST_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._+-]*:[A-Fa-f0-9]+$")


def image_reference_credential_error(value: str) -> str:
    if not value:
        return ""
    if "://" in value or "?" in value or "#" in value:
        return (
            "image reference cannot include URL credentials, a query, or a fragment"
        )
    if any(character.isspace() or ord(character) < 32 for character in value):
        return "image reference cannot include whitespace or control characters"
    if "@" in value and not _DIGEST_RE.fullmatch(value.rsplit("@", 1)[1]):
        return (
            "image reference cannot include URL credentials, a query, or a fragment"
        )
    return ""


def image_tag_error(value: str) -> str:
    """Return why an image reference is unusable, or an empty string."""
    value = value.strip()
    if not value:
        return ""
    credential_error = image_reference_credential_error(value)
    if credential_error:
        return credential_error
    for prefix in _dev_image_prefixes():
        if value.startswith(prefix):
            tag = value[len(prefix):]
            if not tag:
                return f"{prefix.rstrip(':')} image needs a tag after ':'"
            safe = sanitize_docker_tag(tag)
            if tag != safe:
                return (
                    f"invalid dev image tag {tag!r}; docker tags can't contain "
                    f"'/' or other specials — did you mean {prefix}{safe}?"
                )
            return ""
    has_tag = ":" in value and value.rfind(":") > value.rfind("/")
    if not has_tag:
        return (
            f"image reference {value!r} has no tag and would resolve to "
            "`:latest`; pin a specific version tag."
        )
    tag = value.rsplit(":", 1)[1]
    if tag == "latest":
        return (
            "`:latest` is an ambiguous alias and is not allowed; pin a specific "
            "version tag."
        )
    if not _TAG_PART_RE.match(tag):
        return (
            f"invalid image tag {tag!r}; must match {_TAG_PART_RE.pattern} "
            "(start with a letter/digit/underscore, then letters/digits/._-)"
        )
    return ""


async def _run(
    *args: str,
    cwd: Path | None = None,
    timeout: float = 30,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
    except FileNotFoundError:
        return -1, f"command not found: {args[0]}"
    except OSError as exc:
        return -1, f"failed to start {args[0]}: {exc}"
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
    return proc.returncode or 0, (stdout or b"").decode(errors="replace")


async def _stream(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
):
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
    except FileNotFoundError:
        yield ("log", f"Error: command not found: {args[0] if args else '<empty>'}")
        yield ("rc", 127)
        return
    except OSError as exc:
        yield (
            "log",
            f"Error: failed to start {args[0] if args else '<empty>'}: {exc}",
        )
        yield ("rc", 126)
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
        yield ("rc", proc.returncode or 0)
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()


async def clone_or_update(spec: DevBuildSpec, repo_url: str, branch: str):
    """Clone or update a checkout without silently switching its remote."""
    error = repo_url_error(repo_url)
    if error:
        yield ("log", f"Error: {error}")
        yield ("rc", 1)
        return
    branch_error = git_branch_error(branch)
    if branch_error:
        yield ("log", f"Error: {branch_error}")
        yield ("rc", 1)
        return
    try:
        transport_url, git_env, auth_temp_dir = _git_transport(repo_url)
    except ValueError as exc:
        yield ("log", f"Error: {exc}")
        yield ("rc", 1)
        return

    try:
        if spec.src_dir.joinpath(".git").exists():
            yield ("log", f"Updating existing {spec.backend} source...")
            rc, current = await _run(
                "git",
                "remote",
                "get-url",
                "origin",
                cwd=spec.src_dir,
                timeout=30,
                env=git_env,
            )
            if rc != 0:
                message = _redact_git_output(current, repo_url).strip()
                yield (
                    "log",
                    message
                    or f"Error: failed to inspect existing {spec.backend} source",
                )
                yield ("rc", 1)
                return
            current_url = current.strip()
            if sanitize_repo_url(current_url) != sanitize_repo_url(repo_url):
                yield (
                    "log",
                    f"Error: existing {spec.src_dir.name} remote URL differs from the requested repository.",
                )
                yield ("log", f"Existing: {sanitize_repo_url(current_url)}")
                yield ("log", f"Requested: {sanitize_repo_url(repo_url)}")
                yield (
                    "log",
                    f"Move or delete {spec.src_dir.name} yourself if you want to replace the checkout.",
                )
                yield ("rc", 1)
                return
            if current_url != transport_url:
                rc, out = await _run(
                    "git",
                    "remote",
                    "set-url",
                    "origin",
                    transport_url,
                    cwd=spec.src_dir,
                    timeout=30,
                    env=git_env,
                )
                if rc != 0:
                    yield (
                        "log",
                        _redact_git_output(out, repo_url).strip()
                        or "Error: failed to remove credentials from Git origin",
                    )
                    yield ("rc", rc)
                    return

        if not spec.src_dir.exists():
            async for event in _stream(
                ["git", "clone", transport_url, str(spec.src_dir)], env=git_env
            ):
                if event[0] == "log":
                    yield ("log", _redact_git_output(str(event[1]), repo_url))
                    continue
                if event[0] == "rc":
                    if event[1] != 0:
                        yield event
                        return
                    continue
                yield event

        rc, out = await _run(
            "git", "fetch", "origin", cwd=spec.src_dir, timeout=120, env=git_env
        )
        if rc != 0:
            yield (
                "log",
                _redact_git_output(out, repo_url).strip()
                or "Error: git fetch failed",
            )
            yield ("rc", rc)
            return

        rc, out = await _run(
            "git", "checkout", branch, cwd=spec.src_dir, timeout=60, env=git_env
        )
        if rc != 0:
            rc, out = await _run(
                "git",
                "checkout",
                "-b",
                branch,
                f"origin/{branch}",
                cwd=spec.src_dir,
                timeout=60,
                env=git_env,
            )
            if rc != 0:
                yield (
                    "log",
                    _redact_git_output(out, repo_url).strip()
                    or f"Error: failed to checkout branch {branch}",
                )
                yield ("rc", rc)
                return

        rc, out = await _run(
            "git",
            "pull",
            "origin",
            branch,
            cwd=spec.src_dir,
            timeout=120,
            env=git_env,
        )
        if rc != 0:
            yield (
                "log",
                _redact_git_output(out, repo_url).strip()
                or f"Error: git pull failed for branch {branch}",
            )
            yield (
                "log",
                f"Hint: stash or reset local changes in {spec.src_dir.name}/, then retry.",
            )
            yield ("rc", rc)
            return

        rc, sha = await _run(
            "git",
            "rev-parse",
            "--short",
            "HEAD",
            cwd=spec.src_dir,
            timeout=30,
            env=git_env,
        )
        if rc != 0:
            yield (
                "log",
                _redact_git_output(sha, repo_url).strip()
                or "Error: failed to read commit hash",
            )
            yield ("rc", rc)
            return
        yield ("commit", sha.strip())
    finally:
        if auth_temp_dir is not None:
            auth_temp_dir.cleanup()


async def stream_build(
    spec: DevBuildSpec,
    branch: str,
    *,
    repo_url: str = "",
    custom_tag: str = "",
    extra_build_args: tuple[tuple[str, str], ...] = (),
    extra_log_lines: tuple[str, ...] = (),
    pre_build=None,
    extra_labels: tuple[tuple[str, str], ...] = (),
):
    """Clone or update source and stream a backend-prefixed Docker build."""
    resolved_repo = repo_url or spec.default_repo_url
    error = repo_url_error(resolved_repo)
    if error:
        yield ("log", f"Error: {error}")
        yield ("rc", 1)
        return
    image_error = image_reference_credential_error(spec.image_prefix)
    if image_error:
        yield ("log", f"Error: {image_error}")
        yield ("rc", 1)
        return
    metadata_repo = sanitize_repo_url(resolved_repo)
    resolved_branch = branch or spec.default_branch
    branch_error = git_branch_error(resolved_branch)
    if branch_error:
        yield ("log", f"Error: {branch_error}")
        yield ("rc", 1)
        return
    safe_branch = sanitize_docker_tag(resolved_branch)
    main_tag = (
        sanitize_docker_tag(custom_tag)
        if custom_tag
        else f"{safe_branch}-{datetime.now().strftime('%Y%m%d')}"
    )

    yield ("log", f"Building {spec.backend} from source")
    yield ("log", f"Repository: {metadata_repo}")
    yield ("log", f"Branch: {resolved_branch}")
    for extra in extra_log_lines:
        yield ("log", extra)
    yield ("log", f"Tag: {spec.image_prefix}:{main_tag}")

    commit_hash = ""
    async for event in clone_or_update(spec, resolved_repo, resolved_branch):
        if event[0] == "commit":
            commit_hash = event[1]
        else:
            yield event
            if event[0] == "rc" and event[1] != 0:
                return

    dockerfile_path = spec.src_dir / spec.dockerfile_relpath if spec.dockerfile_relpath else None
    if dockerfile_path and not dockerfile_path.exists():
        yield ("log", f"Error: Dockerfile not found at {dockerfile_path}")
        yield ("rc", 1)
        return

    if pre_build is not None:
        ok, msg = await pre_build()
        if msg:
            yield ("log", msg)
        if not ok:
            yield ("rc", 1)
            return

    build_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    label_prefix = _label(spec)
    cmd: list[str] = ["docker", "build"]
    if dockerfile_path:
        cmd.extend(["-f", str(dockerfile_path)])
    if spec.target:
        cmd.extend(["--target", spec.target])
    for arg_k, arg_v in spec.base_build_args:
        cmd.extend(["--build-arg", f"{arg_k}={arg_v}"])
    for arg_k, arg_v in extra_build_args:
        cmd.extend(["--build-arg", f"{arg_k}={arg_v}"])
    cmd.extend([
        "--label", f"{label_prefix}.repo.url={metadata_repo}",
        "--label", f"{label_prefix}.repo.branch={resolved_branch}",
        "--label", f"{label_prefix}.commit.hash={commit_hash}",
        "--label", f"{label_prefix}.build.date={build_date}",
    ])
    for lk, lv in extra_labels:
        cmd.extend(["--label", f"{lk}={lv}"])
    cmd.extend([
        "-t", f"{spec.image_prefix}:{main_tag}",
        "-t", f"{spec.image_prefix}:{safe_branch}",
    ])
    cmd.append(str(spec.src_dir))

    build_env = os.environ.copy()
    build_env.setdefault("DOCKER_BUILDKIT", "1")
    async for event in _stream(cmd, env=build_env):
        if event[0] == "rc" and event[1] != 0:
            yield event
            return
        yield event


async def detect_local_gpu_caps() -> list[str]:
    """Return sorted unique compute capabilities from `nvidia-smi`."""
    rc, out = await _run(
        "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", timeout=10
    )
    if rc != 0 or not out.strip():
        return []
    return sorted({line.strip() for line in out.splitlines() if line.strip()})


def format_arch_torch(caps: list[str]) -> str:
    """vLLM / PyTorch convention: dotted, space-separated (e.g. '8.6 8.9')."""
    return " ".join(caps)


def format_arch_cmake(caps: list[str]) -> str:
    """Format capabilities as semicolon-separated CMake architecture numbers."""
    return ";".join(c.replace(".", "") for c in caps)


async def get_image_label(image_ref: str, label: str) -> str:
    error = image_reference_credential_error(image_ref)
    if error:
        raise RuntimeError(error)
    # Go templates require double-quoted string literals for label keys.
    rc, out = await _run(
        "docker",
        "inspect",
        image_ref,
        '--format={{index .Config.Labels "' + label + '"}}',
        timeout=20,
    )
    if rc != 0:
        lowered = out.lower()
        if "no such image" in lowered or "no such object" in lowered:
            return ""
        raise RuntimeError(out.strip() or f"docker image inspect {image_ref} failed")
    value = out.strip()
    return "" if value == "<no value>" else value


async def image_matches(
    spec: DevBuildSpec, image_tag: str, repo_url: str, branch: str
) -> bool:
    """Return True iff <prefix>:<image_tag> was built from this repo+branch."""
    label_prefix = _label(spec)
    image_ref = f"{spec.image_prefix}:{image_tag}"
    saved_repo = await get_image_label(image_ref, f"{label_prefix}.repo.url")
    saved_branch = await get_image_label(image_ref, f"{label_prefix}.repo.branch")
    if not saved_repo or not saved_branch:
        return False
    return saved_repo == sanitize_repo_url(repo_url) and saved_branch == branch


@dataclass
class DevImage:
    repository: str
    tag: str
    size: str
    created: str


async def list_local_dev_images(spec: DevBuildSpec) -> list[DevImage]:
    """List local dev images, raising when Docker cannot be queried."""
    error = image_reference_credential_error(spec.image_prefix)
    if error:
        raise RuntimeError(error)
    rc, out = await _run(
        "docker", "images", spec.image_prefix,
        "--format", "{{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}",
        timeout=10,
    )
    if rc != 0:
        raise RuntimeError(
            f"docker images {spec.image_prefix} failed or timed out: "
            f"{out.strip() or 'no output'}"
        )
    from tui.common.docker import parse_docker_image_rows

    return [
        DevImage(repository=repository, tag=tag, size=size, created=created)
        for repository, tag, size, created in parse_docker_image_rows(out)
        if tag != "<none>"
    ]


async def image_exists_locally(spec: DevBuildSpec, image_tag: str) -> bool:
    image_ref = f"{spec.image_prefix}:{image_tag}"
    error = image_reference_credential_error(image_ref)
    if error:
        raise RuntimeError(error)
    rc, out = await _run(
        "docker", "image", "inspect", image_ref, timeout=20
    )
    if rc == 0:
        return True
    lowered = out.lower()
    if "no such image" in lowered or "no such object" in lowered:
        return False
    raise RuntimeError(out.strip() or f"docker image inspect {image_ref} failed")
