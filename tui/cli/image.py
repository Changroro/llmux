"""Docker image inventory + dev image build (vLLM)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import typer

from tui.backends.llamacpp.backend import LLAMACPP_OFFICIAL_REPO
from tui.backends.vllm.backend_inspect import VLLM_OFFICIAL_REPO
from tui.cli._runtime import (
    JSON_OPTION,
    emit_json,
    emit_table,
    finish_stream,
    json_command,
    run_async,
    stream_async,
)
from tui.common import system_operations
from tui.common.dev_build import (
    image_reference_credential_error,
    image_tag_error,
    repo_url_error,
    sanitize_docker_tag,
)

app = typer.Typer(help="Docker image inventory + dev image build.", no_args_is_help=True)


@app.command("list")
def list_images(
    repo: str = typer.Option(
        "", "--repo", help="Image repo override; empty = backend default."
    ),
    backend: str = typer.Option(
        "vllm", "--backend", "-b", help="Backend whose images to list."
    ),
    dev: bool = typer.Option(False, "--dev", help="Also list local backend dev images."),
    remote: bool = typer.Option(
        False, "--remote",
        help="Query DockerHub for the latest stable + nightly tags (vllm/vllm-openai).",
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """List local (and optionally remote) docker images."""
    from tui.backends.vllm.backend_inspect import (
        get_dev_images,
        get_docker_images as get_vllm_images,
        get_dockerhub_nightly_date,
        get_dockerhub_release_version,
    )

    if backend not in _PULL_DEFAULTS:
        raise typer.BadParameter(
            f"unknown backend: {backend!r} (choose vllm or llamacpp)",
            param_hint="--backend",
        )
    if remote and backend != "vllm":
        raise typer.BadParameter(
            "remote stable/nightly inventory is available only for the vLLM DockerHub repository",
            param_hint="--remote",
        )
    repo = repo or _PULL_DEFAULTS[backend][0]
    credential_error = image_reference_credential_error(repo)
    if credential_error:
        raise typer.BadParameter(credential_error, param_hint="--repo")
    rows: list[dict] = []
    failures: list[str] = []

    async def _collect():
        if backend == "vllm":
            local_images = await get_vllm_images(repo=repo)
        else:
            from tui.backends.llamacpp.backend import get_docker_images as get_llama_images

            local_images = await get_llama_images(repo=repo)
        for img in local_images:
            rows.append({"source": "local", **asdict(img)})
        if dev:
            if backend == "vllm":
                dev_images = await get_dev_images()
            else:
                from tui.backends.llamacpp.backend_runtime import LLAMACPP_DEV_SPEC
                from tui.common.dev_build import list_local_dev_images

                dev_images = await list_local_dev_images(LLAMACPP_DEV_SPEC)
            for img in dev_images:
                rows.append({"source": "local-dev", **asdict(img)})
        if remote:
            release = await get_dockerhub_release_version()
            nightly = await get_dockerhub_nightly_date()
            if release == "unknown":
                failures.append("DockerHub stable-release lookup failed")
            else:
                rows.append({
                    "source": "remote", "repository": "vllm/vllm-openai",
                    "tag": release, "size": "", "created": "stable",
                })
            if nightly == "unknown":
                failures.append("DockerHub nightly-date lookup failed")
            else:
                rows.append({
                    "source": "remote", "repository": "vllm/vllm-openai",
                    "tag": "nightly", "size": "", "created": nightly,
                })

    try:
        run_async(_collect())
    except RuntimeError as exc:
        typer.echo(f"image query failed — {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_out:
        emit_json(rows)
    else:
        emit_table(rows, columns=["source", "repository", "tag", "size", "created"])
    if failures:
        for line in failures:
            typer.echo(line, err=True)
        raise typer.Exit(code=1)


_PULL_DEFAULTS = {
    "vllm": (VLLM_OFFICIAL_REPO, ""),
    "llamacpp": (LLAMACPP_OFFICIAL_REPO, "server-cuda"),
}


@app.command("pull")
@json_command
def pull_image(
    tag: Optional[str] = typer.Argument(
        None,
        help="Image tag. vLLM requires an explicit tag (e.g. 'v0.20.1', 'nightly'). "
             "llama.cpp defaults to 'server-cuda'.",
    ),
    backend: str = typer.Option(
        "vllm", "--backend", "-b",
        help="Selects per-backend default repo/tag (vllm → vllm/vllm-openai, "
             "llamacpp → ghcr.io/ggml-org/llama.cpp:server-cuda).",
    ),
    repo: str = typer.Option(
        "", "--repo",
        help="Repo override; empty = backend default.",
    ),
    json_out: bool = JSON_OPTION,
) -> None:
    """`docker pull <repo>:<tag>` — backend-aware defaults, raw docker output."""
    if backend not in _PULL_DEFAULTS:
        raise typer.BadParameter(
            f"unknown backend: {backend!r} (choose vllm or llamacpp)",
            param_hint="--backend",
        )
    default_repo, default_tag = _PULL_DEFAULTS[backend]
    repo = repo or default_repo
    tag = tag or default_tag
    if not tag:
        raise typer.BadParameter(
            "vLLM has no default tag — pass a TAG (e.g. 'v0.20.1' or 'nightly').",
            param_hint="TAG",
        )

    full = f"{repo}:{tag}"
    error = image_tag_error(full)
    if error:
        raise typer.BadParameter(error, param_hint="TAG")
    rc, lines = run_async(system_operations.pull_image(full))
    for line in lines:
        typer.echo(line, err=rc != 0)
    if rc != 0:
        raise typer.Exit(code=rc)
    return {"action": "pulled", "backend": backend, "image": full}


@app.command("remove")
@json_command
def remove_image(
    ref: str = typer.Argument(
        ...,
        help=(
            "Image reference to remove (e.g. 'vllm/vllm-openai:v0.10.0', "
            "'vllm-dev:main', 'llamacpp-dev:mtp-clean', or a 12-char id)."
        ),
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Pass --force to `docker rmi` (removes even if a stopped container references it).",
    ),
    json_out: bool = JSON_OPTION,
) -> None:
    """`docker rmi <ref>` — drop a local image without leaving the CLI."""
    credential_error = image_reference_credential_error(ref)
    if credential_error:
        raise typer.BadParameter(credential_error, param_hint="REF")
    rc, output = run_async(system_operations.remove_image(ref, force=force))
    if output:
        typer.echo(output, err=rc != 0)
    if rc != 0:
        raise typer.Exit(code=rc)
    return {"action": "removed", "image": ref}


@app.command("build-dev")
def build_dev(
    backend: str = typer.Option(
        "vllm", "--backend", "-b", help="Which backend to build for (vllm or llamacpp)."
    ),
    branch: str = typer.Option(
        "", "--branch", help="Source branch (default from .env.common)."
    ),
    repo_url: str = typer.Option(
        "", "--repo-url", help="Source repo URL (default from .env.common)."
    ),
    custom_tag: str = typer.Option(
        "", "--tag", "-t", help="Custom output tag (default = branch name)."
    ),
    official: bool = typer.Option(
        False, "--official",
        help="vLLM only: build with upstream Dockerfile defaults (skips local GPU-arch detection patches).",
    ),
    cuda_arch: str = typer.Option(
        "", "--cuda-arch",
        help="llamacpp only: override auto-detection. Pass CMake-format like '89' (Ada) or "
             "'86;89' (mixed). Empty = auto-detect via nvidia-smi.",
    ),
    multi_arch: bool = typer.Option(
        False, "--multi-arch",
        help="llamacpp only: disable GPU auto-detection and build for all archs (portable, slow).",
    ),
    json_out: bool = JSON_OPTION,
) -> None:
    """Build a backend development image from source."""
    if backend not in ("vllm", "llamacpp"):
        typer.echo(f"Error: unknown --backend {backend!r}. Use vllm or llamacpp.", err=True)
        raise typer.Exit(code=2)
    error = repo_url_error(repo_url)
    if error:
        raise typer.BadParameter(error, param_hint="--repo-url")
    sink: dict = {}

    if backend == "vllm":
        from tui.backends.vllm.backend_runtime import (
            _stream_build_dev_image,
            get_dev_build_defaults,
        )

        default_repo, default_branch = get_dev_build_defaults()
        branch = branch or default_branch
        repo_url = repo_url or default_repo
        error = repo_url_error(repo_url)
        if error:
            raise typer.BadParameter(error, param_hint="--repo-url")
        if cuda_arch or multi_arch:
            typer.echo("Warning: --cuda-arch/--multi-arch are llamacpp-only and ignored for vllm", err=True)
        rc = stream_async(
            _stream_build_dev_image(
                branch,
                repo_url=repo_url,
                custom_tag=custom_tag,
                use_official=official,
            ),
            json_out=json_out,
            sink=sink,
        )
    else:
        from tui.backends.llamacpp.backend_runtime import (
            _stream_build_dev_image,
            get_dev_build_defaults,
        )

        default_repo, default_branch = get_dev_build_defaults()
        branch = branch or default_branch
        repo_url = repo_url or default_repo
        error = repo_url_error(repo_url)
        if error:
            raise typer.BadParameter(error, param_hint="--repo-url")
        if official:
            typer.echo("Warning: --official is vllm-only and ignored for llamacpp", err=True)
        rc = stream_async(
            _stream_build_dev_image(
                branch,
                repo_url=repo_url,
                custom_tag=custom_tag,
                cuda_arch=cuda_arch,
                use_multi_arch=multi_arch,
            ),
            json_out=json_out,
            sink=sink,
        )
    from tui.backends.llamacpp.backend_runtime import LLAMACPP_DEV_SPEC
    from tui.backends.vllm.backend_runtime import VLLM_DEV_SPEC

    spec = VLLM_DEV_SPEC if backend == "vllm" else LLAMACPP_DEV_SPEC
    image = f"{spec.image_prefix}:{sanitize_docker_tag(custom_tag or branch)}"
    finish_stream(rc, json_out=json_out, record={"backend": backend, "image": image}, sink=sink)
