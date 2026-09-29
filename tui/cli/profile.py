"""Profile management commands.

Profiles live in `profiles.yaml` (gitignored). Each entry has a `backend`
(vllm | llamacpp). The CLI auto-detects backend from the name; pass --backend
to disambiguate when the same name exists in both.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Optional

import typer

from tui.backends.llamacpp.defaults import (
    QUICK_SETUP_CACHE_TYPE_K,
    QUICK_SETUP_CACHE_TYPE_V,
    QUICK_SETUP_CTX_SIZE,
    QUICK_SETUP_FLASH_ATTN,
    QUICK_SETUP_JINJA,
    QUICK_SETUP_N_GPU_LAYERS,
)
from tui.cli._runtime import (
    BACKENDS,
    detect_backend,
    emit_json,
    emit_table,
    run_async,
)
from tui.common import profile_store

app = typer.Typer(help="Profile CRUD + quick-setup.", no_args_is_help=True)


def _profile_to_row(p: profile_store.StoredProfile) -> dict:
    return {
        "backend": p.backend,
        "name": p.name,
        "port": p.port,
        "gpu_id": p.gpu_id,
        "config": p.config_name or p.name,
        "model": p.model_id or p.model_file or "",
    }



@app.command("list")
def list_profiles(
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help=f"Limit to one backend ({', '.join(BACKENDS)})."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """List all profiles across (or within) backends."""
    # profile_store.list_profiles raises a bare ValueError on an unknown
    # backend, which surfaced as a traceback instead of a usage error.
    if backend and backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")
    backends = [backend] if backend else list(BACKENDS)
    rows = []
    for bk in backends:
        rows.extend(_profile_to_row(p) for p in profile_store.list_profiles(bk))

    if json_out:
        emit_json(rows)
        return
    emit_table(rows, columns=["backend", "name", "port", "gpu_id", "config", "model"])


@app.command("show")
def show_profile(
    name: str = typer.Argument(..., help="Profile name."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Force backend; auto-detect if omitted."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of YAML-ish text."),
) -> None:
    """Show a profile's full record."""
    bk = detect_backend(name, override=backend)
    sp = profile_store.load_profile(name, bk)
    data = asdict(sp)
    data["container_name"] = sp.container_name or sp.name
    data["env_vars"] = {
        key: (
            {"set": True, "length": len(value)}
            if profile_store.sensitive_env_key(key)
            else value
        )
        for key, value in data["env_vars"].items()
    }
    if json_out:
        emit_json(data)
        return
    for k, v in data.items():
        print(f"{k}: {v}")



_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Profile name rule, shared by both backends: docker compose project names are
# lowercase-only.
_PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# GPU id rule. Multi-digit indices are allowed so hosts with 10+ GPUs are
# addressable.
_GPU_ID_RE = re.compile(r"^[0-9]+(,[0-9]+)*$")

PORT_MIN = 1024
PORT_MAX = 65535


def _validate_port(port: int, *, param_hint: str = "--port") -> None:
    if not PORT_MIN <= int(port) <= PORT_MAX:
        raise typer.BadParameter(
            f"port must be in {PORT_MIN}–{PORT_MAX}", param_hint=param_hint
        )


def _validate_gpu_id(gpu_id: str, *, param_hint: str = "--gpu-id") -> None:
    if not _GPU_ID_RE.match(gpu_id):
        raise typer.BadParameter(
            "GPU id must be digit(s) separated by single commas "
            "(e.g. '0', '0,1', '0,10').",
            param_hint=param_hint,
        )


def _validate_tensor_parallel(size: int, *, param_hint: str = "--tensor-parallel") -> None:
    if size < 1:
        raise typer.BadParameter(
            f"tensor-parallel size must be >= 1, got {size}", param_hint=param_hint
        )


def _validate_lora_limit(size: int, *, param_hint: str) -> None:
    if size < 1:
        raise typer.BadParameter(
            f"value must be a positive integer, got {size}", param_hint=param_hint
        )


def _validate_gpu_mem(gpu_mem: str, *, param_hint: str = "--gpu-mem") -> None:
    try:
        value = float(gpu_mem)
    except ValueError:
        raise typer.BadParameter(
            f"gpu-memory-utilization must be a number, got {gpu_mem!r}",
            param_hint=param_hint,
        ) from None
    if not 0.0 < value <= 1.0:
        raise typer.BadParameter(
            "gpu-memory-utilization must be between 0.0 and 1.0 (exclusive of 0.0).",
            param_hint=param_hint,
        )


def _reject_cross_backend_options(
    backend: str,
    *,
    vllm_only: dict[str, bool],
    llamacpp_only: dict[str, bool],
) -> None:
    if backend == "llamacpp":
        offenders = [flag for flag, given in vllm_only.items() if given]
        owner = "vLLM"
    else:
        offenders = [flag for flag, given in llamacpp_only.items() if given]
        owner = "llama.cpp"
    if not offenders:
        return
    verb = "is" if len(offenders) == 1 else "are"
    raise typer.BadParameter(
        f"{', '.join(offenders)} {verb} {owner}-only and not supported by "
        f"backend '{backend}'.",
        param_hint=offenders[0],
    )


def _require_config_exists(
    backend: str, name: str, *, param_hint: str = "--copy-from"
) -> None:
    if backend == "vllm":
        from tui.backends.vllm.backend_common import CONFIG_DIR
    else:
        from tui.backends.llamacpp.backend import CONFIG_DIR

    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise typer.BadParameter(
            f"config not found: {path}", param_hint=param_hint
        )


def _reject_example_config(config_name: str, *, from_profile_name: bool) -> None:
    if config_name != "example":
        return
    if from_profile_name:
        raise typer.BadParameter(
            "'example' is the tracked template config, and a profile named "
            "'example' links to it by default. Pick a different profile name, "
            "or pass --config with another name.",
            param_hint="NAME",
        )
    raise typer.BadParameter(
        "'example' is the tracked template config and may not be linked; "
        "pick a different config name.",
        param_hint="--config",
    )


def _require_linked_config_exists(backend: str, profile_name: str, config_name: str) -> None:
    if config_name == profile_name:
        return
    if backend == "vllm":
        from tui.backends.vllm.backend_common import CONFIG_DIR
    else:
        from tui.backends.llamacpp.backend import CONFIG_DIR

    path = CONFIG_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise typer.BadParameter(
            f"config not found: {path}. Only a config named after the profile "
            f"('{profile_name}') may be missing — that one is auto-created at start.",
            param_hint="--config",
        )


def _validate_profile_name(name: str, backend: str, *, param_hint: str = "NAME") -> None:
    if not _PROFILE_NAME_RE.match(name):
        raise typer.BadParameter(
            f"Profile names must match {_PROFILE_NAME_RE.pattern} "
            "(lowercase: start with [a-z0-9], then [a-z0-9_-] only). "
            "docker compose project names are lowercase-only, so this rule "
            "is shared by both backends.",
            param_hint=param_hint,
        )


def _parse_set_kv(items: list[str], *, backend: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise typer.BadParameter(
                "--set value must be KEY=VALUE", param_hint="--set"
            )
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            raise typer.BadParameter("--set KEY may not be empty", param_hint="--set")
        if not _ENV_KEY_RE.match(key):
            raise typer.BadParameter(
                f"--set KEY {key!r} must match {_ENV_KEY_RE.pattern} "
                "(env-var rules: leading letter or underscore, then alphanumerics/underscores)",
                param_hint="--set",
            )
        if backend and profile_store.is_protected_profile_env_key(key, backend):
            raise typer.BadParameter(
                f"--set KEY {key!r} is reserved by the {backend} backend; "
                f"use the dedicated profile field (e.g. --port, --gpu-id) instead.",
                param_hint="--set",
            )
        reason = profile_store.env_value_rejection(value)
        if reason:
            raise typer.BadParameter(
                f"--set value for {key!r} contains a {reason}, which docker "
                "compose's .env parser cannot read",
                param_hint="--set",
            )
        out[key] = value
    return out


@app.command("new")
def new_profile(
    name: str = typer.Argument(..., help="Profile name (alphanumeric, dash, underscore)."),
    backend: str = typer.Option(
        "vllm", "--backend", "-b", help=f"Backend ({', '.join(BACKENDS)})."
    ),
    port: int = typer.Option(0, "--port", "-p", help="Host port (0 = backend default)."),
    gpu_id: str = typer.Option("0", "--gpu-id", "-g", help="GPU id(s), comma-separated."),
    tensor_parallel: int = typer.Option(
        0, "--tensor-parallel",
        help="vLLM only: tensor_parallel_size (0 = derive from the --gpu-id count).",
    ),
    model: str = typer.Option("", "--model", "-m", help="Hugging Face model id (vLLM)."),
    config_name: str = typer.Option(
        "", "--config", "-c", help="Linked config name (defaults to profile name)."
    ),
    container_name: str = typer.Option(
        "", "--container", help="Container name (defaults to profile name)."
    ),
    enable_lora: bool = typer.Option(False, "--lora/--no-lora", help="vLLM only: enable LoRA."),
    max_loras: int = typer.Option(
        0, "--max-loras", help="vLLM only: maximum simultaneously loaded LoRAs."
    ),
    max_lora_rank: int = typer.Option(
        0, "--max-lora-rank", help="vLLM only: maximum supported LoRA rank."
    ),
    lora_modules: str = typer.Option(
        "", "--lora-modules", help="vLLM only: comma-separated name=path adapters."
    ),
    extra_pip: str = typer.Option(
        "", "--extra-pip", help="vLLM only: extra pip packages installed before serve."
    ),
    set_env: list[str] = typer.Option(
        [], "--set", help="Repeatable: KEY=VALUE entries appended to env_vars."
    ),
    # llama.cpp specific
    model_file: str = typer.Option("", "--model-file", help="llama.cpp: GGUF filename."),
    hf_repo: str = typer.Option("", "--hf-repo", help="llama.cpp: Hugging Face repo for download."),
    hf_file: str = typer.Option("", "--hf-file", help="llama.cpp: HF file for download."),
) -> None:
    """Create a new profile entry in profiles.yaml."""
    if backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")
    _validate_profile_name(name, backend)
    # `new` has no None sentinel, so "was it given?" is "does it differ from the
    # default?". (--no-lora is therefore indistinguishable from omitting it.)
    _reject_cross_backend_options(
        backend,
        vllm_only={
            "--model": bool(model),
            "--extra-pip": bool(extra_pip),
            "--lora": enable_lora,
            "--max-loras": bool(max_loras),
            "--max-lora-rank": bool(max_lora_rank),
            "--lora-modules": bool(lora_modules),
            "--tensor-parallel": bool(tensor_parallel),
        },
        llamacpp_only={
            "--model-file": bool(model_file),
            "--hf-repo": bool(hf_repo),
            "--hf-file": bool(hf_file),
        },
    )
    # The container name becomes a docker object name — same lowercase rule the
    # TUI form enforces.
    if container_name:
        _validate_profile_name(container_name, backend, param_hint="--container")
    # port 0 is the documented "use the backend default" sentinel; any other
    # out-of-range value is a mistake.
    if port:
        _validate_port(port)
    if gpu_id:
        _validate_gpu_id(gpu_id)
    if tensor_parallel:
        _validate_tensor_parallel(tensor_parallel)
    if max_loras:
        _validate_lora_limit(max_loras, param_hint="--max-loras")
    if max_lora_rank:
        _validate_lora_limit(max_lora_rank, param_hint="--max-lora-rank")
    owner = profile_store.find_name_owner(name)
    if owner is not None:
        raise typer.BadParameter(
            f"profile '{name}' already exists (backend '{owner}'). Profile names "
            "must be unique across both backends.",
            param_hint="NAME",
        )
    _reject_example_config(config_name or name, from_profile_name=not config_name)
    if config_name:
        _require_linked_config_exists(backend, name, config_name)

    defaults = profile_store.effective_defaults(backend)
    sp = profile_store.StoredProfile(
        name=name,
        backend=backend,
        container_name=container_name or name,
        port=port or int(defaults["port"]),
        gpu_id=gpu_id or str(defaults["gpu_id"]),
        config_name=config_name or name,
        tensor_parallel_size=tensor_parallel or (len(gpu_id.split(",")) if gpu_id else 1),
        model_id=model,
        enable_lora=enable_lora,
        max_loras=max_loras or None,
        max_lora_rank=max_lora_rank or None,
        lora_modules=lora_modules,
        extra_pip_packages=extra_pip,
        env_vars=_parse_set_kv(set_env, backend=backend),
        model_file=model_file,
        hf_repo=hf_repo,
        hf_file=hf_file,
    )
    profile_store.create_profile(sp)
    print(f"Created profile '{name}' (backend={backend})")


@app.command("edit")
def edit_profile(
    name: str = typer.Argument(..., help="Profile name."),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    port: Optional[int] = typer.Option(
        None, "--port", "-p", help="Host port (0 = backend default)."
    ),
    gpu_id: Optional[str] = typer.Option(None, "--gpu-id", "-g"),
    tensor_parallel: Optional[int] = typer.Option(
        None, "--tensor-parallel",
        help="vLLM only: tensor_parallel_size. Overrides the value --gpu-id would derive.",
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    config_name: Optional[str] = typer.Option(None, "--config", "-c"),
    container_name: Optional[str] = typer.Option(None, "--container"),
    enable_lora: Optional[bool] = typer.Option(None, "--lora/--no-lora"),
    max_loras: Optional[int] = typer.Option(
        None, "--max-loras", help="vLLM only: positive integer; 0 clears it."
    ),
    max_lora_rank: Optional[int] = typer.Option(
        None, "--max-lora-rank", help="vLLM only: positive integer; 0 clears it."
    ),
    lora_modules: Optional[str] = typer.Option(
        None, "--lora-modules", help="vLLM only: comma-separated name=path adapters."
    ),
    extra_pip: Optional[str] = typer.Option(None, "--extra-pip"),
    set_env: list[str] = typer.Option(
        [], "--set", help="Repeatable: KEY=VALUE entries to add/override in env_vars."
    ),
    unset_env: list[str] = typer.Option(
        [], "--unset", help="Repeatable: KEY to remove from env_vars."
    ),
    model_file: Optional[str] = typer.Option(None, "--model-file"),
    hf_repo: Optional[str] = typer.Option(None, "--hf-repo"),
    hf_file: Optional[str] = typer.Option(None, "--hf-file"),
    image_tag: Optional[str] = typer.Option(
        None, "--image-tag",
        help="Docker image override (e.g. 'llamacpp-dev:mtp_main'). Pass empty string to clear.",
    ),
) -> None:
    """Edit fields of an existing profile (only specified options change)."""
    bk = detect_backend(name, override=backend)
    _validate_profile_name(name, bk)
    _reject_cross_backend_options(
        bk,
        vllm_only={
            "--model": model is not None,
            "--extra-pip": extra_pip is not None,
            "--lora/--no-lora": enable_lora is not None,
            "--max-loras": max_loras is not None,
            "--max-lora-rank": max_lora_rank is not None,
            "--lora-modules": lora_modules is not None,
            "--tensor-parallel": tensor_parallel is not None,
        },
        llamacpp_only={
            "--model-file": model_file is not None,
            "--hf-repo": hf_repo is not None,
            "--hf-file": hf_file is not None,
        },
    )
    env_updates = _parse_set_kv(set_env, backend=bk)
    if image_tag is not None:
        from tui.common.dev_build import image_tag_error

        err = image_tag_error(image_tag)
        if err:
            raise typer.BadParameter(err, param_hint="--image-tag")
    if max_loras is not None and max_loras < 0:
        _validate_lora_limit(max_loras, param_hint="--max-loras")
    if max_lora_rank is not None and max_lora_rank < 0:
        _validate_lora_limit(max_lora_rank, param_hint="--max-lora-rank")

    def apply_updates(sp: profile_store.StoredProfile) -> None:
        if port is not None:
            if port:
                _validate_port(port)
                sp.port = port
            else:
                sp.port = int(profile_store.effective_defaults(bk)["port"])
        if gpu_id is not None:
            _validate_gpu_id(gpu_id)
            sp.gpu_id = gpu_id
            if bk == "vllm":
                derived = len(gpu_id.split(",")) if gpu_id else 1
                if tensor_parallel is None and derived != sp.tensor_parallel_size:
                    print(
                        f"tensor_parallel_size adjusted to {derived} to match --gpu-id "
                        "(pass --tensor-parallel to override)"
                    )
                sp.tensor_parallel_size = derived
        if tensor_parallel is not None:
            _validate_tensor_parallel(tensor_parallel)
            sp.tensor_parallel_size = tensor_parallel
        if model is not None:
            sp.model_id = model
        if config_name is not None:
            _reject_example_config(
                config_name or name,
                from_profile_name=not config_name,
            )
            if config_name:
                _require_linked_config_exists(bk, name, config_name)
            sp.config_name = config_name
        if container_name is not None:
            if container_name:
                _validate_profile_name(container_name, bk, param_hint="--container")
            sp.container_name = container_name
        if enable_lora is not None:
            sp.enable_lora = enable_lora
        if max_loras is not None:
            sp.max_loras = max_loras or None
        if max_lora_rank is not None:
            sp.max_lora_rank = max_lora_rank or None
        if lora_modules is not None:
            sp.lora_modules = lora_modules
        if extra_pip is not None:
            sp.extra_pip_packages = extra_pip
        if model_file is not None:
            sp.model_file = model_file
        if hf_repo is not None:
            sp.hf_repo = hf_repo
        if hf_file is not None:
            sp.hf_file = hf_file
        if image_tag is not None:
            sp.image_tag = image_tag
        sp.env_vars.update(env_updates)
        for key in unset_env:
            sp.env_vars.pop(key, None)

    profile_store.update_profile(name, bk, apply_updates)
    print(f"Updated profile '{name}' (backend={bk})")


def _require_stopped(sp: profile_store.StoredProfile, *, action: str) -> None:
    from tui.common import docker as common_docker

    container = sp.container_name or sp.name
    try:
        running = run_async(common_docker.running_container_names())
    except Exception as exc:
        raise typer.BadParameter(
            f"could not enumerate running containers via `docker ps` ({exc}); "
            f"refusing to {action} without confirming the container is stopped.",
            param_hint="NAME",
        ) from exc
    if container in running:
        raise typer.BadParameter(
            f"container '{container}' is running; stop it first "
            f"(`llmux down {sp.name}`) before you {action}.",
            param_hint="NAME",
        )


@app.command("rename")
def rename_profile(
    old: str = typer.Argument(..., help="Existing profile name."),
    new: str = typer.Argument(..., help="New profile name."),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
) -> None:
    """Rename a profile. Requires its container to be stopped."""
    bk = detect_backend(old, override=backend)
    _validate_profile_name(new, bk, param_hint="NEW")
    sp = profile_store.load_profile(old, bk)
    if sp is None:
        raise typer.BadParameter(f"profile '{old}' not found in backend '{bk}'", param_hint="OLD")
    _require_stopped(sp, action="rename the profile")
    try:
        renamed = profile_store.rename_profile(old, new, bk)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="NEW") from exc
    print(f"Renamed profile '{old}' → '{new}' (backend={bk})")
    print(f"  container: {renamed.container_name or renamed.name}")
    print(f"  config:    {renamed.config_name or renamed.name}")


@app.command("clone")
def clone_profile(
    src: str = typer.Argument(..., help="Profile to copy from."),
    dst: str = typer.Argument(..., help="New profile name."),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
) -> None:
    """Clone SRC to a new profile DST (same backend, same linked config)."""
    bk = detect_backend(src, override=backend)
    _validate_profile_name(dst, bk, param_hint="DST")
    if profile_store.load_profile(src, bk) is None:
        raise typer.BadParameter(f"profile '{src}' not found in backend '{bk}'", param_hint="SRC")
    owner = profile_store.find_name_owner(dst)
    if owner is not None:
        raise typer.BadParameter(
            f"profile '{dst}' already exists in backend '{owner}'; profile names "
            "must be unique across both backends.",
            param_hint="DST",
        )
    try:
        sp = profile_store.clone_profile(src, dst, bk)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="DST") from exc
    print(f"Cloned profile '{src}' → '{dst}' (backend={bk})")
    print(f"  container: {sp.container_name or sp.name}")
    print(f"  config:    {sp.config_name or sp.name}")
    if sp.port:
        print(f"  port:      {sp.port}  (same as '{src}' — change it before running both)")


@app.command("delete")
def delete_profile(
    name: str = typer.Argument(..., help="Profile name."),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    with_config: bool = typer.Option(
        False, "--with-config", help="Also delete the linked config YAML."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete a profile (and optionally its linked config)."""
    bk = detect_backend(name, override=backend)
    sp = profile_store.load_profile(name, bk)
    if sp is None:
        raise typer.BadParameter(
            f"profile '{name}' not found in backend '{bk}'", param_hint="NAME"
        )
    _require_stopped(sp, action="delete the profile")
    if not yes:
        suffix = " AND its config YAML" if with_config else ""
        confirm = typer.confirm(f"Delete profile '{name}' (backend={bk}){suffix}?")
        if not confirm:
            raise typer.Exit(code=0)

    if with_config:
        if bk == "vllm":
            from tui.backends.vllm.backend_storage import delete_profile as _d

            _d(name, delete_config=True)
        else:
            from tui.backends.llamacpp.backend import delete_profile as _d

            _d(name, delete_config_too=True)
    else:
        profile_store.delete_profile(name, bk)
    print(f"Deleted profile '{name}' (backend={bk})")



_LLAMACPP_REPO_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")


def _normalize_hf_repo(raw: str) -> str:
    """Strip huggingface.co URL wrapping → bare 'org/name' (mirrors the TUI)."""
    s = raw.strip().rstrip("/")
    if not s:
        return ""
    if "huggingface.co/" in s:
        s = s.split("huggingface.co/", 1)[1]
        if s.startswith("api/models/"):
            s = s[len("api/models/") :]
    parts = s.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return s


@app.command("quick-setup")
def quick_setup(
    model: Optional[str] = typer.Argument(
        None,
        help="HF model id (vLLM only — required for --backend vllm, e.g. Qwen/Qwen3-8B). "
             "Ignored for --backend llamacpp; use --hf-repo + --hf-file there.",
    ),
    backend: str = typer.Option(
        "vllm", "--backend", "-b", help=f"Backend ({', '.join(BACKENDS)})."
    ),
    name: str = typer.Option(
        "", "--name", "-n", help="Profile name (auto-derived from model/repo if omitted)."
    ),
    port: int = typer.Option(
        0, "--port", "-p",
        help="Host port (0 = backend default: vLLM 8000, llama.cpp 8080).",
    ),
    gpu_id: str = typer.Option("0", "--gpu-id", "-g", help="GPU id(s), comma-separated."),
    # vLLM-only
    gpu_memory_utilization: str = typer.Option(
        "0.9", "--gpu-mem", help="vLLM only: gpu-memory-utilization (0.0–1.0)."
    ),
    enable_lora: bool = typer.Option(False, "--lora/--no-lora", help="vLLM only."),
    recipe: bool = typer.Option(
        False, "--recipe", help="vLLM only: apply the official recipe for MODEL."
    ),
    recipe_from: str = typer.Option(
        "", "--recipe-from", help="vLLM only: borrow the recipe of another model."
    ),
    recipe_variant: str = typer.Option(
        "", "--variant", help="vLLM only: recipe precision variant."
    ),
    recipe_feature: list[str] = typer.Option(
        [], "--feature", help="vLLM only: repeatable recipe feature."
    ),
    # llama.cpp-only
    hf_repo: str = typer.Option(
        "", "--hf-repo",
        help="llama.cpp only: HuggingFace repo (e.g. unsloth/Qwen3-30B-A3B-GGUF).",
    ),
    hf_file: str = typer.Option(
        "", "--hf-file",
        help="llama.cpp only: GGUF filename inside the repo (validated against HF API when reachable).",
    ),
    ctx_size: str = typer.Option(
        QUICK_SETUP_CTX_SIZE, "--ctx-size",
        help="llama.cpp only: --ctx-size (token context). Empty = backend default.",
    ),
    n_gpu_layers: str = typer.Option(
        QUICK_SETUP_N_GPU_LAYERS, "--n-gpu-layers",
        help=(
            f"llama.cpp only: --n-gpu-layers ({QUICK_SETUP_N_GPU_LAYERS} = all). "
            "Empty = backend default."
        ),
    ),
    batch_size: str = typer.Option(
        "", "--batch-size",
        help="llama.cpp only: --batch-size. Empty = backend default.",
    ),
    cache_type_k: str = typer.Option(
        QUICK_SETUP_CACHE_TYPE_K, "--cache-type-k",
        help="llama.cpp only: --cache-type-k (f16/bf16/q8_0/q4_0). Empty = backend default.",
    ),
    cache_type_v: str = typer.Option(
        QUICK_SETUP_CACHE_TYPE_V, "--cache-type-v",
        help="llama.cpp only: --cache-type-v (f16/bf16/q8_0/q4_0). Empty = backend default.",
    ),
    flash_attn: bool = typer.Option(
        QUICK_SETUP_FLASH_ATTN,
        "--flash-attn/--no-flash-attn",
        help="llama.cpp only: --flash-attn.",
    ),
    jinja: bool = typer.Option(
        QUICK_SETUP_JINJA, "--jinja/--no-jinja",
        help="llama.cpp only: --jinja (required for /v1/chat/completions).",
    ),
    override_tensors: str = typer.Option(
        "", "--override-tensors",
        help="llama.cpp only: --override-tensors regex (MoE expert offload).",
    ),
    # Shared (now both backends)
    copy_config_from: str = typer.Option(
        "", "--copy-from",
        help="Copy extra_params (vLLM) / all params (llama.cpp) from an existing config name.",
    ),
) -> None:
    """Create a profile + config in one step (mirrors TUI Quick Setup, both backends)."""
    if backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")

    # Same "given == differs from default" heuristic as `new` (see the helper's
    # docstring for its one blind spot: passing a flag's default explicitly).
    _reject_cross_backend_options(
        backend,
        vllm_only={
            "--gpu-mem": gpu_memory_utilization != "0.9",
            "--lora": enable_lora,
            "--recipe": recipe,
            "--recipe-from": bool(recipe_from),
            "--variant": bool(recipe_variant),
            "--feature": bool(recipe_feature),
        },
        llamacpp_only={
            "--hf-repo": bool(hf_repo),
            "--hf-file": bool(hf_file),
            "--ctx-size": ctx_size != QUICK_SETUP_CTX_SIZE,
            "--n-gpu-layers": n_gpu_layers != QUICK_SETUP_N_GPU_LAYERS,
            "--batch-size": bool(batch_size),
            "--cache-type-k": cache_type_k != QUICK_SETUP_CACHE_TYPE_K,
            "--cache-type-v": cache_type_v != QUICK_SETUP_CACHE_TYPE_V,
            "--no-flash-attn": flash_attn != QUICK_SETUP_FLASH_ATTN,
            "--no-jinja": jinja != QUICK_SETUP_JINJA,
            "--override-tensors": bool(override_tensors),
        },
    )

    if backend == "vllm":
        _validate_gpu_mem(gpu_memory_utilization)
        _quick_setup_vllm(
            model=model,
            name=name,
            # `0` = backend default — resolved through effective_defaults() so a
            # user `defaults:` override in profiles.yaml is honored, same as
            # `profile new`.
            port=port or int(profile_store.effective_defaults("vllm")["port"]),
            gpu_id=gpu_id,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=enable_lora,
            copy_config_from=copy_config_from,
            use_recipe=recipe or bool(recipe_from or recipe_variant or recipe_feature),
            recipe_from=recipe_from,
            recipe_variant=recipe_variant,
            recipe_features=recipe_feature,
        )
        return

    # llama.cpp path: warn (don't fail) if the positional MODEL was supplied —
    # it's meaningless here (HF model id makes no sense for a GGUF profile) and
    # silently dropping it is a real footgun in scripts. Emit clearly on stderr
    # so a caller can see they probably meant `--name MODEL` (or know the
    # auto-derive from --hf-repo is what they'll actually get).
    if model and not name:
        typer.echo(
            f"Warning: positional MODEL '{model}' is ignored for --backend "
            "llamacpp; the profile name will be auto-derived from --hf-repo. "
            f"Pass --name {model!r} explicitly if you meant that as the name.",
            err=True,
        )
    elif model and name:
        typer.echo(
            f"Warning: positional MODEL '{model}' is ignored for --backend "
            f"llamacpp (using --name '{name}').",
            err=True,
        )

    _quick_setup_llamacpp(
        hf_repo=hf_repo,
        hf_file=hf_file,
        name=name,
        port=port or int(profile_store.effective_defaults("llamacpp")["port"]),
        gpu_id=gpu_id,
        ctx_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
        batch_size=batch_size,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        flash_attn=flash_attn,
        jinja=jinja,
        override_tensors=override_tensors,
        copy_config_from=copy_config_from,
    )


def _quick_setup_vllm(
    *,
    model: Optional[str],
    name: str,
    port: int,
    gpu_id: str,
    gpu_memory_utilization: str,
    enable_lora: bool,
    copy_config_from: str,
    use_recipe: bool,
    recipe_from: str,
    recipe_variant: str,
    recipe_features: list[str],
) -> None:
    if not model:
        raise typer.BadParameter(
            "vllm quick-setup requires a HF model id (positional MODEL).",
            param_hint="MODEL",
        )

    _validate_port(port)
    _validate_gpu_id(gpu_id)

    if not name:
        tail = model.rsplit("/", 1)[-1]
        derived = re.sub(r"[^a-z0-9-]", "-", tail.lower()).strip("-")
        if not derived:
            raise typer.BadParameter(
                "could not derive name from model; pass --name explicitly", param_hint="--name"
            )
        name = derived

    if not _PROFILE_NAME_RE.match(name):
        raise typer.BadParameter("derived name has invalid characters; pass --name explicitly")

    from tui.backends.vllm.backend_storage import (
        load_config,
        save_config,
    )
    from tui.backends.vllm.backend_common import CONFIG_DIR, Config

    if use_recipe and copy_config_from:
        raise typer.BadParameter(
            "--copy-from cannot be combined with recipe options",
            param_hint="--copy-from",
        )

    extra_params: dict = {}
    disabled_params: dict = {}
    if use_recipe:
        from tui.cli.config import _pick_recipe_variant
        from tui.common.recipes import RecipeUnavailable, build_config, fetch_recipe

        source_model = recipe_from.strip() or model
        try:
            fetched = run_async(fetch_recipe(source_model))
        except RecipeUnavailable as exc:
            typer.echo(f"could not reach the recipe index — {exc}", err=True)
            raise typer.Exit(code=1) from exc
        if fetched is None:
            raise typer.BadParameter(
                f"no vLLM recipe found for {source_model}",
                param_hint="--recipe-from" if recipe_from else "MODEL",
            )
        chosen = _pick_recipe_variant(fetched, recipe_variant)
        recipe_model, extra_params = build_config(
            fetched,
            chosen,
            list(recipe_features),
        )
        if recipe_from:
            recipe_model = model
        model = recipe_model
    elif copy_config_from:
        # load_config() returns an empty Config for a missing file, so without
        # this check a typo'd --copy-from silently produced an empty config.
        _require_config_exists("vllm", copy_config_from)
        src = load_config(copy_config_from)
        extra_params = dict(src.extra_params)
        disabled_params = dict(src.disabled_params)

    with profile_store.quick_setup_transaction(name, "vllm", CONFIG_DIR) as final_name:
        config = Config(
            name=final_name,
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            extra_params=extra_params,
            disabled_params=disabled_params,
        )
        save_config(config, template=src.path.read_text() if copy_config_from else None)
        profile_store.create_profile(
            profile_store.StoredProfile(
                name=final_name,
                backend="vllm",
                container_name=final_name,
                port=port,
                gpu_id=gpu_id,
                tensor_parallel_size=len(gpu_id.split(",")) if gpu_id else 1,
                config_name=final_name,
                model_id=model,
                enable_lora=enable_lora,
            )
        )
    print(f"Created profile + config: {final_name}")


def _quick_setup_llamacpp(
    *,
    hf_repo: str,
    hf_file: str,
    name: str,
    port: int,
    gpu_id: str,
    ctx_size: str,
    n_gpu_layers: str,
    batch_size: str,
    cache_type_k: str,
    cache_type_v: str,
    flash_attn: bool,
    jinja: bool,
    override_tensors: str,
    copy_config_from: str,
) -> None:
    repo = _normalize_hf_repo(hf_repo)
    if not repo or not _LLAMACPP_REPO_RE.match(repo):
        raise typer.BadParameter(
            "llama.cpp quick-setup requires a valid HF repo (org/name).",
            param_hint="--hf-repo",
        )
    if not hf_file:
        raise typer.BadParameter(
            "llama.cpp quick-setup requires --hf-file (GGUF filename inside the repo).",
            param_hint="--hf-file",
        )
    _validate_port(port)
    _validate_gpu_id(gpu_id)

    from tui.backends.llamacpp.backend import (
        CONFIG_DIR as llamacpp_config_dir,
        Config as LcppConfig,
        HfListingUnavailable,
        list_hf_repo_files,
        load_config as l_load_config,
        save_config as l_save_config,
        validate_name as l_validate_name,
    )

    # Validate the GGUF filename against the live repo listing when reachable.
    # A failed lookup does not hard-fail (a headless caller may already know
    # what's in the repo) but it must say the check was skipped — otherwise a
    # typo'd --hf-file sails through and only surfaces at download time.
    listing_available = True
    try:
        files = run_async(list_hf_repo_files(repo))
    except HfListingUnavailable as exc:
        typer.echo(
            f"Warning: could not list {repo} ({exc}) — skipping --hf-file validation.",
            err=True,
        )
        files = []
        listing_available = False
    gguf_files = [
        str(f.get("path", ""))
        for f in files
        if isinstance(f, dict)
        and f.get("type") == "file"
        and str(f.get("path", "")).lower().endswith(".gguf")
    ]
    if listing_available and not gguf_files:
        raise typer.BadParameter(
            f"{repo} contains no GGUF files.", param_hint="--hf-file"
        )
    if gguf_files and hf_file not in gguf_files:
        available = "\n  ".join(gguf_files[:20])
        more = f"\n  ... ({len(gguf_files) - 20} more)" if len(gguf_files) > 20 else ""
        raise typer.BadParameter(
            f"'{hf_file}' not found in {repo}. Available GGUF files:\n  {available}{more}",
            param_hint="--hf-file",
        )

    # Auto-derive name: <repo-tail-without-GGUF>, lowercased + safe chars.
    if not name:
        base = repo.rsplit("/", 1)[-1]
        base = re.sub(r"[-_]?GGUF$", "", base, flags=re.I)
        name = re.sub(r"[^a-z0-9_-]+", "-", base.lower()).strip("-")
    if not name:
        raise typer.BadParameter(
            "could not derive name from repo; pass --name explicitly", param_hint="--name"
        )
    if not l_validate_name(name):
        raise typer.BadParameter(
            "llama.cpp profile names must be lowercase: start with [a-z0-9], "
            "then [a-z0-9_-] only.",
            param_hint="--name",
        )

    params: dict = {}
    disabled_params: dict = {}
    if copy_config_from:
        # See _quick_setup_vllm — load_config() falls back to an empty Config.
        _require_config_exists("llamacpp", copy_config_from)
        src = l_load_config(copy_config_from)
        params.update(src.params)
        disabled_params = dict(src.disabled_params)
    params["model-file"] = hf_file

    def _set_int(key: str, raw: str, *, param_hint: str) -> None:
        if not raw:
            params.pop(key, None)
            return
        try:
            params[key] = int(raw)
        except ValueError:
            # Fail loudly here rather than storing a non-numeric string that
            # only breaks at llama-server start — port/gpu-id in this same
            # command validate eagerly, so these should too.
            raise typer.BadParameter(
                f"{param_hint} must be an integer, got {raw!r}",
                param_hint=param_hint,
            ) from None

    _set_int("ctx-size", ctx_size, param_hint="--ctx-size")
    _set_int("n-gpu-layers", n_gpu_layers, param_hint="--n-gpu-layers")
    _set_int("batch-size", batch_size, param_hint="--batch-size")
    if cache_type_k:
        params["cache-type-k"] = cache_type_k
    else:
        params.pop("cache-type-k", None)
    if cache_type_v:
        params["cache-type-v"] = cache_type_v
    else:
        params.pop("cache-type-v", None)

    if flash_attn:
        params["flash-attn"] = True
    else:
        params.pop("flash-attn", None)
    if jinja:
        params["jinja"] = True
    else:
        params.pop("jinja", None)
    if override_tensors:
        params["override-tensors"] = [override_tensors]
    else:
        params.pop("override-tensors", None)

    with profile_store.quick_setup_transaction(
        name,
        "llamacpp",
        llamacpp_config_dir,
    ) as final_name:
        params.setdefault("alias", final_name)
        config = LcppConfig(
            name=final_name,
            params=params,
            disabled_params=disabled_params,
        )
        l_save_config(config, template=src.path.read_text() if copy_config_from else None)
        profile_store.create_profile(
            profile_store.StoredProfile(
                name=final_name,
                backend="llamacpp",
                container_name=final_name,
                port=int(port),
                gpu_id=gpu_id,
                config_name=final_name,
                model_file=hf_file,
                hf_repo=repo,
                hf_file=hf_file,
            )
        )
    print(f"Created profile + config: {final_name}")
