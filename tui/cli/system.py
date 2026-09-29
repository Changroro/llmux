from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import typer

from tui.cli._runtime import emit_json, emit_table, run_async

app = typer.Typer(help="System info & env validation.", no_args_is_help=True)


def _nearest_existing_path(path: Path) -> Path:
    candidate = path
    while True:
        try:
            candidate.stat()
            return candidate
        except FileNotFoundError:
            parent = candidate.parent
            if parent == candidate:
                raise
            candidate = parent


@app.command("gpu")
def gpu(
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Print nvidia-smi summary (one row per GPU)."""
    from tui.common.docker import get_gpu_info

    try:
        gpus = run_async(get_gpu_info())
    except RuntimeError as exc:
        if json_out:
            emit_json({"status": "error", "error": str(exc)})
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    rows = [asdict(g) for g in gpus]
    if json_out:
        emit_json(rows)
        return
    if not rows:
        print("(no GPUs detected)")
        return
    emit_table(
        rows,
        columns=["index", "name", "memory_used", "memory_total", "utilization", "temperature"],
    )


@app.command("mem-estimate")
def mem_estimate(
    model_id: str = typer.Argument(..., help="Hugging Face model id, e.g. Qwen/Qwen3-8B."),
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit a JSON record with parsed est_gb + per-GPU ratios.",
    ),
) -> None:
    """Estimate a model's VRAM footprint and per-GPU fit."""
    import re as _re

    from tui.common.docker import get_gpu_info, parse_gpu_reading
    from tui.common.mem import estimate_model_memory

    async def _gather() -> tuple[str, list]:
        estimate = await estimate_model_memory(model_id)
        gpus = await get_gpu_info()
        return estimate, gpus

    try:
        estimate, gpus = run_async(_gather())
    except RuntimeError as exc:
        if json_out:
            emit_json(
                {"status": "error", "model_id": model_id, "error": str(exc)}
            )
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    match = _re.search(r"~([\d.]+)GB", estimate)
    est_gb = float(match.group(1)) if match else 0.0
    estimated = math.isfinite(est_gb) and est_gb > 0
    n_gpus = len(gpus)
    per_gpu_gb = est_gb / n_gpus if (n_gpus > 1 and estimated) else est_gb

    fit_rows: list[dict] = []
    any_over = False
    any_unknown = not gpus
    for g in gpus:
        try:
            total_gb = parse_gpu_reading(g.memory_total, "memory total") / 1024
            if total_gb <= 0:
                raise ValueError(f"invalid GPU memory total: {g.memory_total!r}")
        except ValueError as exc:
            any_unknown = True
            fit_rows.append(
                {
                    "index": g.index,
                    "name": g.name,
                    "total_gb": None,
                    "per_gpu_gb": round(per_gpu_gb, 1),
                    "ratio": None,
                    "over": None,
                    "error": str(exc),
                }
            )
            continue
        ratio = per_gpu_gb / total_gb
        over = ratio > 1.0
        if over:
            any_over = True
        fit_rows.append(
            {
                "index": g.index,
                "name": g.name,
                "total_gb": round(total_gb, 1),
                "per_gpu_gb": round(per_gpu_gb, 1),
                "ratio": round(ratio, 3),
                "over": over,
                "error": None,
            }
        )

    over_state = True if any_over else None if any_unknown else False
    status = "error" if any_unknown or not estimated else "over" if any_over else "ok"

    if json_out:
        emit_json(
            {
                "model_id": model_id,
                "estimate": estimate,
                "est_gb": est_gb,
                "n_gpus": n_gpus,
                "per_gpu_gb": round(per_gpu_gb, 2),
                "gpus": fit_rows,
                "any_over": over_state,
                "estimated": estimated,
                "status": status,
            }
        )
        raise typer.Exit(
            code=0 if (estimated and not any_over and not any_unknown) else 1
        )

    print(estimate)
    if not gpus:
        print("(GPU fit UNKNOWN — no GPUs detected)")
        raise typer.Exit(code=1)
    if not estimated:
        print("(no parseable size in estimate — skipping per-GPU fit view)")
        raise typer.Exit(code=1)
    if n_gpus > 1:
        print(f"TP={n_gpus}: {per_gpu_gb:.1f} GB/GPU")
    for r in fit_rows:
        if r["over"] is None:
            print(
                f"  GPU{r['index']}  UNKNOWN  {r['per_gpu_gb']:.1f}/? GB  "
                f"({r['error']})"
            )
        elif r["over"]:
            print(
                f"  GPU{r['index']}  OVER  {r['per_gpu_gb']:.1f}/{r['total_gb']:.0f} GB"
            )
        else:
            print(
                f"  GPU{r['index']}  {r['ratio'] * 100:3.0f}%  "
                f"{r['per_gpu_gb']:.1f}/{r['total_gb']:.0f} GB"
            )
    if any_over or any_unknown:
        raise typer.Exit(code=1)


@app.command("disk")
def disk(
    backend: str = typer.Option(
        "llamacpp", "--backend", "-b", help="Storage backend (vllm or llamacpp)."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit JSON instead of a human-readable summary."
    ),
) -> None:
    """Show backend model/cache storage and filesystem usage."""
    if backend not in ("vllm", "llamacpp"):
        raise typer.BadParameter(
            f"unknown backend: {backend!r} (choose vllm or llamacpp)",
            param_hint="--backend",
        )
    if backend == "vllm":
        from tui.common.docker import get_disk_usage
        from tui.common.env import host_expand, parse_env_file
        from tui.common.profile_store import PROJECT_ROOT

        common = PROJECT_ROOT / ".env.common"
        try:
            env = parse_env_file(common) if common.exists() else {}
            raw_cache = env.get("HF_CACHE_PATH", "").strip()
            if not raw_cache:
                raise ValueError(f"HF_CACHE_PATH is not set in {common}")
            cache = host_expand(raw_cache)
            cache_path = Path(cache)
            target = _nearest_existing_path(cache_path)
            cache_exists = target == cache_path
            used, avail, pct = run_async(get_disk_usage(str(target)))
        except Exception as exc:
            if json_out:
                emit_json({"status": "error", "backend": backend, "error": str(exc)})
            else:
                typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        result = {
            "backend": backend,
            "project_root": str(PROJECT_ROOT),
            "hf_cache_path": cache,
            "hf_cache_exists": cache_exists,
            "df_target": str(target),
            "df_used": used,
            "df_avail": avail,
            "df_percent": pct,
        }
        if json_out:
            emit_json(result)
            return
        print(f"Project root : {PROJECT_ROOT}")
        print(f"HF cache     : {cache}")
        print(f"df {target}: used {used}, avail {avail}  ({pct})")
        return

    from tui.backends.llamacpp.backend import (
        ROOT,
        _get_hf_cache_dir,
        _get_model_dir,
        get_disk_usage,
        list_cached_gguf,
    )

    try:
        model_dir = _get_model_dir()
        hf_cache_dir = _get_hf_cache_dir()
        cached = list_cached_gguf()
        cached_total_gb = round(
            sum(c["size_bytes"] for c in cached) / 1024**3, 1
        )
        model_dir_exists = model_dir.exists()
        files: list[dict] = []
        if model_dir_exists:
            for path in model_dir.glob("*.gguf"):
                size_bytes = path.stat().st_size
                files.append(
                    {
                        "name": path.name,
                        "size_bytes": size_bytes,
                        "size_gb": round(size_bytes / 1024**3, 1),
                    }
                )
            files.sort(key=lambda item: item["size_bytes"], reverse=True)

        target = str(_nearest_existing_path(model_dir))
        used, avail, pct = run_async(get_disk_usage(target))
        total_gb = round(sum(f["size_bytes"] for f in files) / 1024**3, 1)
    except Exception as exc:
        if json_out:
            emit_json({"status": "error", "backend": backend, "error": str(exc)})
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_out:
        emit_json(
            {
                "backend": backend,
                "project_root": str(ROOT),
                "model_dir": str(model_dir),
                "model_dir_exists": model_dir_exists,
                "hf_cache_dir": str(hf_cache_dir),
                "df_target": target,
                "df_used": used,
                "df_avail": avail,
                "df_percent": pct,
                "gguf_files": files,
                "gguf_total_gb": total_gb,
                "hf_cache_gguf_files": cached,
                "hf_cache_gguf_total_gb": cached_total_gb,
            }
        )
        return

    print(f"Project root : {ROOT}")
    print(f"Model dir    : {model_dir}")
    if not model_dir_exists:
        print("(model dir does not exist yet — no GGUF files to list)")
    else:
        print(f"GGUF files   : {len(files)}  (total {total_gb:.1f} GB)")
        for f in files:
            print(f"  {f['name']}  {f['size_gb']:.1f} GB")
    print(f"HF cache dir : {hf_cache_dir}")
    print(f"HF cache GGUF: {len(cached)}  (total {cached_total_gb:.1f} GB)")
    for c in cached:
        print(f"  {c['repo']}/{c['name']}  {c['size_gb']:.1f} GB")
    print(f"df {target}: used {used}, avail {avail}  ({pct})")


@app.command("env-check")
def env_check(
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Validate `.env.common` and report key paths."""
    from tui.common.profile_store import PROJECT_ROOT
    from tui.common.env import parse_env_file, validate_common_env

    common = PROJECT_ROOT / ".env.common"
    ok, messages = validate_common_env(common)
    warnings = [message for message in messages if message.startswith("Warning:")]
    findings = {
        "project_root": str(PROJECT_ROOT),
        "env_common_path": str(common),
        "env_common_exists": common.exists(),
        "issues": [] if ok else [message for message in messages if message not in warnings],
        "warnings": warnings,
    }
    env = parse_env_file(common) if common.exists() else {}
    token = env.get("HF_TOKEN", "")
    findings["HF_TOKEN"] = {"set": bool(token), "length": len(token)}
    findings["HF_CACHE_PATH"] = env.get("HF_CACHE_PATH", "")

    findings["status"] = "ok" if ok else "error"

    if json_out:
        emit_json(findings)
        raise typer.Exit(code=0 if ok else 1)

    print(f"Project root : {findings['project_root']}")
    print(f"Env common   : {findings['env_common_path']}")
    if not findings["env_common_exists"]:
        print("Status       : MISSING .env.common")
        raise typer.Exit(code=1)

    print(
        f"  {'HF_TOKEN':<14}: "
        + (f"<set, {len(token)} chars>" if token else "(unset — optional, needed for gated models)")
    )
    print(f"  {'HF_CACHE_PATH':<14}: {findings['HF_CACHE_PATH'] or '(unset)'}")
    print(f"  {'VLLM_VERSION':<14}: (injected at start time)")

    if findings["warnings"]:
        print("\nWarnings:")
        for warning in findings["warnings"]:
            print(f"  - {warning}")

    if not ok:
        if findings["issues"]:
            print("\nIssues:")
            for issue in findings["issues"]:
                print(f"  - {issue}")
        raise typer.Exit(code=1)
    print("\nStatus       : OK")
