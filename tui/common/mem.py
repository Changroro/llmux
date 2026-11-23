from __future__ import annotations

import os
import re

from tui.common.env import parse_env_file
from tui.common.ssl_ctx import redact_sensitive_text


async def estimate_model_memory(model_id: str, hf_token: str | None = None, **options) -> str:
    """Human estimate string ("~12.3GB (model: … + KV: …)" or an error text).

    `options` are passed to hf-mem: max_model_len, kv_cache_dtype, revision, gguf_file.
    """
    try:
        from hf_mem import arun  # type: ignore[import-not-found]
    except ImportError as exc:
        return f"hf-mem import failed: {type(exc).__name__}: {exc}"

    try:
        if hf_token is None:
            from tui.common.profile_store import PROJECT_ROOT

            common_env = parse_env_file(PROJECT_ROOT / ".env.common")
            hf_token = common_env.get("HF_TOKEN", "") or os.environ.get("HF_TOKEN", "")

        kwargs: dict = {"model_id": model_id, "experimental": True, **options}
        if hf_token and not hf_token.startswith("your_"):
            kwargs["hf_token"] = hf_token

        try:
            result = await arun(**kwargs)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "kv-cache-dtype" in msg or "kv_cache_dtype" in msg:
                kwargs["kv_cache_dtype"] = "fp8"
                result = await arun(**kwargs)
            else:
                raise

        mem_bytes = getattr(result, "memory", 0) or 0
        if isinstance(mem_bytes, dict):
            return (
                "hf-mem runtime failed: multiple GGUF files found; "
                "select a specific GGUF file"
            )
        kv_bytes = getattr(result, "kv_cache", 0) or 0
        total_bytes = getattr(result, "total_memory", None) or (mem_bytes + kv_bytes)
        if not total_bytes:
            return "hf-mem runtime failed: no memory data returned"
        total_gb = total_bytes / (1024**3)
        mem_gb = mem_bytes / (1024**3)
        kv_gb = kv_bytes / (1024**3)
        if kv_gb > 0:
            return f"~{total_gb:.1f}GB (model: {mem_gb:.1f}GB + KV: {kv_gb:.1f}GB)"
        return f"~{total_gb:.1f}GB"
    except Exception as exc:
        err = redact_sensitive_text(str(exc), (hf_token or "",))
        if "403" in err:
            return "gated model - HF_TOKEN required"
        if "404" in err or "not found" in err.lower():
            return "model not found on HuggingFace"
        exc_module = type(exc).__module__.split(".", 1)[0]
        exc_name = type(exc).__name__
        network_modules = {"aiohttp", "httpcore", "httpx", "requests", "urllib3"}
        network_markers = (
            "connection",
            "network",
            "offline",
            "timed out",
            "timeout",
            "dns",
            "ssl",
            "http status",
        )
        if isinstance(exc, (ConnectionError, TimeoutError)) or (
            exc_module in network_modules
            or any(marker in err.lower() for marker in network_markers)
        ):
            return f"hf-mem network failed: {exc_name}: {err}"
        return f"hf-mem runtime failed: {exc_name}: {err}"


_HUMAN_INT_RE = re.compile(r"(\d+(?:\.\d+)?)([kKmMgGtT])")


def _vllm_int_or_auto(value, key: str) -> int | None:
    """vLLM's human_readable_int_or_auto: 1k=1000, 1K=1024, 25.6k; -1/auto -> None."""
    if value is None or value == "":
        return None
    text = str(value).strip()
    if text == "-1" or text.lower() == "auto":
        return None
    match = _HUMAN_INT_RE.fullmatch(text)
    try:
        if match is None:
            return int(text)
        number, suffix = match.groups()
        if suffix.islower():
            return int(float(number) * 1000 ** ("kmgt".index(suffix) + 1))
        return int(number) * 1024 ** ("KMGT".index(suffix) + 1)
    except ValueError as exc:
        raise ValueError(f"{key}: cannot read {value!r} the way vLLM does") from exc


def profile_estimate_request(stored) -> tuple[str, dict, list[str] | None, float, int | None]:
    """(model id, hf-mem options, GPU ids used, gpu-memory-utilization, split) for one profile.

    GPU ids are in device order (None = every visible GPU); `split` is how many
    GPUs share one copy of the weights (None = all of the used ones).
    """
    from tui.common.docker import GPU_WILDCARD, parse_gpu_ids

    wildcard = GPU_WILDCARD in parse_gpu_ids(stored.gpu_id)
    gpu_ids: list[str] | None = None if wildcard else [
        part.strip() for part in str(stored.gpu_id).split(",") if part.strip()
    ]
    split: int | None = None

    config_name = stored.config_name or stored.name
    options: dict = {}
    if stored.backend == "vllm":
        from tui.backends.vllm import backend_storage

        cfg = backend_storage.load_config(config_name)
        params = {k.replace("_", "-"): v for k, v in cfg.extra_params.items()}
        model = cfg.model or stored.model_id
        max_len = _vllm_int_or_auto(params.get("max-model-len"), "max-model-len")
        if max_len is not None:
            options["max_model_len"] = max_len
        if params.get("kv-cache-dtype") and params["kv-cache-dtype"] != "auto":
            options["kv_cache_dtype"] = str(params["kv-cache-dtype"])
        if params.get("revision"):
            options["revision"] = str(params["revision"])
        utilization = float(cfg.gpu_memory_utilization)
        # vLLM shards weights over TP x PP GPUs and replicates them DP times,
        # on the first visible devices in GPU_ID order.
        split = int(stored.tensor_parallel_size or 1) * int(params.get("pipeline-parallel-size") or 1)
        used = split * int(params.get("data-parallel-size") or 1)
        if gpu_ids is not None:
            if used > len(gpu_ids):
                raise ValueError(
                    f"profile {stored.name} needs {used} GPU(s) (TP x PP x DP) but gpu_id lists {len(gpu_ids)}"
                )
            gpu_ids = gpu_ids[:used]
    else:
        from tui.backends.llamacpp import backend as lbackend
        from tui.common.prepare import resolve_gguf_file

        params = lbackend.load_config(config_name).params
        model = stored.hf_repo
        gguf = resolve_gguf_file(params.get("model-file"), stored.model_file, stored.hf_file)
        if gguf:
            options["gguf_file"] = gguf
        if params.get("ctx-size"):
            options["max_model_len"] = int(params["ctx-size"])
        cache_type = str(params.get("cache-type-k") or "").upper()
        if cache_type and cache_type != "AUTO":
            options["kv_cache_dtype"] = cache_type
        utilization = 1.0
    if not model:
        raise ValueError(f"profile {stored.backend}/{stored.name} has no Hugging Face model to estimate")
    return model, options, gpu_ids, utilization, split


def gpu_fit(
    est_gb: float, gpus, *, gpu_ids=None, utilization: float = 1.0, split: int | None = None
):
    """Split `est_gb` over the GPUs a model runs on. Returns (per-GPU GB, rows).

    `gpu_ids` (ordered; None = all GPUs) picks the devices, `split` how many of
    them share the weights (default: all of them).
    """
    from tui.common.docker import parse_gpu_reading

    selected = list(gpus)
    if gpu_ids is not None:
        by_index = {str(g.index): g for g in gpus}
        missing = [gid for gid in gpu_ids if gid not in by_index]
        if missing:
            raise ValueError(f"profile GPU {', '.join(missing)} not found (have: {', '.join(sorted(by_index)) or 'none'})")
        selected = [by_index[gid] for gid in gpu_ids]
    divisor = split or len(selected)
    per_gpu_gb = est_gb / divisor if divisor > 1 else est_gb
    rows = []
    for g in selected:
        try:
            total_gb = parse_gpu_reading(g.memory_total, "memory total") / 1024
            if total_gb <= 0:
                raise ValueError(f"invalid GPU memory total: {g.memory_total!r}")
        except ValueError as exc:
            rows.append({"index": g.index, "name": g.name, "total_gb": None, "budget_gb": None,
                         "per_gpu_gb": round(per_gpu_gb, 1), "ratio": None, "over": None, "error": str(exc)})
            continue
        budget_gb = total_gb * utilization
        rows.append({
            "index": g.index, "name": g.name, "total_gb": round(total_gb, 1),
            "budget_gb": round(budget_gb, 1), "per_gpu_gb": round(per_gpu_gb, 1),
            "ratio": round(per_gpu_gb / budget_gb, 3), "over": per_gpu_gb > budget_gb, "error": None,
        })
    return per_gpu_gb, rows
