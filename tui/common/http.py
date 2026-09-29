from __future__ import annotations

import asyncio
import csv
import json
import statistics
import time
import urllib.request

BENCH_PROMPT = "Explain the theory of relativity in about 150 words."
BENCH_MAX_TOKENS = 200
BENCH_RUNS = 3
BENCH_WARMUP = 1
# Qwen uses this key; other OpenAI-compatible templates ignore unknown kwargs.
BENCH_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


async def config_api_key(
    params: dict, container_name: str, *, backend: str = "vllm", env: dict | None = None,
) -> str:
    env = env or {}
    key = params.get("api-key") or env.get("LLAMA_API_KEY" if backend == "llamacpp" else "VLLM_API_KEY", "")
    if isinstance(key, list):
        key = key[0] if key else ""
    key_file = params.get("api-key-file") or (env.get("LLAMA_ARG_API_KEY_FILE") if backend == "llamacpp" else None)
    from_file = not key and bool(key_file)
    if not key and key_file:
        from tui.common.docker import run_command

        path = key_file
        if not isinstance(path, str) or not path.strip():
            raise ValueError("api-key-file must be a non-empty container path")
        rc, output = await run_command("docker", "exec", container_name, "cat", "--", path)
        if rc:
            raise RuntimeError("could not read api-key-file inside the server container")
        key = next((line for line in output.splitlines() if line and not line.startswith("#")), "")
        if not key:
            raise ValueError("api-key-file contains no key")
    if not isinstance(key, str) or any(ord(c) < 32 or ord(c) == 127 for c in key):
        raise ValueError("api-key must be a string without control characters")
    if backend == "llamacpp" and key and not from_file:
        key = next((value for value in next(csv.reader([key])) if value), "")
    return key


async def profile_api_key(backend: str, name: str) -> str:
    from tui.common import profile_store

    profile = profile_store.load_profile(name, backend)
    if profile is None:
        raise ValueError(f"profile {backend}/{name} not found")
    config_name = profile.config_name or profile.name
    if backend == "vllm":
        from tui.backends.vllm.backend_storage import load_config

        params = load_config(config_name).extra_params
    else:
        from tui.backends.llamacpp.backend import load_config

        params = load_config(config_name).params
    return await config_api_key(params, profile.container_name or profile.name, backend=backend, env=profile.env_vars)


async def chat_completion_bench(
    port: int | str,
    model: str,
    prompt: str = BENCH_PROMPT,
    max_tokens: int = BENCH_MAX_TOKENS,
    timeout: int = 600,
    *,
    api_key: str = "",
) -> dict:
    """단일 /v1/chat/completions 호출 → {elapsed, usage}."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": dict(BENCH_CHAT_TEMPLATE_KWARGS),
        }
    ).encode()

    loop = asyncio.get_running_loop()

    def _do() -> dict:
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
        elapsed = time.time() - t0
        d = json.loads(raw, strict=False)
        return {"elapsed": elapsed, "usage": d.get("usage", {})}

    return await loop.run_in_executor(None, _do)


async def run_bench(
    port: int | str,
    model: str,
    *,
    prompt: str = BENCH_PROMPT,
    max_tokens: int = BENCH_MAX_TOKENS,
    runs: int = BENCH_RUNS,
    warmup: int = BENCH_WARMUP,
    api_key: str = "",
) -> dict:
    """Warm up, then summarize measured completion throughput with the median."""
    if runs < 1:
        raise ValueError(f"runs must be at least 1, got {runs}")
    if warmup < 0:
        raise ValueError(f"warmup must be at least 0, got {warmup}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")

    for _ in range(warmup):
        await chat_completion_bench(
            port, model, prompt=prompt, max_tokens=max_tokens, **({"api_key": api_key} if api_key else {})
        )

    results: list[dict] = []
    for _ in range(runs):
        r = await chat_completion_bench(
            port, model, prompt=prompt, max_tokens=max_tokens, **({"api_key": api_key} if api_key else {})
        )
        usage = r.get("usage")
        if not isinstance(usage, dict) or "completion_tokens" not in usage:
            raise RuntimeError(
                "benchmark response is missing usage.completion_tokens"
            )
        raw_tokens = usage["completion_tokens"]
        if type(raw_tokens) is not int:
            raise RuntimeError(
                f"invalid usage.completion_tokens: {raw_tokens!r}"
            )
        tokens = raw_tokens
        if not 0 <= tokens <= max_tokens:
            raise RuntimeError(
                "invalid usage.completion_tokens outside the requested range "
                f"0..{max_tokens}: {tokens}"
            )
        elapsed = float(r.get("elapsed", 0.0) or 0.0)
        if elapsed <= 0:
            raise RuntimeError(f"invalid benchmark elapsed time: {elapsed}")
        tps = tokens / elapsed
        results.append({"tokens": tokens, "elapsed": elapsed, "tps": tps})

    all_tps = [r["tps"] for r in results]
    return {
        "model": model,
        "runs": results,
        "median_tps": statistics.median(all_tps),
        "min_tps": min(all_tps),
        "max_tps": max(all_tps),
    }


async def list_served_models(port: int | str, timeout: int = 5, *, api_key: str = "") -> list[str]:
    """Return model ids from `/v1/models`, raising when discovery fails."""
    loop = asyncio.get_running_loop()

    def _do() -> list[str]:
        try:
            request = urllib.request.Request(
                f"http://localhost:{port}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            )
            with urllib.request.urlopen(request, timeout=timeout) as r:
                d = json.loads(r.read())
            if not isinstance(d, dict) or not isinstance(d.get("data"), list):
                raise ValueError("response must contain a data list")
            models: list[str] = []
            for item in d["data"]:
                if not isinstance(item, dict):
                    raise ValueError("model entries must be objects")
                model_id = item.get("id")
                if not isinstance(model_id, str) or not model_id:
                    raise ValueError("model entries must contain a non-empty id")
                models.append(model_id)
            return models
        except Exception as exc:
            raise RuntimeError(
                f"model discovery failed on port {port}: {type(exc).__name__}: {exc}"
            ) from exc

    return await loop.run_in_executor(None, _do)
