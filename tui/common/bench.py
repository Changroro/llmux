"""Load benchmark against an OpenAI-compatible server — backend-agnostic.

Chat requests stream (SSE) so time-to-first-token and per-token latency can be
measured; embedding and rerank servers get their own endpoints. Blocking stdlib
HTTP runs in a thread pool sized to the concurrency, so no extra dependency.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from tui.common.http import BENCH_CHAT_TEMPLATE_KWARGS, BENCH_MAX_TOKENS, BENCH_PROMPT
from tui.common.ssl_ctx import open_url

KINDS = ("chat", "embedding", "rerank")
_PATHS = {"chat": "/v1/chat/completions", "embedding": "/v1/embeddings", "rerank": "/v1/rerank"}
_RERANK_DOCUMENTS = [
    "Relativity relates space and time.",
    "Bananas are rich in potassium.",
    "Light speed is constant for all observers.",
]


def detect_kind(backend: str, params: dict) -> str:
    """What the server was started to serve, read from its config flags."""
    if backend == "llamacpp":
        if params.get("reranking") or params.get("rerank") or params.get("pooling") == "rank":
            return "rerank"
        if params.get("embedding") or params.get("embeddings"):
            return "embedding"
        return "chat"
    task = str(params.get("task") or params.get("convert") or "")
    if task in ("score", "rerank"):
        return "rerank"
    if task in ("embed", "embedding") or params.get("runner") == "pooling":
        return "embedding"
    return "chat"


def profile_kind(backend: str, config_name: str) -> str:
    """detect_kind for a profile's linked config."""
    if backend == "vllm":
        from tui.backends.vllm.backend_storage import load_config

        return detect_kind(backend, load_config(config_name).extra_params)
    from tui.backends.llamacpp.backend import load_config

    return detect_kind(backend, load_config(config_name).params)


def _payload(kind: str, model: str, prompt: str, max_tokens: int) -> dict:
    if kind == "chat":
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": dict(BENCH_CHAT_TEMPLATE_KWARGS),
        }
    if kind == "embedding":
        return {"model": model, "input": prompt}
    return {"model": model, "query": prompt, "documents": _RERANK_DOCUMENTS}


def _one_request(url: str, body: bytes, kind: str, api_key: str, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    start = time.perf_counter()
    first = None
    out_tokens = in_tokens = 0
    chunks = 0
    with open_url(urllib.request.Request(url, data=body, headers=headers), timeout=timeout) as response:
        if kind != "chat":
            usage = json.loads(response.read()).get("usage") or {}
            in_tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
        else:
            done = False
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(f"server error in stream: {event['error']}")
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        chunks += 1
                        if first is None:
                            first = time.perf_counter()
                usage = event.get("usage")
                if usage:
                    out_tokens = int(usage.get("completion_tokens") or 0)
                    in_tokens = int(usage.get("prompt_tokens") or 0)
            if not done:
                raise RuntimeError("stream ended without [DONE]")
    end = time.perf_counter()
    out_tokens = out_tokens or chunks  # servers without stream usage: one token per chunk
    result = {"e2e": end - start, "out": out_tokens, "in": in_tokens, "ttft": None, "tpot": None}
    if first is not None:
        result["ttft"] = first - start
        if out_tokens > 1:
            result["tpot"] = (end - first) / (out_tokens - 1)
    return result


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank percentile.
    return {
        f"p{q}": ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]
        for q in (50, 90, 99)
    }


async def run_load(
    port: int | str,
    model: str,
    *,
    kind: str,
    concurrency: int,
    requests: int,
    warmup: int = 1,
    prompt: str = BENCH_PROMPT,
    max_tokens: int = BENCH_MAX_TOKENS,
    api_key: str = "",
    timeout: float = 600,
) -> dict:
    """Send `requests` requests with at most `concurrency` in flight and summarize."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, got {kind!r}")
    for name, value, low in (("concurrency", concurrency, 1), ("requests", requests, 1), ("warmup", warmup, 0)):
        if value < low:
            raise ValueError(f"{name} must be at least {low}, got {value}")
    if kind == "chat" and max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")

    url = f"http://127.0.0.1:{port}{_PATHS[kind]}"
    body = json.dumps(_payload(kind, model, prompt, max_tokens)).encode()
    loop = asyncio.get_running_loop()
    results: list[dict] = []
    errors: list[str] = []

    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        async def send() -> dict:
            return await loop.run_in_executor(pool, _one_request, url, body, kind, api_key, timeout)

        for _ in range(warmup):
            try:
                await send()
            except Exception as exc:  # noqa: BLE001 — a warmup failure shows up in the measured run
                errors.append(f"warmup: {type(exc).__name__}: {exc}")

        gate = asyncio.Semaphore(concurrency)

        async def measured() -> None:
            async with gate:
                try:
                    results.append(await send())
                except Exception as exc:  # noqa: BLE001 — counted, the run goes on
                    errors.append(f"{type(exc).__name__}: {exc}")

        started = time.perf_counter()
        await asyncio.gather(*(measured() for _ in range(requests)))
        duration = time.perf_counter() - started
    finally:
        # Never join on the event loop: a cancelled run leaves in-flight requests to their timeout.
        pool.shutdown(wait=False, cancel_futures=True)

    out_tokens = sum(r["out"] for r in results)
    in_tokens = sum(r["in"] for r in results)
    return {
        "model": model,
        "kind": kind,
        "concurrency": concurrency,
        "requests": requests,
        "succeeded": len(results),
        "failed": requests - len(results),
        "duration_s": duration,
        "req_s": len(results) / duration if duration > 0 else 0.0,
        "output_tokens": out_tokens,
        "output_tok_s": out_tokens / duration if duration > 0 else 0.0,
        "input_tokens": in_tokens,
        "input_tok_s": in_tokens / duration if duration > 0 else 0.0,
        "ttft_s": _percentiles([r["ttft"] for r in results if r["ttft"] is not None]),
        "tpot_s": _percentiles([r["tpot"] for r in results if r["tpot"] is not None]),
        "e2e_s": _percentiles([r["e2e"] for r in results]),
        "errors": errors[:5],
    }


def uses_load_mode(kind: str, concurrency: int | None, requests: int | None) -> bool:
    """Plain chat keeps the sequential median bench; anything else runs the load bench."""
    return kind != "chat" or concurrency is not None or requests is not None


def load_summary(r: dict) -> str:
    """One line for a TUI notification."""
    def ms(p: dict | None, key: str = "p50") -> str:
        return "—" if not p else f"{p[key] * 1000:.0f} ms"

    rate = f"{r['output_tok_s']:.1f} tok/s" if r["kind"] == "chat" else f"{r['input_tok_s']:.1f} in-tok/s"
    head = f"{r['kind']} c={r['concurrency']}: {r['succeeded']} ok / {r['failed']} failed, {r['req_s']:.2f} req/s, {rate}"
    if r["kind"] == "chat":
        return f"{head}, TTFT p50 {ms(r['ttft_s'])}, per-token p50 {ms(r['tpot_s'])}"
    return f"{head}, latency p50 {ms(r['e2e_s'])} p99 {ms(r['e2e_s'], 'p99')}"
