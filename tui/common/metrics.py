from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import math
import re
import urllib.request


class MetricsUnavailableError(RuntimeError):
    pass

_PROMPT_PREFIXES = (
    "vllm:prompt_tokens_total",
    "llamacpp:prompt_tokens_total",
)
_GENERATION_PREFIXES = (
    "vllm:generation_tokens_total",
    "llamacpp:tokens_predicted_total",
)

# Metric aliases are ordered newest-first so renamed families are not double-counted.
_COUNTER_GAUGE_FIELDS: dict[str, tuple[str, ...]] = {
    "prompt_tokens": _PROMPT_PREFIXES,
    "generation_tokens": _GENERATION_PREFIXES,
    "requests_running": ("vllm:num_requests_running", "llamacpp:requests_processing"),
    "requests_waiting": ("vllm:num_requests_waiting", "llamacpp:requests_deferred"),
    "kv_cache_usage": (
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
    ),
    "prefix_hits": ("vllm:prefix_cache_hits_total", "llamacpp:prompt_tokens_cached_total"),
    "prefix_queries": ("vllm:prefix_cache_queries_total",),
    "ext_prefix_hits": ("vllm:external_prefix_cache_hits_total",),
    "ext_prefix_queries": ("vllm:external_prefix_cache_queries_total",),
    "preemptions": ("vllm:num_preemptions_total",),
    "requests_finished": ("vllm:request_success_total",),
    "prompt_tps_gauge": ("llamacpp:prompt_tokens_seconds",),
    "gen_tps_gauge": ("llamacpp:predicted_tokens_seconds",),
    "draft_tokens": (
        "vllm:spec_decode_num_draft_tokens_total",
        "llamacpp:spec_decode_num_draft_tokens_total",
    ),
    "accepted_tokens": (
        "vllm:spec_decode_num_accepted_tokens_total",
        "llamacpp:spec_decode_num_accepted_tokens_total",
    ),
}

# vLLM renamed `time_per_output_token`; llama.cpp exposes none of these histograms.
_HISTOGRAM_BASES: dict[str, tuple[str, ...]] = {
    "ttft": ("vllm:time_to_first_token_seconds",),
    "e2e": ("vllm:e2e_request_latency_seconds",),
    "tpot": ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"),
    "queue": ("vllm:request_queue_time_seconds",),
    "prefill": ("vllm:request_prefill_time_seconds",),
    "decode": ("vllm:request_decode_time_seconds",),
    "infer": ("vllm:request_inference_time_seconds",),
}


@dataclass
class Hist:
    """A Prometheus histogram summed across label sets."""

    sum: float | None = None
    count: float | None = None
    buckets: dict[float, float] = field(default_factory=dict)  # le → cumulative

    def __post_init__(self) -> None:
        if self.sum is not None and not math.isfinite(self.sum):
            self.sum = None
        if self.count is not None and not math.isfinite(self.count):
            self.count = None
        valid_buckets = {
            le: value
            for le, value in self.buckets.items()
            if (math.isfinite(le) or le == math.inf) and math.isfinite(value)
        }
        self.buckets = (
            valid_buckets if len(valid_buckets) == len(self.buckets) else {}
        )

    def avg(self) -> float | None:
        if self.sum is None or self.count is None or self.count <= 0:
            return None
        return self.sum / self.count

    def quantile(self, q: float) -> float | None:
        """Approximate a quantile by interpolating cumulative buckets."""
        if self.count is None or self.count <= 0 or not self.buckets:
            return None
        ordered = sorted(self.buckets.items())  # by le; math.inf sorts last
        target = q * self.count
        prev_le, prev_c = 0.0, 0.0
        for le, cum in ordered:
            if cum >= target:
                if math.isinf(le):
                    return prev_le or None
                span = cum - prev_c
                if span <= 0:
                    return le
                return prev_le + (le - prev_le) * (target - prev_c) / span
            prev_le, prev_c = le, cum
        return prev_le or None


@dataclass
class MetricsSnapshot:
    """A metrics snapshot whose unavailable fields remain `None`."""

    backend: str = "unknown"                 # "vllm" | "llamacpp" | "unknown"
    prompt_tokens: float | None = None       # cumulative counter
    generation_tokens: float | None = None   # cumulative counter
    requests_running: float | None = None    # gauge
    requests_waiting: float | None = None    # gauge
    kv_cache_usage: float | None = None      # gauge, 0..1
    prefix_hits: float | None = None         # counter (tokens)
    prefix_queries: float | None = None      # counter (tokens)
    ext_prefix_hits: float | None = None     # counter (tokens)
    ext_prefix_queries: float | None = None  # counter (tokens)
    preemptions: float | None = None         # counter
    requests_finished: float | None = None   # counter (request_success_total)
    prompt_tps_gauge: float | None = None    # llama.cpp reports tok/s directly
    gen_tps_gauge: float | None = None
    draft_tokens: float | None = None        # counter, speculative decoding
    accepted_tokens: float | None = None     # counter, speculative decoding
    ttft: Hist | None = None
    e2e: Hist | None = None
    tpot: Hist | None = None
    queue: Hist | None = None
    prefill: Hist | None = None
    decode: Hist | None = None
    infer: Hist | None = None

    def __post_init__(self) -> None:
        for name in _COUNTER_GAUGE_FIELDS:
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                setattr(self, name, None)

    def token_counters(self) -> tuple[float, float] | None:
        if self.prompt_tokens is None or self.generation_tokens is None:
            return None
        return self.prompt_tokens, self.generation_tokens


_METRIC_NAME_RE = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*")
_LABEL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SAMPLE_VALUE_RE = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|Inf|NaN)"
)
_TIMESTAMP_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class _MetricSample:
    name: str
    labels: dict[str, str]
    value: float


def _metric_name_candidate(line: str) -> str | None:
    match = _METRIC_NAME_RE.match(line)
    if match is None:
        return None
    end = match.end()
    if end < len(line) and line[end] != "{" and not line[end].isspace():
        return None
    return match.group(0)


def _parse_labels(line: str, start: int) -> tuple[dict[str, str], int] | None:
    labels: dict[str, str] = {}
    cursor = start + 1
    if cursor < len(line) and line[cursor] == "}":
        return labels, cursor + 1
    while cursor < len(line):
        name_match = _LABEL_NAME_RE.match(line, cursor)
        if name_match is None:
            return None
        name = name_match.group(0)
        cursor = name_match.end()
        if name in labels or cursor >= len(line) or line[cursor] != "=":
            return None
        cursor += 1
        if cursor >= len(line) or line[cursor] != '"':
            return None
        cursor += 1
        value: list[str] = []
        while cursor < len(line) and line[cursor] != '"':
            character = line[cursor]
            if character != "\\":
                value.append(character)
                cursor += 1
                continue
            cursor += 1
            if cursor >= len(line) or line[cursor] not in {'\\', '"', "n"}:
                return None
            escaped = line[cursor]
            value.append("\n" if escaped == "n" else escaped)
            cursor += 1
        if cursor >= len(line):
            return None
        cursor += 1
        labels[name] = "".join(value)
        if cursor >= len(line):
            return None
        if line[cursor] == "}":
            return labels, cursor + 1
        if line[cursor] != ",":
            return None
        cursor += 1
        if cursor < len(line) and line[cursor] == "}":
            return labels, cursor + 1
    return None


def _parse_metric_sample(line: str) -> _MetricSample | None:
    name = _metric_name_candidate(line)
    if name is None:
        return None
    cursor = len(name)
    labels: dict[str, str] = {}
    if cursor < len(line) and line[cursor] == "{":
        parsed_labels = _parse_labels(line, cursor)
        if parsed_labels is None:
            return None
        labels, cursor = parsed_labels
    if cursor >= len(line) or not line[cursor].isspace():
        return None
    fields = line[cursor:].split()
    if len(fields) not in (1, 2) or _SAMPLE_VALUE_RE.fullmatch(fields[0]) is None:
        return None
    if len(fields) == 2:
        if _TIMESTAMP_RE.fullmatch(fields[1]) is None:
            return None
        timestamp = int(fields[1])
        if timestamp < -(2**63) or timestamp > 2**63 - 1:
            return None
    value = float(fields[0])
    if not math.isfinite(value):
        return None
    return _MetricSample(name, labels, value)


def _bucket_le(labels: dict[str, str]) -> float | None:
    if "le" not in labels:
        return None
    token = labels["le"]
    try:
        value = math.inf if token in ("+Inf", "Inf") else float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) or value == math.inf else None


def parse_snapshot(text: str) -> MetricsSnapshot:
    """Parse a monitor snapshot from Prometheus exposition text."""
    cg_values: dict[str, dict[str, float]] = {}
    cg_seen: set[tuple[str, str]] = set()
    cg_invalid: set[tuple[str, str]] = set()
    hist_values: dict[str, dict[str, Hist]] = {}
    hist_seen: set[tuple[str, str]] = set()
    hist_invalid: set[tuple[str, str]] = set()
    backend = "unknown"

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sample = _parse_metric_sample(line)
        name = sample.name if sample is not None else _metric_name_candidate(line)
        if name is None:
            continue
        if sample is not None and backend == "unknown":
            if name.startswith("vllm:"):
                backend = "vllm"
            elif name.startswith("llamacpp:"):
                backend = "llamacpp"

        for fld, families in _COUNTER_GAUGE_FIELDS.items():
            if name in families:
                key = (fld, name)
                cg_seen.add(key)
                if sample is None:
                    cg_invalid.add(key)
                else:
                    family_values = cg_values.setdefault(fld, {})
                    total = family_values.get(name, 0.0) + sample.value
                    if math.isfinite(total):
                        family_values[name] = total
                    else:
                        cg_invalid.add(key)
                break
        else:
            for fld, bases in _HISTOGRAM_BASES.items():
                matched = False
                for base in bases:
                    hist = hist_values.setdefault(fld, {}).setdefault(base, Hist())
                    key = (fld, base)
                    if name == f"{base}_sum":
                        hist_seen.add(key)
                        if sample is None:
                            hist_invalid.add(key)
                        else:
                            total = (hist.sum or 0.0) + sample.value
                            if math.isfinite(total):
                                hist.sum = total
                            else:
                                hist_invalid.add(key)
                        matched = True
                    elif name == f"{base}_count":
                        hist_seen.add(key)
                        if sample is None:
                            hist_invalid.add(key)
                        else:
                            total = (hist.count or 0.0) + sample.value
                            if math.isfinite(total):
                                hist.count = total
                            else:
                                hist_invalid.add(key)
                        matched = True
                    elif name == f"{base}_bucket":
                        hist_seen.add(key)
                        le = _bucket_le(sample.labels) if sample is not None else None
                        if le is not None and sample is not None:
                            total = hist.buckets.get(le, 0.0) + sample.value
                            if math.isfinite(total):
                                hist.buckets[le] = total
                            else:
                                hist_invalid.add(key)
                        else:
                            hist_invalid.add(key)
                        matched = True
                    if matched:
                        break
                if matched:
                    break

    cg: dict[str, float] = {}
    for fld, families in _COUNTER_GAUGE_FIELDS.items():
        for family in families:
            key = (fld, family)
            if key not in cg_seen:
                continue
            if key not in cg_invalid and family in cg_values.get(fld, {}):
                cg[fld] = cg_values[fld][family]
            break

    hists: dict[str, Hist] = {}
    for fld, bases in _HISTOGRAM_BASES.items():
        for base in bases:
            key = (fld, base)
            if key not in hist_seen:
                continue
            if key not in hist_invalid:
                hists[fld] = hist_values[fld][base]
            break

    if backend == "llamacpp" and "prefix_hits" in cg and "prompt_tokens" in cg:
        # llama.cpp's prompt_tokens_total excludes the tokens served from cache.
        cg["prefix_queries"] = cg["prefix_hits"] + cg["prompt_tokens"]

    return MetricsSnapshot(backend=backend, **cg, **hists)


def parse_token_counters(text: str) -> tuple[float, float] | None:
    """Return both cumulative token counters, or `None` if either is absent."""
    values: dict[str, float] = {}
    seen: set[str] = set()
    invalid: set[str] = set()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        sample = _parse_metric_sample(line)
        name = sample.name if sample is not None else _metric_name_candidate(line)
        if name not in _PROMPT_PREFIXES and name not in _GENERATION_PREFIXES:
            continue
        seen.add(name)
        if sample is None:
            invalid.add(name)
        else:
            total = values.get(name, 0.0) + sample.value
            if math.isfinite(total):
                values[name] = total
            else:
                invalid.add(name)

    def chosen(families: tuple[str, ...]) -> float | None:
        for family in families:
            if family not in seen:
                continue
            return None if family in invalid else values.get(family)
        return None

    prompt = chosen(_PROMPT_PREFIXES)
    generation = chosen(_GENERATION_PREFIXES)
    return None if prompt is None or generation is None else (prompt, generation)


async def fetch_token_counters(
    port: int | str, timeout: int = 2
) -> tuple[float, float] | None:
    """Fetch token counters; return `None` only when the response omits them."""
    loop = asyncio.get_running_loop()

    def _do() -> tuple[float, float] | None:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/metrics", timeout=timeout
            ) as r:
                body = r.read().decode(errors="replace")
        except Exception as exc:
            raise MetricsUnavailableError(
                f"metrics request failed on port {port}: {type(exc).__name__}: {exc}"
            ) from exc
        return parse_token_counters(body)

    return await loop.run_in_executor(None, _do)


async def fetch_snapshot(
    port: int | str, timeout: int = 2
) -> MetricsSnapshot:
    """Fetch a metrics snapshot, raising when the endpoint cannot be read."""
    loop = asyncio.get_running_loop()

    def _do() -> MetricsSnapshot:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/metrics", timeout=timeout
            ) as r:
                body = r.read().decode(errors="replace")
        except Exception as exc:
            raise MetricsUnavailableError(
                f"metrics request failed on port {port}: {type(exc).__name__}: {exc}"
            ) from exc
        return parse_snapshot(body)

    return await loop.run_in_executor(None, _do)


class ThroughputTracker:
    """Turn cumulative counters into per-second rates via successive deltas."""

    def __init__(self) -> None:
        self._last: dict[str, tuple[float, float, float]] = {}

    def update(
        self, key: str, counters: tuple[float, float], now: float
    ) -> tuple[float, float] | None:
        """Return rates once a valid monotonic baseline exists."""
        prompt, generation = counters
        if not all(math.isfinite(value) for value in (prompt, generation, now)):
            self.forget(key)
            return None
        prev = self._last.get(key)
        self._last[key] = (prompt, generation, now)

        if prev is None:
            return None

        prev_prompt, prev_generation, prev_now = prev
        dt = now - prev_now
        if dt <= 0:
            return None
        if prompt < prev_prompt or generation < prev_generation:
            return None

        return (
            (prompt - prev_prompt) / dt,
            (generation - prev_generation) / dt,
        )

    def forget(self, key: str) -> None:
        self._last.pop(key, None)
