from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tui.common.docker import parse_gpu_reading
from tui.common.i18n import t
from tui.common.metrics import Hist, MetricsSnapshot

_VLLM_BLUE = "#4c8dff"
_LLAMA_GREEN = "#51cf66"
_BORDER = "#3b4048"
_ALERT = "#f87171"
_TITLE = "#8b93a7"
_DIM = "#6b7280"

MIN_INTERVAL = 0.25
MAX_INTERVAL = 5.0
INTERVAL_STEP = 0.25

REQUEST_GAUGE_FULL = 32.0

_STOPS = [
    (59, 130, 246),   # blue
    (34, 211, 238),   # cyan
    (74, 222, 128),   # green
    (250, 204, 21),   # yellow
    (251, 146, 60),   # orange
    (239, 68, 68),    # red
]

_BRAILLE_BLANK = 0x2800
# Dot bit per (col 0|1, row 0..3) in Unicode braille cell numbering.
_DOT_BITS = {
    (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
    (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80,
}
_BLOCKS = "▁▂▃▄▅▆▇█"


def _heat(ratio: float) -> str:
    r = max(0.0, min(1.0, ratio))
    x = r * (len(_STOPS) - 1)
    i = int(x)
    f = x - i
    a = _STOPS[i]
    b = _STOPS[min(i + 1, len(_STOPS) - 1)]
    return "#%02x%02x%02x" % tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))


def braille_lines(
    values, cols: int, rows: int, max_val: float | None = None
) -> list[Text]:
    """Render an area graph in a braille cell grid."""
    cols = max(1, cols)
    rows = max(1, rows)
    dot_cols = cols * 2
    dot_rows = rows * 4
    series = [None if v is None else max(0.0, float(v)) for v in list(values)[-dot_cols:]]
    series = [None] * (dot_cols - len(series)) + series
    measured = [v for v in series if v is not None]
    hi = max_val if max_val is not None and max_val > 0 else (
        max(measured) if measured else 0.0
    )

    grid = [[False] * dot_cols for _ in range(dot_rows)]
    if hi > 0:
        for dx, v in enumerate(series):
            if v is None:
                continue
            level = int(round(min(1.0, v / hi) * dot_rows))
            for filled in range(level):
                grid[dot_rows - 1 - filled][dx] = True

    out: list[Text] = []
    for cy in range(rows):
        chars: list[str] = []
        for cx in range(cols):
            bits = 0
            for (col, row), bit in _DOT_BITS.items():
                if grid[cy * 4 + row][cx * 2 + col]:
                    bits |= bit
            chars.append(chr(_BRAILLE_BLANK + bits))
        ratio = (rows - 1 - cy) / (rows - 1) if rows > 1 else 1.0
        out.append(Text("".join(chars), style=_heat(ratio)))
    return out


def _spark(values, width: int) -> str:
    vals = list(values)[-width:]
    if not vals:
        return ""
    measured = [v for v in vals if v is not None]
    if not measured:
        return ""
    hi = max(measured) or 1.0
    return "".join(
        " " if v is None else _BLOCKS[min(7, int(v / hi * 7))] for v in vals
    )


def gradient_bar(ratio: float | None, width: int) -> Text:
    """A heat-spectrum bar: filled cells run blue→red along their length."""
    bar = Text()
    if ratio is None:
        bar.append("░" * width, style=_DIM)
        return bar
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    for i in range(width):
        if i < filled:
            bar.append("█", style=_heat(i / max(width - 1, 1)))
        else:
            bar.append("░", style=_BORDER)
    return bar


def _dur(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _num(value: float | None, fmt: str = "{:.0f}") -> str:
    return "—" if value is None else fmt.format(value)


def _fmt_uptime(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


@dataclass
class Derived:
    gen_tps: float | None = None
    prompt_tps: float | None = None
    prefix_rate: float | None = None
    done_tps: float | None = None
    lat: dict[str, float | None] = field(default_factory=dict)


class MonitorState:
    """Rolling history + previous samples → per-window rates, peaks, uptime."""

    def __init__(self, history: int = 240) -> None:
        self.gen: deque[float | None] = deque(maxlen=history)
        self.prompt: deque[float | None] = deque(maxlen=history)
        self.kv: deque[float | None] = deque(maxlen=history)
        self.peak_gen: float | None = None
        self.peak_prompt: float | None = None
        self.peak_kv: float | None = None
        self._prev_tokens: tuple[float, float, float] | None = None
        self._prev_prefix: tuple[float, float] | None = None
        self._prev_done: tuple[float, float] | None = None
        self._prev_hist: dict[str, tuple[float, float]] = {}

    def reset_peaks(self) -> None:
        self.peak_gen = self.peak_prompt = self.peak_kv = None

    def update(self, snap: MetricsSnapshot | None, now: float, _lag_ms: float) -> Derived:
        d = Derived()
        if snap is None:
            self.gen.append(None)
            self.prompt.append(None)
            self.kv.append(None)
            self._prev_tokens = None
            self._prev_prefix = None
            self._prev_done = None
            self._prev_hist.clear()
            return d

        counters = snap.token_counters()
        if counters is not None:
            p, g = counters
            if self._prev_tokens is not None:
                p0, g0, t0 = self._prev_tokens
                dt = now - t0
                if dt > 0 and p >= p0 and g >= g0:
                    d.prompt_tps = (p - p0) / dt
                    d.gen_tps = (g - g0) / dt
            self._prev_tokens = (p, g, now)
        else:
            self._prev_tokens = None
        if d.gen_tps is None and snap.gen_tps_gauge is not None:
            d.gen_tps = snap.gen_tps_gauge
        if d.prompt_tps is None and snap.prompt_tps_gauge is not None:
            d.prompt_tps = snap.prompt_tps_gauge

        self.gen.append(d.gen_tps)
        self.prompt.append(d.prompt_tps)
        kv_pct = snap.kv_cache_usage * 100 if snap.kv_cache_usage is not None else None
        self.kv.append(kv_pct)
        if d.gen_tps is not None:
            self.peak_gen = d.gen_tps if self.peak_gen is None else max(
                self.peak_gen, d.gen_tps
            )
        if d.prompt_tps is not None:
            self.peak_prompt = d.prompt_tps if self.peak_prompt is None else max(
                self.peak_prompt, d.prompt_tps
            )
        if kv_pct is not None:
            self.peak_kv = kv_pct if self.peak_kv is None else max(
                self.peak_kv, kv_pct
            )

        if snap.prefix_hits is not None and snap.prefix_queries is not None:
            h, q = snap.prefix_hits, snap.prefix_queries
            if self._prev_prefix is not None:
                h0, q0 = self._prev_prefix
                if q - q0 > 0 and h >= h0:
                    d.prefix_rate = (h - h0) / (q - q0)
                elif q > 0:
                    d.prefix_rate = h / q
            elif q > 0:
                d.prefix_rate = h / q
            self._prev_prefix = (h, q)

        if snap.requests_finished is not None:
            f = snap.requests_finished
            if self._prev_done is not None:
                f0, t0 = self._prev_done
                dt = now - t0
                if dt > 0 and f >= f0:
                    d.done_tps = (f - f0) / dt
            self._prev_done = (f, now)

        for fld in ("ttft", "e2e", "tpot", "queue", "prefill", "decode", "infer"):
            d.lat[fld] = self._win_avg(fld, getattr(snap, fld))
        return d

    def _win_avg(self, field_name: str, h: Hist | None) -> float | None:
        if h is None or h.sum is None or h.count is None:
            self._prev_hist.pop(field_name, None)
            return None
        prev = self._prev_hist.get(field_name)
        self._prev_hist[field_name] = (h.sum, h.count)
        if prev is not None and h.count > prev[1] and h.sum >= prev[0]:
            return (h.sum - prev[0]) / (h.count - prev[1])
        return h.avg()


def _throughput_panel(snap, state: MonitorState, d: Derived, width: int) -> Panel:
    cols = max(20, width - 6)
    measured_gen = [value for value in state.gen if value is not None]
    scale_candidates = list(measured_gen)
    if state.peak_gen is not None:
        scale_candidates.append(state.peak_gen)
    scale = max(scale_candidates) if scale_candidates else None
    graph = braille_lines(state.gen, cols, 5, max_val=scale)

    stats = Text()
    gen = d.gen_tps
    prompt = d.prompt_tps
    stats.append("  generation ", style=_TITLE)
    stats.append(f"{gen:7.1f}" if gen is not None else "      —", style="bold #51cf66")
    stats.append(f"  ▲{_num(state.peak_gen)}", style=_DIM)
    stats.append("     prompt ", style=_TITLE)
    stats.append(f"{prompt:6.1f}" if prompt is not None else "     —", style=f"bold {_VLLM_BLUE}")
    stats.append(f"  ▲{_num(state.peak_prompt)}", style=_DIM)
    stats.append(f"     scale {_num(scale)} tok/s", style=_DIM)

    prompt_spark = Text("  prompt   ", style=_DIM)
    prompt_spark.append(_spark(state.prompt, cols), style=_VLLM_BLUE)

    body = Group(*graph, prompt_spark, stats)
    return Panel(
        body, title="THROUGHPUT · tok/s", title_align="left",
        border_style=_BORDER, style=_TITLE, box=box.ROUNDED, padding=(0, 1),
    )


def _kv_panel(snap, state: MonitorState, width: int) -> Panel:
    cols = max(12, width - 6)
    kv_now = state.kv[-1] if state.kv else None
    has_kv = snap is not None and snap.kv_cache_usage is not None
    graph = braille_lines(state.kv, cols, 4, max_val=100.0)
    stat = Text()
    stat.append("  KV ", style=_TITLE)
    stat.append(f"{kv_now:4.0f}%" if has_kv and kv_now is not None else "   —", style="bold white")
    peak = "—" if state.peak_kv is None else f"{state.peak_kv:.0f}%"
    stat.append(f"   ▲peak {peak}", style=_DIM)
    body = Group(*graph, stat)
    return Panel(
        body, title="KV CACHE · %", title_align="left",
        border_style=_BORDER, style=_TITLE, box=box.ROUNDED, padding=(0, 1),
    )


def _requests_panel(snap, d: Derived) -> Panel:
    tbl = Table.grid(expand=True, padding=(0, 1))
    tbl.add_column(style=_TITLE, no_wrap=True)
    tbl.add_column(justify="right")

    run = None if snap is None else snap.requests_running
    wait = None if snap is None else snap.requests_waiting
    preempt = None if snap is None else snap.preemptions

    run_cell = Text()
    run_cell.append(_num(run), style="bold white")
    run_cell.append("  ")
    run_cell.append_text(gradient_bar(min((run or 0.0) / REQUEST_GAUGE_FULL, 1.0) if run is not None else None, 14))
    tbl.add_row("running", run_cell)

    wait_cell = Text()
    wait_cell.append(_num(wait), style="bold #facc15" if (wait or 0) else "white")
    wait_cell.append("  ")
    wait_cell.append_text(gradient_bar(min((wait or 0.0) / REQUEST_GAUGE_FULL, 1.0) if wait is not None else None, 14))
    tbl.add_row("waiting", wait_cell)

    tbl.add_row("done/s", Text(_num(d.done_tps, "{:.1f}"), style="white"))
    tbl.add_row("preempt", Text(_num(preempt), style="#fb923c" if (preempt or 0) else _DIM))
    return Panel(
        tbl, title="REQUESTS", title_align="left",
        border_style=_BORDER, style=_TITLE, box=box.ROUNDED, padding=(0, 1),
    )


def _cache_panel(snap, d: Derived) -> Panel:
    tbl = Table.grid(expand=True, padding=(0, 1))
    tbl.add_column(style=_TITLE, no_wrap=True)
    tbl.add_column()
    tbl.add_column(justify="right", no_wrap=True)

    def row(label: str, ratio: float | None):
        pct = "—" if ratio is None else f"{ratio * 100:.0f}%"
        tbl.add_row(label, gradient_bar(ratio, 18), Text(pct, style="white"))

    kv = None if snap is None else snap.kv_cache_usage
    row("KV", kv)
    row("prefix", d.prefix_rate)
    ext = None
    if (
        snap is not None
        and snap.ext_prefix_hits is not None
        and snap.ext_prefix_queries is not None
        and snap.ext_prefix_queries > 0
    ):
        ext = snap.ext_prefix_hits / snap.ext_prefix_queries
    row("external", ext)
    return Panel(
        tbl, title="CACHE HIT", title_align="left",
        border_style=_BORDER, style=_TITLE, box=box.ROUNDED, padding=(0, 1),
    )


def _latency_panel(snap, d: Derived) -> Panel:
    tbl = Table.grid(expand=True, padding=(0, 2))
    tbl.add_column(style=_TITLE, no_wrap=True)
    tbl.add_column(justify="right")
    tbl.add_column(justify="right")
    tbl.add_column(justify="right")

    def hist(name: str, h: Hist | None):
        if h is None:
            tbl.add_row(name, Text("—", style=_DIM), "", "")
            return
        p50 = _dur(h.quantile(0.50))
        p95 = _dur(h.quantile(0.95))
        p99 = _dur(h.quantile(0.99))
        tbl.add_row(
            name,
            Text(f"p50 {p50}", style="white"),
            Text(f"p95 {p95}", style="#facc15"),
            Text(f"p99 {p99}", style="#fb923c"),
        )

    hist("TTFT", None if snap is None else snap.ttft)
    hist("E2E", None if snap is None else snap.e2e)

    tpot = d.lat.get("tpot")
    tpot_txt = _dur(tpot)
    if tpot is None and snap is not None and snap.gen_tps_gauge:
        tpot_txt = f"≈{1000 / snap.gen_tps_gauge:.0f}ms"
    queue = _dur(d.lat.get("queue"))
    tbl.add_row("TPOT", Text(f"avg {tpot_txt}", style="white"), Text("queue", style=_TITLE), Text(queue, style="white"))

    prefill = _dur(d.lat.get("prefill"))
    decode = _dur(d.lat.get("decode"))
    infer = _dur(d.lat.get("infer"))
    tbl.add_row(
        "phases",
        Text(f"prefill {prefill}", style="white"),
        Text(f"decode {decode}", style="white"),
        Text(f"infer {infer}", style="white"),
    )
    return Panel(
        tbl, title="LATENCY", title_align="left",
        border_style=_BORDER, style=_TITLE, box=box.ROUNDED, padding=(0, 1),
    )


def _gpu_title(gpus) -> str:
    names: list[str] = []
    for g in gpus:
        name = (g.name or "").strip()
        if name and name not in names:
            names.append(name)
    return "GPU · " + " · ".join(names) if names else "GPU"


def _gpu_panel(gpus, pcie: dict[str, tuple[float, float]]) -> Panel:
    tbl = Table.grid(expand=True, padding=(0, 1))
    for _ in range(6):
        tbl.add_column()
    if not gpus:
        return Panel(
            Text(t("No GPUs detected", "감지된 GPU 없음"), style=_DIM),
            title="GPU", title_align="left", border_style=_BORDER,
            style=_TITLE, box=box.ROUNDED, padding=(0, 1),
        )
    for g in gpus:
        try:
            util = parse_gpu_reading(g.utilization, "utilization")
        except ValueError:
            util = None
        try:
            used = parse_gpu_reading(g.memory_used, "memory used")
        except ValueError:
            used = None
        try:
            total = parse_gpu_reading(g.memory_total, "memory total")
        except ValueError:
            total = None
        mem_ratio = (
            used / total
            if used is not None and total is not None and total > 0
            else None
        )
        util_cell = Text()
        util_cell.append_text(gradient_bar(None if util is None else util / 100, 12))
        util_cell.append(" —" if util is None else f" {util:3.0f}%", style="white")
        mem_cell = Text()
        mem_cell.append_text(gradient_bar(mem_ratio, 12))
        used_text = "—" if used is None else f"{used / 1024:.1f}"
        total_text = "—" if total is None or total <= 0 else f"{total / 1024:.0f}"
        mem_cell.append(f" {used_text}/{total_text}G", style="white")
        try:
            temperature = f"{parse_gpu_reading(g.temperature, 'temperature'):g}°C"
        except ValueError:
            temperature = "—"
        try:
            power_value = parse_gpu_reading(g.power, "power")
            power = f"{power_value:g}W"
        except ValueError:
            power = "—"
        rx, tx = pcie.get(g.index, (None, None))
        pcie_txt = "—" if rx is None else f"rx {rx:.0f} tx {tx:.0f}"
        tbl.add_row(
            Text(f"GPU{g.index}", style="bold white"),
            util_cell,
            mem_cell,
            Text(temperature, style="white"),
            Text(power, style="white"),
            Text(pcie_txt + " MB/s", style=_DIM),
        )
    return Panel(
        tbl, title=_gpu_title(gpus), title_align="left", border_style=_BORDER,
        style=_TITLE, box=box.ROUNDED, padding=(0, 1),
    )


def _footer(paused: bool) -> Text:
    f = Text(style=_DIM)
    for key, label in (
        ("q", t("quit", "종료")),
        ("p", t("resume" if paused else "pause", "재개" if paused else "일시정지")),
        ("r", t("reset peaks", "피크 초기화")),
        ("+/-", t("interval", "주기")),
        ("l", t("lang", "언어")),
    ):
        f.append(f"  {key} ", style="bold #9aa4b2")
        f.append(label, style=_DIM)
    return f


@dataclass
class ModelEntry:
    row: object
    snap: MetricsSnapshot | None
    state: MonitorState
    d: Derived


def _model_label(row) -> Text:
    badge_color = _VLLM_BLUE if row.backend == "vllm" else _LLAMA_GREEN
    label = Text()
    label.append(f" {'vLLM' if row.backend == 'vllm' else 'llama.cpp'} ",
                 style=f"bold black on {badge_color}")
    label.append("  ")
    label.append(row.profile_name, style="bold white")
    if row.model:
        label.append(f"  {row.model}", style=_DIM)
    if row.port:
        label.append(f"  :{row.port}", style=_DIM)
    return label


def _model_detail(entry: ModelEntry, width: int) -> RenderableType:
    half = max(28, width // 2 - 2)
    mid = Table.grid(expand=True, padding=(0, 1))
    mid.add_column(ratio=1)
    mid.add_column(ratio=1)
    mid.add_row(
        Group(_kv_panel(entry.snap, entry.state, half), _requests_panel(entry.snap, entry.d)),
        Group(_cache_panel(entry.snap, entry.d), _latency_panel(entry.snap, entry.d)),
    )
    return Group(
        _model_label(entry.row),
        _throughput_panel(entry.snap, entry.state, entry.d, width),
        mid,
    )


def _model_compact(entry: ModelEntry, width: int) -> Panel:
    snap, d, state = entry.snap, entry.d, entry.state
    body = Table.grid(expand=True, padding=(0, 1))
    body.add_column(ratio=1)

    line = Text()
    line.append("  gen ", style=_TITLE)
    line.append(f"{d.gen_tps:7.1f}" if d.gen_tps is not None else "      —", style="bold #51cf66")
    line.append(" tok/s", style=_DIM)
    line.append(f"  ▲{_num(state.peak_gen)}", style=_DIM)
    line.append("   ")
    line.append(_spark(state.gen, max(10, width // 3)), style="#51cf66")
    body.add_row(line)

    second = Text()
    kv = None if snap is None else snap.kv_cache_usage
    second.append("  KV ", style=_TITLE)
    second.append(f"{kv * 100:3.0f}%" if kv is not None else "  —", style="white")
    second.append("  ")
    second.append_text(gradient_bar(kv, 14))
    run = None if snap is None else snap.requests_running
    wait = None if snap is None else snap.requests_waiting
    second.append("   run ", style=_TITLE)
    second.append(_num(run), style="white")
    second.append("  wait ", style=_TITLE)
    second.append(_num(wait), style="white")
    ttft = None if snap is None or snap.ttft is None else snap.ttft.quantile(0.50)
    second.append("   TTFT p50 ", style=_TITLE)
    second.append(_dur(ttft), style="white")
    second.append("   TPOT ", style=_TITLE)
    second.append(_dur(d.lat.get("tpot")), style="white")
    body.add_row(second)

    return Panel(
        body, title=_model_label(entry.row), title_align="left",
        border_style=_BORDER, style=_TITLE, box=box.ROUNDED, padding=(0, 1),
    )


def render_dashboard(
    entries: list[ModelEntry],
    gpus,
    pcie: dict[str, tuple[float, float]],
    width: int,
    *,
    paused: bool = False,
    interval: float = 1.0,
    uptime: float = 0.0,
    lag_ms: float = 0.0,
    notices: list[str] | None = None,
) -> RenderableType:
    """Render every GPU and each running model."""
    width = width if width and width > 0 else 100

    header = Table.grid(expand=True)
    header.add_column(justify="left", ratio=1)
    header.add_column(justify="right")
    left = Text()
    left.append("▌ ", style=_VLLM_BLUE)
    left.append("llmux monitor", style=f"bold {_TITLE}")
    left.append("   ")
    if entries:
        left.append(t(f"{len(entries)} model(s) running", f"모델 {len(entries)}개 실행 중"),
                    style="bold white")
    else:
        left.append(t("no model running", "실행 중인 모델 없음"), style=_DIM)
    right = Text()
    right.append(f"poll {interval:.2g}s", style=_DIM)
    right.append(" · ", style=_BORDER)
    right.append(f"lag {lag_ms:.0f}ms", style=_DIM)
    right.append(" · ", style=_BORDER)
    right.append(f"up {_fmt_uptime(uptime)}", style=_DIM)
    if paused:
        right.append("  ")
        right.append(" PAUSED ", style="bold black on #facc15")
    header.add_row(left, right)

    blocks: list[RenderableType] = [header, _gpu_panel(gpus, pcie)]
    if notices:
        blocks.append(Panel(
            Text("\n".join(notices), style=_ALERT),
            title=t("SCAN ERRORS", "스캔 오류"), title_align="left",
            border_style=_ALERT, style=_TITLE, box=box.ROUNDED, padding=(0, 1),
        ))
    if not entries:
        empty = (
            t("Model scan failed — this list is incomplete, not empty.",
              "모델 스캔 실패 — 목록이 비어 있는 게 아니라 불완전합니다.")
            if notices else
            t("Nothing is running — start one with `llmux up <profile>`.",
              "실행 중인 모델이 없습니다 — `llmux up <프로필>` 로 시작하세요.")
        )
        blocks.append(Panel(
            Text(empty, style=_ALERT if notices else _DIM),
            title="MODELS", title_align="left", border_style=_BORDER,
            style=_TITLE, box=box.ROUNDED, padding=(0, 1),
        ))
    elif len(entries) == 1:
        blocks.append(_model_detail(entries[0], width))
    else:
        blocks.extend(_model_compact(e, width) for e in entries)
    blocks.append(_footer(paused))
    return Group(*blocks)
