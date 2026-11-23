from __future__ import annotations

import asyncio
from typing import Optional

import typer

from tui.cli._runtime import (
    BACKENDS,
    detect_backend,
    docker_logs_follow,
    docker_logs_once,
    emit_json,
    emit_table,
    fail,
    finish_stream,
    gather_conflict_warnings,
    partition_conflict_warnings,
    run_async,
    stream_async,
)
from tui.common import docker as common_docker
from tui.common import profile_store
from tui.common.exit_codes import CONFLICT, ERROR, INTERRUPTED, UNREACHABLE, USAGE
from tui.common.readiness import profile_port
from tui.common.dev_build import repo_url_error
from tui.common.http import (
    BENCH_MAX_TOKENS,
    BENCH_PROMPT,
    BENCH_RUNS,
    BENCH_WARMUP,
)

app = typer.Typer(help="Container lifecycle (start, stop, logs, status).", no_args_is_help=True)


class _StatusProbeFailed(RuntimeError):
    """`docker ps` failed, so container state is unknown rather than stopped."""


class _MetricsProbeFailed(RuntimeError):
    def __init__(self, failures: list[str]) -> None:
        self.failures = tuple(failures)
        super().__init__("; ".join(failures))



_JSON_OPTION = typer.Option(
    False, "--json", help="Logs go to stderr; stdout gets one JSON result record."
)


def _stream_up(bk: str, profile: str, **kwargs):
    if bk == "vllm":
        from tui.backends.vllm.backend_runtime import stream_container_up
    else:
        from tui.backends.llamacpp.backend_runtime import stream_container_up
    return stream_container_up(profile, **kwargs)


@app.command("up")
def up(
    profile: str = typer.Argument(..., help="Profile name (from profiles.yaml)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help=f"Force backend ({', '.join(BACKENDS)}); auto-detect if omitted."
    ),
    tag: str = typer.Option(
        "", "--tag", "-t",
        help="Image tag override. Without --dev: a full image ref "
        "(vLLM vllm/vllm-openai:<tag>, llama.cpp ghcr.io/foo/bar:v1). "
        "With --dev: pass just the dev tag (it's sanitized and resolved to "
        "<backend>-dev:<tag>). Empty = profile's pinned image_tag or backend default.",
    ),
    dev: bool = typer.Option(
        False, "--dev",
        help=(
            "Use the locally-built dev image (vllm-dev:<tag> for vLLM, "
            "llamacpp-dev:<tag> for llama.cpp). Triggers a one-off build if the "
            "tag is missing locally or was built from a different repo/branch."
        ),
    ),
    default_image: bool = typer.Option(
        False, "--default-image",
        help=(
            "Mirror the TUI's 'Default Image' selection — explicitly drop the "
            "profile's pinned image_tag for this run only and fall back to the "
            "compose default image. Useful when a profile has a stale dev tag "
            "pinned and you want to sanity-check against the upstream image "
            "without editing profiles.yaml."
        ),
    ),
    pull: bool = typer.Option(
        False, "--pull", help="Force --pull always for non-dev images."
    ),
    repo_url: str = typer.Option(
        "", "--repo-url",
        help="--dev only: override the source repo URL (default from .env.common per backend).",
    ),
    branch: str = typer.Option(
        "", "--branch",
        help="--dev only: override the source branch (default from .env.common per backend).",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Allow a detected GPU overlap. Port conflicts and failed port probes "
             "always abort.",
    ),
    replace: bool = typer.Option(
        False, "--replace",
        help="Stop llmux models sharing this profile's GPUs before starting it — "
             "only after every start check passed, the port is free on the host "
             "and the image is pulled/built.",
    ),
    stall_timeout: Optional[float] = typer.Option(
        None, "--stall-timeout", min=1,
        help="Seconds without log/download progress before startup counts as "
             "stalled (default 45).",
    ),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", min=1,
        help="Total seconds to wait for /v1/models once nothing is downloading "
             "(default 1800). A stalled or timed-out server is left running and "
             "exits 4; continue with `llmux wait`.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Validate and print the docker compose command. Writes no file and "
             "never builds, pulls, stops or starts anything.",
    ),
    json_out: bool = _JSON_OPTION,
) -> None:
    """Start a profile's container. Streams compose output to stdout."""
    record: dict = {"profile": profile}
    if replace and force:
        raise typer.BadParameter("--replace and --force are mutually exclusive")
    if dev and default_image:
        fail(
            "Error: --dev and --default-image are mutually exclusive (one forces a "
            "dev tag, the other clears any pinned image).",
            code=USAGE, json_out=json_out, record=record,
        )
    if tag and default_image:
        fail(
            "Error: --tag and --default-image are mutually exclusive (one names an "
            "image, the other falls back to the compose default).",
            code=USAGE, json_out=json_out, record=record,
        )

    error = repo_url_error(repo_url)
    if error:
        raise typer.BadParameter(error, param_hint="--repo-url")

    bk = detect_backend(profile, override=backend)
    record["backend"] = bk
    record["port"] = profile_port(profile_store.load_profile(profile, bk))

    if tag and not dev:
        from tui.common.dev_build import image_tag_error

        image_ref = tag
        if bk == "vllm":
            from tui.backends.vllm.backend_inspect import resolve_vllm_image_ref

            image_ref = resolve_vllm_image_ref(tag)
        error = image_tag_error(image_ref)
        if error:
            fail(f"Error: invalid image reference: {error}", code=USAGE, json_out=json_out, record=record)

    up_kwargs: dict = dict(
        use_dev=dev, use_default_image=default_image, tag=tag, pull=pull,
        repo_url=repo_url, branch=branch,
    )
    if stall_timeout is not None:
        up_kwargs["stall_timeout"] = stall_timeout
    if timeout is not None:
        up_kwargs["ready_timeout"] = timeout

    sources: list = []
    if replace:
        from tui.common.model_switch import plan_switch

        try:
            target, sources = run_async(plan_switch(bk, profile))
        except RuntimeError as exc:
            try:
                run_async(common_docker.container_snapshots())
                code = CONFLICT
            except RuntimeError:
                code = UNREACHABLE
            fail(f"Conflict pre-flight: cannot replace — {exc}", code=code, json_out=json_out, record=record)
    else:
        try:
            warnings = run_async(gather_conflict_warnings(profile, bk))
        except Exception as exc:
            fail(f"Conflict pre-flight failed: {exc}", code=UNREACHABLE, json_out=json_out, record=record)
        hard_conflicts, gpu_conflicts = partition_conflict_warnings(warnings)
        blocking = hard_conflicts or (gpu_conflicts if not force else [])
        if blocking:
            lines = [
                f"Conflict pre-flight: cannot start '{profile}' ({bk}) — "
                f"{len(blocking)} issue(s) detected:",
                *(f"  • {warning}" for warning in blocking),
                "Aborting. Re-run with --force to allow the GPU overlap."
                if gpu_conflicts and not hard_conflicts
                else "Aborting. Resolve the port check before retrying.",
            ]
            fail("\n".join(lines), code=CONFLICT, json_out=json_out, record=record)
        if force and gpu_conflicts:
            typer.echo("Warning: continuing despite GPU overlap:", err=True)
            for warning in gpu_conflicts:
                typer.echo(f"  • {warning}", err=True)

    names = [f"{p.backend}/{p.name}" for p in sources]
    sink: dict = {}
    if dry_run:
        if names:
            typer.echo("Dry run: would stop " + ", ".join(names), err=True)
            record["would_stop"] = names
        rc = stream_async(_stream_up(bk, profile, **up_kwargs, dry_run=True), json_out=json_out, sink=sink)
        finish_stream(rc, json_out=json_out, record=record, sink=sink)

    if sources:
        from tui.common.model_switch import stop_switch_sources

        # Every start check runs — and the image is pulled/built — before anything is stopped.
        rc = stream_async(
            _stream_up(bk, profile, **up_kwargs, dry_run=True, stage_image=True),
            json_out=json_out, sink=sink, quiet=True,
        )
        if rc != 0:
            finish_stream(rc, json_out=json_out, record=record, sink=sink)
        sink.pop("plan", None)
        typer.echo("Stopping: " + ", ".join(names), err=True)
        try:
            run_async(stop_switch_sources(target, sources))
        except (OSError, RuntimeError, ValueError) as exc:
            fail(f"Error: model switch aborted — {exc}", code=ERROR, json_out=json_out, record=record)
        record["stopped"] = names

    rc = stream_async(
        _stream_up(bk, profile, **up_kwargs, image_staged=bool(sources)), json_out=json_out, sink=sink
    )
    finish_stream(rc, json_out=json_out, record=record, sink=sink)


@app.command("wait")
def wait(
    profile: str = typer.Argument(..., help="Profile name (from profiles.yaml)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Force backend; auto-detect if omitted."
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", min=0, help="Seconds to wait before giving up (exit 4)."
    ),
    interval: float = typer.Option(
        3.0, "--interval", "-i", min=0.1, help="Seconds between probes."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit one JSON result record."),
) -> None:
    """Block until a started profile serves /v1/models.

    Exit 0 when ready, 4 when still loading at --timeout, 1 when the container
    is missing or crashed, 5 when docker cannot be queried.
    """
    from tui.common.readiness import wait_ready

    bk = detect_backend(profile, override=backend)
    stored = profile_store.load_profile(profile, bk)
    try:
        code, message, models = run_async(wait_ready(stored, timeout=timeout, interval=interval))
    except KeyboardInterrupt:
        raise typer.Exit(code=INTERRUPTED)
    record = {"profile": profile, "backend": bk, "port": profile_port(stored)}
    if code != 0:
        fail(f"Error: {message}", code=code, json_out=json_out, record=record)
    if json_out:
        emit_json({**record, "ok": True, "code": 0, "served_models": models})
    else:
        print(f"{message}: {', '.join(models)}")


@app.command("prepare")
def prepare(
    profile: str = typer.Argument(..., help="Profile name (from profiles.yaml)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Force backend; auto-detect if omitted."
    ),
    max_workers: Optional[int] = typer.Option(
        None, "--max-workers", "-w", min=1,
        help="Parallel download connections (HF caps each at a few MB/s). "
             "Overrides PREPARE_MAX_WORKERS from .env.common.",
    ),
    json_out: bool = _JSON_OPTION,
) -> None:
    """Render runtime files and download weights without starting the server."""
    bk = detect_backend(profile, override=backend)
    if bk == "vllm":
        from tui.backends.vllm.backend_runtime import stream_container_prepare
    else:
        from tui.backends.llamacpp.backend_runtime import stream_container_prepare
    sink: dict = {}
    rc = stream_async(stream_container_prepare(profile, max_workers=max_workers), json_out=json_out, sink=sink)
    finish_stream(rc, json_out=json_out, record={"profile": profile, "backend": bk}, sink=sink)


@app.command("down")
def down(
    profile: str = typer.Argument(..., help="Profile name (from profiles.yaml)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Force backend; auto-detect if omitted."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit one JSON result record."),
) -> None:
    """Stop a profile's container. Stopping one that isn't running succeeds."""
    bk = detect_backend(profile, override=backend)
    if bk == "vllm":
        from tui.backends.vllm.backend_runtime import container_down
    else:
        from tui.backends.llamacpp.backend_runtime import container_down
    rc, msg = run_async(container_down(profile))
    msg = (msg or "").rstrip()
    record = {"profile": profile, "backend": bk}
    if rc != 0:
        fail(f"Error: {msg}", code=rc, json_out=json_out, record=record)
    if json_out:
        emit_json({**record, "ok": True, "code": 0, "message": msg})
    elif msg:
        print(msg)


@app.command("logs")
def logs(
    profile: str = typer.Argument(..., help="Profile name (from profiles.yaml)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Force backend; auto-detect if omitted."
    ),
    tail: int = typer.Option(
        200, "--tail", "-n", help="Number of recent lines to show before following."
    ),
    follow: bool = typer.Option(
        True, "--follow/--no-follow", "-f/-F",
        help="Follow log output (default). Use --no-follow to print recent lines and exit.",
    ),
) -> None:
    """Stream container logs (Ctrl-C to stop following)."""
    bk = detect_backend(profile, override=backend)
    sp = profile_store.load_profile(profile, bk)
    container_name = sp.container_name or sp.name

    runner = docker_logs_follow if follow else docker_logs_once
    try:
        rc = run_async(runner(container_name, tail=tail))
    except KeyboardInterrupt:
        rc = 130
    raise typer.Exit(code=rc)



@app.command("benchmark")
def benchmark(
    profile: str = typer.Argument(..., help="Profile name (from profiles.yaml)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Force backend; auto-detect if omitted."
    ),
    prompt: str = typer.Option(
        BENCH_PROMPT, "--prompt",
        help="Prompt to send to /v1/chat/completions.",
    ),
    max_tokens: int = typer.Option(
        BENCH_MAX_TOKENS, "--max-tokens", help="max_tokens for the bench request."
    ),
    runs: int = typer.Option(
        BENCH_RUNS, "--runs", help="Measured runs; the reported figure is their median."
    ),
    warmup: int = typer.Option(
        BENCH_WARMUP, "--warmup", help="Warmup runs discarded before measuring."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit a JSON record instead of a human line."
    ),
) -> None:
    """Benchmark a running container (tok/s). Mirrors the TUI dashboard bench."""
    if max_tokens < 1:
        raise typer.BadParameter("must be at least 1", param_hint="--max-tokens")
    if runs < 1:
        raise typer.BadParameter("must be at least 1", param_hint="--runs")
    if warmup < 0:
        raise typer.BadParameter("must be at least 0", param_hint="--warmup")
    bk = detect_backend(profile, override=backend)
    sp = profile_store.load_profile(profile, bk)
    if not sp.port:
        typer.echo(f"profile '{profile}' has no port set", err=True)
        raise typer.Exit(code=2)

    async def _run() -> dict:
        from tui.common.http import list_served_models, profile_api_key, run_bench

        api_key = await profile_api_key(bk, profile)
        if bk == "vllm":
            served = await list_served_models(sp.port, **({"api_key": api_key} if api_key else {}))
            model = served[0] if served else (sp.model_id or "")
            if not model:
                raise RuntimeError(
                    "could not identify a served model "
                    "(/v1/models returned nothing and profile has no model_id)"
                )
        else:
            from tui.backends.llamacpp.backend import load_config as l_load_config

            cfg = l_load_config(sp.config_name or sp.name)
            model = str(cfg.get("alias", cfg.name))

        r = await run_bench(
            sp.port,
            model,
            prompt=prompt,
            max_tokens=max_tokens,
            runs=runs,
            warmup=warmup,
            **({"api_key": api_key} if api_key else {}),
        )
        return {
            "profile": profile,
            "backend": bk,
            "port": sp.port,
            "warmup": warmup,
            **r,
        }

    try:
        result = run_async(_run())
    except Exception as exc:
        typer.echo(f"benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1)

    if json_out:
        emit_json(result)
        return
    head = f"{result['profile']} ({result['backend']}, port {result['port']})"
    print(f"{head}  [{result['model']}]  warmup={result['warmup']}")
    for i, r in enumerate(result["runs"], start=1):
        print(
            f"  run {i}: {r['tokens']} tok / {r['elapsed']:.2f}s "
            f"= {r['tps']:.1f} tok/s"
        )
    print(
        f"  median: {result['median_tps']:.1f} tok/s "
        f"({result['min_tps']:.1f}–{result['max_tps']:.1f})"
    )



@app.command("stats")
def stats(
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b",
        help=f"Limit to one backend ({', '.join(BACKENDS)}); show all if omitted.",
    ),
    interval: float = typer.Option(
        2.0, "--interval", "-i", help="Seconds between /metrics samples."
    ),
    once: bool = typer.Option(
        False, "--once",
        help="Take two samples one interval apart, print one tick, then exit.",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit one compact JSON line per tick (NDJSON)."
    ),
) -> None:
    """Poll live token throughput from each running container."""
    import json as _json
    import time

    if backend and backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")
    if interval <= 0:
        raise typer.BadParameter("interval must be > 0", param_hint="--interval")
    backends = [backend] if backend else list(BACKENDS)

    from tui.common.metrics import (
        MetricsUnavailableError,
        ThroughputTracker,
        fetch_token_counters,
    )

    tracker = ThroughputTracker()

    async def _running() -> list[dict]:
        out: list[dict] = []
        for bk in backends:
            if bk == "vllm":
                from tui.backends.vllm.backend_runtime import get_container_statuses
            else:
                from tui.backends.llamacpp.backend_runtime import (
                    get_container_statuses,
                )
            try:
                statuses = await get_container_statuses()
            except RuntimeError as exc:
                raise _StatusProbeFailed(f"{bk}: {exc}") from exc
            for s in statuses:
                if s.running and s.port:
                    out.append(
                        {"backend": bk, "profile": s.profile_name, "port": s.port}
                    )
        out.sort(key=lambda r: (r["backend"], r["profile"]))
        return out

    async def _sample() -> list[dict]:
        rows: list[dict] = []
        failures: list[str] = []
        targets = await _running()
        samples = await asyncio.gather(
            *(fetch_token_counters(t["port"]) for t in targets),
            return_exceptions=True,
        )
        now = time.monotonic()
        for t, counters in zip(targets, samples):
            key = f"{t['backend']}:{t['profile']}"
            if isinstance(counters, MetricsUnavailableError):
                tracker.forget(key)
                detail = str(counters).strip() or "metrics endpoint unavailable"
                failures.append(
                    f"{t['backend']}/{t['profile']} (port {t['port']}): {detail}"
                )
                continue
            if isinstance(counters, BaseException):
                raise counters
            if counters is None:
                tracker.forget(key)
                rate = None
            else:
                rate = tracker.update(key, counters, now)
            rows.append(
                {
                    **t,
                    "prompt_tok_per_s": rate[0] if rate else None,
                    "generation_tok_per_s": rate[1] if rate else None,
                }
            )
        if failures:
            raise _MetricsProbeFailed(failures)
        return rows

    def _emit(rows: list[dict]) -> None:
        if json_out:
            print(_json.dumps(rows, ensure_ascii=False, default=str), flush=True)
            return
        if not rows:
            print("(no running containers)", flush=True)
            return
        for r in rows:
            head = f"{r['profile']} ({r['backend']}, port {r['port']})"
            if r["generation_tok_per_s"] is None:
                print(f"{head}: n/a", flush=True)
            else:
                print(
                    f"{head}: prompt {r['prompt_tok_per_s']:.1f} tok/s"
                    f"  ·  gen {r['generation_tok_per_s']:.1f} tok/s",
                    flush=True,
                )

    async def _drive() -> None:
        baseline = True
        while True:
            rows = await _sample()
            if baseline:
                baseline = False
                if not rows:
                    _emit(rows)
                    return
            else:
                _emit(rows)
                if once:
                    return
            await asyncio.sleep(interval)

    try:
        run_async(_drive())
    except KeyboardInterrupt:
        raise typer.Exit(code=130)
    except _StatusProbeFailed as exc:
        typer.echo(f"container status probe failed — {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except _MetricsProbeFailed as exc:
        for failure in exc.failures:
            typer.echo(f"metrics probe failed — {failure}", err=True)
        raise typer.Exit(code=1) from exc



@app.command("ps")
def ps(
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b",
        help=f"Limit to one backend ({', '.join(BACKENDS)}); show all if omitted.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
    running_only: bool = typer.Option(
        False, "--running", "-r", help="Only show running containers."
    ),
) -> None:
    """List profiles, their container status, and whether each one is serving.

    `ready` is true once /v1/models answers; `served_models` lists its ids.
    `ready` is null with `ready_error` when the server answered wrongly or its
    API key could not be read.
    `model` is the weights source (HF repo id or path) on both backends.
    """
    from tui.common.readiness import probe_served

    if backend and backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")
    backends = [backend] if backend else list(BACKENDS)
    rows = []

    async def _collect():
        images = await common_docker.container_images()
        for bk in backends:
            if bk == "vllm":
                from tui.backends.vllm.backend_runtime import get_container_statuses
            else:
                from tui.backends.llamacpp.backend_runtime import (
                    get_container_statuses,
                )

            try:
                statuses = await get_container_statuses()
            except RuntimeError as exc:
                raise _StatusProbeFailed(f"{bk}: {exc}") from exc
            for s in statuses:
                stored = profile_store.load_profile(s.profile_name, bk) or profile_store.StoredProfile(
                    name=s.profile_name, backend=bk
                )
                row = {
                    "backend": bk,
                    "profile": s.profile_name,
                    "container": s.container_name,
                    "status": s.status_text,
                    "running": s.running,
                    "health": s.health or None,
                    "ready": False,
                    "served_models": [],
                    "port": int(s.port) if s.port else None,
                    "gpu": s.gpu_id,
                    "image": images.get(s.container_name),
                    "model": s.model if bk == "vllm" else (stored.hf_repo or stored.model_file or ""),
                }
                if bk == "llamacpp":
                    row["hf_file"] = stored.hf_file or stored.model_file or ""
                rows.append(row)
        probes = await asyncio.gather(
            *(
                probe_served(row["backend"], row["profile"], row["port"])
                if row["running"] and row["port"] else _not_running()
                for row in rows
            )
        )
        for row, (state, models, detail) in zip(rows, probes):
            row["served_models"] = models
            # null = the server answered but readiness could not be judged (see ready_error).
            row["ready"] = None if state == "error" else state == "ready"
            if state == "error":
                row["ready_error"] = detail

    try:
        run_async(_collect())
    except (_StatusProbeFailed, RuntimeError) as exc:
        typer.echo(f"container status probe failed — {exc}", err=True)
        raise typer.Exit(code=UNREACHABLE) from exc

    out = [row for row in rows if row["running"] or not running_only]
    if json_out:
        emit_json(out)
        return

    emit_table(
        out,
        columns=["backend", "profile", "status", "ready", "port", "gpu", "container", "model"],
    )


async def _not_running() -> tuple[str, list[str], str]:
    return "loading", [], ""



@app.command("render-env")
def render_env(
    profile: Optional[str] = typer.Argument(
        None, help="Profile name to render. Omit to re-render all profiles."
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Limit when rendering all (no PROFILE given)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit one JSON result record."),
) -> None:
    """Re-render `.runtime/<backend>/<profile>.env` from `profiles.yaml`."""
    if backend and backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")
    if profile is None:
        rendered: list[dict] = []
        failed: list[dict] = []
        for bk in ([backend] if backend else list(BACKENDS)):
            for sp in profile_store.list_profiles(bk):
                try:
                    path = str(profile_store.render_env_for_profile(sp.name, sp.backend))
                except ValueError as exc:
                    failed.append({"profile": sp.name, "backend": bk, "error": str(exc)})
                    continue
                rendered.append({"profile": sp.name, "backend": bk, "path": path})
                if not json_out:
                    print(path)
        if failed:
            typer.echo("Failed to render:", err=True)
            for f in failed:
                typer.echo(f"  - {f['backend']}/{f['profile']}: {f['error']}", err=True)
        code = ERROR if failed else 0
        if json_out:
            record = {"ok": not failed, "code": code, "rendered": rendered, "failed": failed}
            if failed:
                record["error"] = "; ".join(f"{f['backend']}/{f['profile']}: {f['error']}" for f in failed)
            emit_json(record)
        raise typer.Exit(code=code)

    bk = detect_backend(profile, override=backend)
    record = {"profile": profile, "backend": bk}
    try:
        out_path = profile_store.render_env_for_profile(profile, bk)
    except ValueError as exc:
        fail(
            f"Error: cannot render {bk}/{profile}.env — {exc}\n"
            f"  Fix it with: llmux profile edit {profile} --unset <KEY>",
            code=ERROR, json_out=json_out, record=record,
        )
    if json_out:
        emit_json({**record, "ok": True, "code": 0, "path": str(out_path)})
    else:
        print(out_path)
