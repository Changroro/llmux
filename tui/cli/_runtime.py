from __future__ import annotations

import asyncio
import contextlib
import functools
import json as _json
import shlex
import sys
from typing import Any, AsyncIterator, Iterable

import typer

from tui.common import profile_store

BACKENDS = ("vllm", "llamacpp")


def detect_backend(name: str, *, override: str | None = None) -> str:
    """Resolve a profile name to one backend."""
    if override:
        if override not in BACKENDS:
            raise typer.BadParameter(
                f"unknown backend: {override} (choose from {', '.join(BACKENDS)})",
                param_hint="--backend",
            )
        if profile_store.load_profile(name, override) is None:
            raise typer.BadParameter(
                f"profile '{name}' not found in backend '{override}'",
                param_hint="PROFILE",
            )
        return override

    matches = [b for b in BACKENDS if profile_store.load_profile(name, b) is not None]
    if not matches:
        raise typer.BadParameter(
            f"profile '{name}' not found in profiles.yaml",
            param_hint="PROFILE",
        )
    if len(matches) > 1:
        raise typer.BadParameter(
            f"profile '{name}' exists in multiple backends ({', '.join(matches)}); "
            "disambiguate with --backend",
            param_hint="PROFILE",
        )
    return matches[0]


def model_source(sp: profile_store.StoredProfile) -> str:
    """The weights a profile serves: the vLLM config `model`, or the GGUF repo/file."""
    if sp.backend == "vllm":
        from tui.backends.vllm.backend_storage import load_config

        try:
            model = load_config(sp.config_name or sp.name).model.strip()
        except (OSError, ValueError):
            model = ""
        return model or sp.model_id or ""
    return sp.hf_repo


def run_async(coro):
    """Drive an async coroutine to completion, returning its result."""
    return asyncio.run(coro)


# Runtime log lines carrying these prefixes are diagnostics: they go to stderr.
_DIAGNOSTIC_PREFIXES = ("Error", "✗", "Warning", "⚠")
_ERROR_PREFIXES = ("Error", "✗")


def stream_async(
    agen: AsyncIterator,
    *,
    json_out: bool = False,
    sink: dict | None = None,
    quiet: bool = False,
) -> int:
    """Print log events from an async generator and return its final rc event.

    Diagnostic lines go to stderr, the rest to stdout; with `json_out` every
    line goes to stderr so stdout stays a single JSON record. `sink` collects
    error lines and the dry-run `plan` for `finish_stream`. `quiet` drops
    everything but diagnostics.
    """

    async def _drive() -> int:
        rc: int | None = None
        protocol_error = ""
        try:
            async for evt in agen:
                if rc is not None:
                    protocol_error = "received an event after terminal rc"
                    break
                if not isinstance(evt, (tuple, list)) or len(evt) != 2:
                    protocol_error = f"malformed event: {evt!r}"
                    break
                kind = evt[0]
                if kind == "log":
                    text = str(evt[1])
                    stripped = text.lstrip()
                    diagnostic = stripped.startswith(_DIAGNOSTIC_PREFIXES)
                    if sink is not None and stripped.startswith(_ERROR_PREFIXES):
                        sink.setdefault("errors", []).append(stripped)
                    if diagnostic or not quiet:
                        print(text, file=sys.stderr if json_out or diagnostic else sys.stdout, flush=True)
                elif kind == "plan":
                    if sink is not None:
                        sink["plan"] = evt[1]
                    if not quiet:
                        command = " ".join(shlex.quote(str(a)) for a in evt[1].get("command", []))
                        print(command, file=sys.stderr if json_out else sys.stdout, flush=True)
                elif kind == "rc":
                    try:
                        rc = int(evt[1])
                    except (TypeError, ValueError):
                        protocol_error = f"invalid terminal rc: {evt[1]!r}"
                        break
                else:
                    protocol_error = f"unknown event kind: {kind!r}"
                    break
        finally:
            with contextlib.suppress(
                ProcessLookupError, OSError, asyncio.CancelledError
            ):
                await agen.aclose()
        if not protocol_error and rc is None:
            protocol_error = "stream ended without a terminal rc"
        if protocol_error:
            print(
                f"Error: lifecycle stream protocol error — {protocol_error}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        return rc

    try:
        return asyncio.run(_drive())
    except KeyboardInterrupt:
        return 130


def finish_stream(rc: int, *, json_out: bool, record: dict, sink: dict) -> None:
    """Emit the JSON result record (in JSON mode) and exit with `rc`."""
    if json_out:
        result = {**record, "ok": rc == 0, "code": rc}
        if rc != 0:
            result["error"] = "\n".join(sink.get("errors", [])) or f"exit code {rc}"
        if "plan" in sink:
            result["plan"] = sink["plan"]
        emit_json(result)
    raise typer.Exit(code=rc)


def fail(message: str, *, code: int, json_out: bool = False, record: dict | None = None) -> None:
    """Report a CLI-level failure on stderr (plus a JSON record) and exit."""
    typer.echo(message, err=True)
    if json_out:
        emit_json({**(record or {}), "ok": False, "code": code, "error": message})
    raise typer.Exit(code=code)


_json_emitted = False


def emit_json(data: Any, *, stream=None) -> None:
    """Print a JSON payload to stdout, followed by a newline."""
    global _json_emitted
    out = stream or sys.stdout
    _json.dump(data, out, ensure_ascii=False, indent=2, default=str)
    out.write("\n")
    _json_emitted = True


def json_emitted() -> bool:
    return _json_emitted


def json_command(fn):
    """Give a write command a `--json` result record.

    The command takes a `json_out` parameter and returns its record fields.
    With --json its human output moves to stderr and stdout gets
    `{"ok": true, "code": 0, **record}`. Failures exit through `main()`,
    which emits the matching error record.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not kwargs.get("json_out"):
            return fn(*args, **kwargs)
        stdout = sys.stdout
        with contextlib.redirect_stdout(sys.stderr):
            record = fn(*args, **kwargs) or {}
        emit_json({"ok": True, "code": 0, **record}, stream=stdout)

    return wrapper


JSON_OPTION = typer.Option(
    False, "--json", help="Human output goes to stderr; stdout gets one JSON result record."
)


def emit_table(rows: Iterable[dict], columns: list[str]) -> None:
    """Print a minimal padded table to stdout. No external deps."""
    rows = list(rows)
    if not rows:
        return
    widths = {c: len(c) for c in columns}
    for r in rows:
        for c in columns:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))




async def gather_conflict_warnings(profile_name: str, backend: str) -> list[str]:
    """Return conflicts from a complete cross-backend pre-flight scan."""
    from tui.backends.llamacpp.adapter import LlamacppAdapter
    from tui.backends.vllm.adapter import VllmAdapter
    from tui.common import docker as common_docker
    from tui.common.conflicts import (
        external_port_conflicts,
        gpu_conflicts,
        port_conflicts,
    )

    try:
        running = await common_docker.running_container_names()
    except Exception as exc:
        raise RuntimeError(
            f"could not enumerate running containers via `docker ps`: {exc}"
        ) from exc

    rows = []
    try:
        rows.extend(VllmAdapter().rows(running))
    except Exception as exc:
        raise RuntimeError(f"vLLM profile scan failed: {exc}") from exc
    try:
        rows.extend(LlamacppAdapter().rows(running))
    except Exception as exc:
        raise RuntimeError(f"llama.cpp profile scan failed: {exc}") from exc

    target = next(
        (r for r in rows if r.backend == backend and r.profile_name == profile_name),
        None,
    )
    if target is None:
        raise RuntimeError(
            f"profile '{profile_name}' disappeared during conflict pre-flight"
        )

    warnings: list[str] = []
    for m in port_conflicts(target, rows):
        warnings.append(f"port conflict (llmux): {m}")
    for m in gpu_conflicts(target, rows):
        warnings.append(f"GPU conflict: {m}")
    try:
        ext_ports = await common_docker.running_container_ports()
        external_conflicts = external_port_conflicts(target, rows, ext_ports)
    except Exception as exc:
        raise RuntimeError(
            f"could not inspect running container ports: {exc}"
        ) from exc
    for m in external_conflicts:
        warnings.append(f"port conflict (external): {m}")
    return warnings


def partition_conflict_warnings(warnings: list[str]) -> tuple[list[str], list[str]]:
    soft: list[str] = []
    hard: list[str] = []
    for warning in warnings:
        if warning.startswith("GPU conflict:"):
            soft.append(warning)
        else:
            hard.append(warning)
    return hard, soft


async def docker_logs_once(container_name: str, *, tail: int) -> int:
    """Print the last `tail` container log lines and return Docker's status."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "--tail", str(tail), container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        print("Error: docker executable not found", file=sys.stderr)
        return 127
    try:
        if proc.stdout is not None:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                print(line.decode("utf-8", errors="replace").rstrip("\n"), flush=True)
        await proc.wait()
        return proc.returncode or 0
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()


async def docker_logs_follow(container_name: str, *, tail: int) -> int:
    """Stream `docker logs -f` and return Docker's status."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "-f", "--tail", str(tail), container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        print("Error: docker executable not found", file=sys.stderr)
        return 127
    try:
        if proc.stdout is not None:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                print(line.decode("utf-8", errors="replace").rstrip("\n"), flush=True)
        await proc.wait()
        return proc.returncode or 0
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
            try:
                await proc.wait()
            except (asyncio.CancelledError, ProcessLookupError, OSError):
                pass
