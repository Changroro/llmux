from __future__ import annotations

import asyncio
import contextlib
import json as _json
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


def run_async(coro):
    """Drive an async coroutine to completion, returning its result."""
    return asyncio.run(coro)


def stream_async(agen: AsyncIterator) -> int:
    """Print log events from an async generator and return its final rc event."""

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
                    print(evt[1], flush=True)
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


def emit_json(data: Any) -> None:
    """Print a JSON payload (sorted keys, no trailing newline-magic)."""
    _json.dump(data, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")


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
