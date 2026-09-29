"""Subprocess execution helpers for backend operations."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from .backend_common import SCRIPT_DIR


async def run_command(*args: str, timeout: float = 30) -> tuple[int, str]:
    """Run a command and return (returncode, combined stdout+stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(SCRIPT_DIR),
        )
    except FileNotFoundError:
        return -1, f"Executable not found: {args[0] if args else '<empty>'}"
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        return -1, "Command timed out"
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
    rc = proc.returncode if proc.returncode is not None else -1
    return rc, (stdout or b"").decode(errors="replace")


async def run_command_with_options(
    *args: str,
    timeout: float = 30,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a command with explicit cwd/env and return combined output."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd or SCRIPT_DIR),
            env=env,
        )
    except FileNotFoundError:
        return -1, f"Executable not found: {args[0] if args else '<empty>'}"
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        return -1, "Command timed out"
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
    rc = proc.returncode if proc.returncode is not None else -1
    return rc, (stdout or b"").decode(errors="replace")


async def stream_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
):
    """Yield stdout/stderr lines followed by the final return code."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd or SCRIPT_DIR),
            env=env,
        )
    except FileNotFoundError:
        yield ("log", f"✗ Executable not found: {args[0] if args else '<empty>'}")
        yield ("rc", -1)
        return
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield ("log", line.decode(errors="replace").rstrip("\n"))
        await proc.wait()
        rc = proc.returncode if proc.returncode is not None else -1
        yield ("rc", rc)
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
