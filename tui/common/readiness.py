"""Is a profile's server serving? Shared by `llmux ps/wait` and the TUI."""

from __future__ import annotations

import asyncio
import http.client
import urllib.error

from tui.common import docker, profile_store
from tui.common.exit_codes import ERROR, NOT_READY, OK, UNREACHABLE

_CRASHED = {"exited", "dead", "restarting"}


def profile_port(stored: profile_store.StoredProfile) -> int:
    return int(stored.port or profile_store.DEFAULTS[stored.backend]["port"])


def _classify(exc: BaseException) -> str:
    """"loading" while the server cannot answer yet; "error" for a real misconfiguration."""
    cause = exc.__cause__ or exc
    if isinstance(cause, urllib.error.HTTPError):
        return "loading" if cause.code == 503 else "error"
    if isinstance(cause, (ConnectionError, TimeoutError, http.client.HTTPException, urllib.error.URLError)):
        return "loading"
    return "error"


async def probe_served(backend: str, name: str, port: int, *, timeout: float = 2) -> tuple[str, list[str], str]:
    """(state, model ids, detail): state is "ready", "loading" or "error".

    "loading" = nothing answers yet (connection refused/reset, timeout, HTTP 503).
    "error" = the server answered wrongly or its API key cannot be read.
    """
    from tui.common.http import list_served_models, profile_api_key

    try:
        key = await profile_api_key(backend, name)
    except (OSError, RuntimeError, ValueError) as exc:
        return "error", [], f"API key lookup failed: {exc}"
    try:
        models = await list_served_models(
            port, timeout=max(1, int(timeout)), **({"api_key": key} if key else {})
        )
    except RuntimeError as exc:
        state = _classify(exc)
        cause = exc.__cause__ or exc
        return state, [], "" if state == "loading" else f"{type(cause).__name__}: {cause}"
    return ("ready" if models else "loading"), models, ""


async def wait_ready(
    stored: profile_store.StoredProfile, *, timeout: float, interval: float = 3.0
) -> tuple[int, str, list[str]]:
    """Poll until the server serves a model. Returns (exit code, message, models)."""
    container = stored.container_name or stored.name
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    first = True
    while True:
        try:
            snapshot = (await docker.container_snapshots(include_stopped=True)).get(container)
        except RuntimeError as exc:
            return UNREACHABLE, f"container status probe failed — {exc}", []
        if snapshot is None:
            return ERROR, f"{stored.name} is not running (no container '{container}')", []
        # running+unhealthy only means the load outlived the healthcheck grace period.
        if snapshot.lifecycle.value in _CRASHED:
            return ERROR, f"{stored.name} has stopped ({snapshot.display_status}); check `llmux logs {stored.name}`", []
        if snapshot.running:
            # A probe that starts after the deadline cannot count; the first always does.
            in_time = first or loop.time() <= deadline
            first = False
            state, models, detail = await probe_served(
                stored.backend, stored.name, profile_port(stored),
                timeout=min(2.0, max(deadline - loop.time(), 0.5)),
            )
            if state == "error":
                return ERROR, f"{stored.name} answered but is misconfigured — {detail}", []
            if state == "ready" and in_time:
                return OK, f"{stored.name} is ready", models
        remaining = deadline - loop.time()
        if remaining <= 0:
            return NOT_READY, f"{stored.name} is not ready after {timeout:g}s ({snapshot.display_status})", []
        await asyncio.sleep(min(interval, remaining))
