from __future__ import annotations

import errno
import ipaddress
import socket
import re

from tui.common.adapter import DashboardRow
from tui.common.docker import (
    GPU_WILDCARD,
    gpu_sets_overlap,
    parse_gpu_ids,
    run_command,
)


def _format_gpu_label(gpu_id: str) -> str:
    return "all GPUs" if gpu_id == GPU_WILDCARD else f"GPU {gpu_id}"


async def gpu_conflict_messages(
    *,
    profile_name: str,
    container_name: str,
    profile_gpu_id: str,
    backend: str,
) -> list[str]:
    from tui.common import profile_store

    rc, out = await run_command("docker", "ps", "--format", "{{.Names}}", timeout=10)
    if rc != 0:
        raise RuntimeError(
            "could not inspect running containers for GPU overlap: "
            f"{out.strip() or 'docker ps failed'}"
        )
    running_names = {line.strip() for line in out.splitlines() if line.strip()}
    profile_gpu_ids = parse_gpu_ids(profile_gpu_id)
    if not profile_gpu_ids:
        return []
    messages: list[str] = []
    for backend_name in ("vllm", "llamacpp"):
        for other in profile_store.list_profiles(backend_name):
            if backend_name == backend and other.name == profile_name:
                continue
            other_container = other.container_name or other.name
            if other_container == container_name:
                continue
            if other_container not in running_names:
                continue
            other_gpu_ids = parse_gpu_ids(other.gpu_id)
            for gpu_id in sorted(gpu_sets_overlap(profile_gpu_ids, other_gpu_ids)):
                friendly = "vLLM" if backend_name == "vllm" else "llama.cpp"
                messages.append(
                    f"Warning: {_format_gpu_label(gpu_id)} is also used by running "
                    f"{friendly} container '{other_container}'"
                )
    return messages


def _row_gpu_ids(row: DashboardRow) -> set[str]:
    gpu_raw = getattr(row, "gpu_id", "") or ""
    if not gpu_raw and row.raw is not None:
        gpu_raw = getattr(row.raw, "gpu_id", "") or ""
    return parse_gpu_ids(str(gpu_raw))


def port_conflicts(target: DashboardRow, rows: list[DashboardRow]) -> list[str]:
    """Return running rows that share the target port."""
    if target.port is None:
        return []
    msgs: list[str] = []
    for r in rows:
        if r is target:
            continue
        if r.backend == target.backend and r.profile_name == target.profile_name:
            continue
        if not r.running or r.port is None:
            continue
        if r.port == target.port:
            label = _format_backend(r.backend)
            msgs.append(
                f"Port {target.port} is occupied by {label} profile "
                f"'{r.profile_name}' (container '{r.container_name}')"
            )
    return msgs


def external_port_conflicts(
    target: DashboardRow,
    rows: list[DashboardRow],
    external_ports: dict[str, str],
) -> list[str]:
    """Return unmanaged containers that expose the target port."""
    if target.port is None:
        return []
    known_containers = {r.container_name for r in rows if r.container_name}
    msgs: list[str] = []
    for cname, ports in external_ports.items():
        if cname in known_containers:
            continue
        if target.port in published_tcp_host_ports(ports):
            msgs.append(
                f"Port {target.port} is occupied by external container "
                f"'{cname}' (not managed by llmux)"
            )
    return msgs


_UNPUBLISHED_PORT_RE = re.compile(r"^(\d+)(?:-(\d+))?/(tcp|udp|sctp)$")
_PUBLISHED_PORT_RE = re.compile(
    r"^(?:(?P<address>.+):)?"
    r"(?P<host_start>\d+)(?:-(?P<host_end>\d+))?->"
    r"(?P<container_start>\d+)(?:-(?P<container_end>\d+))?/"
    r"(?P<protocol>tcp|udp|sctp)$"
)


def _port_range(start_raw: str, end_raw: str | None, raw: str) -> range:
    start = int(start_raw)
    end = int(end_raw) if end_raw is not None else start
    if start < 1 or end > 65535 or end < start:
        raise RuntimeError(f"invalid Docker port mapping: {raw!r}")
    return range(start, end + 1)


def published_tcp_host_ports(raw: str) -> set[int]:
    if not raw:
        return set()
    published: set[int] = set()
    for segment_raw in raw.split(","):
        segment = segment_raw.strip()
        if not segment:
            raise RuntimeError(f"invalid Docker port mapping: {raw!r}")
        unpublished = _UNPUBLISHED_PORT_RE.fullmatch(segment)
        if unpublished is not None:
            _port_range(unpublished.group(1), unpublished.group(2), segment)
            continue
        match = _PUBLISHED_PORT_RE.fullmatch(segment)
        if match is None:
            raise RuntimeError(f"invalid Docker port mapping: {segment!r}")
        address = match.group("address")
        if address:
            normalized = address[1:-1] if address.startswith("[") and address.endswith("]") else address
            try:
                ipaddress.ip_address(normalized)
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid Docker port mapping: {segment!r}"
                ) from exc
        host_ports = _port_range(
            match.group("host_start"), match.group("host_end"), segment
        )
        container_ports = _port_range(
            match.group("container_start"), match.group("container_end"), segment
        )
        if len(host_ports) != len(container_ports):
            raise RuntimeError(f"invalid Docker port mapping: {segment!r}")
        if match.group("protocol") == "tcp":
            published.update(host_ports)
    return published


def gpu_conflicts(target: DashboardRow, rows: list[DashboardRow]) -> list[str]:
    """running 상태 row 중 GPU index 가 교집합 있는 것. `all`/`-1` wildcard 지원."""
    target_gpus = _row_gpu_ids(target)
    if not target_gpus:
        return []
    msgs: list[str] = []
    for r in rows:
        if r is target:
            continue
        if r.backend == target.backend and r.profile_name == target.profile_name:
            continue
        if not r.running:
            continue
        other_gpus = _row_gpu_ids(r)
        if not other_gpus:
            continue
        shared = gpu_sets_overlap(target_gpus, other_gpus)
        if not shared:
            continue
        label = _format_backend(r.backend)
        for gpu in sorted(shared):
            display = "all GPUs" if gpu == GPU_WILDCARD else f"GPU {gpu}"
            msgs.append(
                f"{display} is used by {label} profile '{r.profile_name}' "
                f"(container '{r.container_name}')"
            )
    return msgs


def local_port_in_use(port: int, host: str) -> bool:
    """True when a host socket already holds `host:port` (what docker would bind)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SO_REUSEADDR: ignore our own probes' TIME_WAIT; a LISTENing socket still conflicts.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, int(port)))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            return True
        raise RuntimeError(f"local port bind probe failed for {host}:{port}: {exc}") from exc
    finally:
        sock.close()
    return False


def _format_backend(name: str) -> str:
    return {"vllm": "vLLM", "llamacpp": "llama.cpp"}.get(name, name)
