from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import math
import shutil


@dataclass
class GpuInfo:
    index: str
    name: str
    memory_used: str   # MiB
    memory_total: str  # MiB
    utilization: str
    temperature: str
    power: str = ""    # W (may be "[N/A]" on GPUs that don't report it)


class ContainerLifecycle(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"
    UNKNOWN = "unknown"


class ContainerHealth(str, Enum):
    NONE = "none"
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ContainerSnapshot:
    name: str
    lifecycle: ContainerLifecycle
    health: ContainerHealth
    raw_state: str
    raw_status: str

    @property
    def running(self) -> bool:
        return self.lifecycle in {
            ContainerLifecycle.RUNNING,
            ContainerLifecycle.PAUSED,
            ContainerLifecycle.RESTARTING,
            ContainerLifecycle.REMOVING,
        }

    @property
    def display_status(self) -> str:
        if (
            self.lifecycle is ContainerLifecycle.RUNNING
            and self.health is not ContainerHealth.NONE
        ):
            return self.health.value
        return self.lifecycle.value


class RunningContainerNames(set[str]):
    def __init__(self, snapshots: dict[str, ContainerSnapshot]) -> None:
        super().__init__(snapshots)
        self.snapshots = dict(snapshots)


class GpuReadingLevel(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


def parse_gpu_reading(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid GPU {field}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"invalid GPU {field}: {value!r}")
    return parsed


def _gpu_reading_level(
    value: str, field: str, warning: float, critical: float
) -> GpuReadingLevel:
    parsed = parse_gpu_reading(value, field)
    if parsed >= critical:
        return GpuReadingLevel.CRITICAL
    if parsed >= warning:
        return GpuReadingLevel.WARNING
    return GpuReadingLevel.NORMAL


def gpu_utilization_level(value: str) -> GpuReadingLevel:
    return _gpu_reading_level(value, "utilization", 50.0, 80.0)


def gpu_temperature_level(value: str) -> GpuReadingLevel:
    return _gpu_reading_level(value, "temperature", 60.0, 80.0)


async def run_command(*args: str, timeout: float = 10) -> tuple[int, str]:
    """subprocess 실행. (exitcode, combined_stdout_stderr) 반환. 타임아웃/부재 시 (1, '')."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        return 1, ""
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        return 1, ""
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
    return proc.returncode or 0, stdout.decode("utf-8", errors="replace")


async def image_identity(image_ref: str) -> str | None:
    from tui.common.dev_build import image_reference_credential_error

    error = image_reference_credential_error(image_ref)
    if error:
        raise RuntimeError(error)
    rc, output = await run_command(
        "docker",
        "image",
        "inspect",
        image_ref,
        "--format",
        "{{.Id}}",
        timeout=20,
    )
    if rc == 0:
        identity = output.strip()
        if not identity:
            raise RuntimeError(f"docker image inspect {image_ref} returned no image ID")
        return identity
    lowered = output.lower()
    if "no such image" in lowered or "no such object" in lowered:
        return None
    raise RuntimeError(output.strip() or f"docker image inspect {image_ref} failed")


async def get_gpu_info() -> list[GpuInfo]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is not installed or not on PATH")
    rc, out = await run_command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
        timeout=5,
    )
    if rc != 0:
        raise RuntimeError(out.strip() or "nvidia-smi GPU query failed")
    gpus: list[GpuInfo] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 7:
            gpus.append(GpuInfo(*parts))
        elif len(parts) == 6:
            gpus.append(GpuInfo(*parts))
        else:
            raise RuntimeError(f"unexpected nvidia-smi output: {line!r}")
    return gpus


async def get_pcie_stats() -> dict[str, tuple[float, float]]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is not installed or not on PATH")
    rc, out = await run_command(
        "nvidia-smi", "dmon", "-c", "1", "-s", "t", timeout=5
    )
    if rc != 0:
        raise RuntimeError(out.strip() or "nvidia-smi dmon PCIe query failed")
    stats: dict[str, tuple[float, float]] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise RuntimeError(f"unexpected nvidia-smi dmon output: {line!r}")
        idx = parts[0]
        try:
            rx, tx = float(parts[1]), float(parts[2])
        except ValueError as exc:
            raise RuntimeError(
                f"unexpected nvidia-smi dmon output: {line!r}"
            ) from exc
        if not math.isfinite(rx) or not math.isfinite(tx) or rx < 0 or tx < 0:
            raise RuntimeError(f"unexpected nvidia-smi dmon output: {line!r}")
        if idx in stats:
            raise RuntimeError(f"duplicate GPU in nvidia-smi dmon output: {idx!r}")
        stats[idx] = (rx, tx)
    return stats


def _container_lifecycle(raw_state: str) -> ContainerLifecycle:
    try:
        return ContainerLifecycle(raw_state.strip().lower())
    except ValueError:
        return ContainerLifecycle.UNKNOWN


def _container_health(
    lifecycle: ContainerLifecycle, raw_status: str
) -> ContainerHealth:
    lowered = raw_status.lower()
    if "(health: starting)" in lowered:
        return ContainerHealth.STARTING
    if "(unhealthy)" in lowered:
        return ContainerHealth.UNHEALTHY
    if "(healthy)" in lowered:
        return ContainerHealth.HEALTHY
    if lifecycle is ContainerLifecycle.RUNNING and "(" in raw_status:
        return ContainerHealth.UNKNOWN
    return ContainerHealth.NONE


def parse_container_snapshots(output: str) -> dict[str, ContainerSnapshot]:
    snapshots: dict[str, ContainerSnapshot] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise RuntimeError(f"unexpected docker ps output: {line!r}")
        name, raw_state, raw_status = (part.strip() for part in parts)
        if name in snapshots:
            raise RuntimeError(f"duplicate container in docker ps output: {name!r}")
        lifecycle = _container_lifecycle(raw_state)
        snapshots[name] = ContainerSnapshot(
            name=name,
            lifecycle=lifecycle,
            health=_container_health(lifecycle, raw_status),
            raw_state=raw_state,
            raw_status=raw_status,
        )
    return snapshots


async def container_snapshots(
    *, include_stopped: bool = False
) -> dict[str, ContainerSnapshot]:
    args = ["docker", "ps"]
    if include_stopped:
        args.append("--all")
    args.extend(
        ["--format", "{{.Names}}\t{{.State}}\t{{.Status}}"]
    )
    rc, out = await run_command(*args, timeout=5)
    if rc != 0:
        raise RuntimeError(out.strip() or "docker ps failed")
    return parse_container_snapshots(out)


async def running_container_names() -> RunningContainerNames:
    return RunningContainerNames(await container_snapshots())


def container_is_running(running: set[str], name: str) -> bool:
    if isinstance(running, RunningContainerNames):
        snapshot = running.snapshots.get(name)
        return snapshot.running if snapshot is not None else False
    return name in running


async def running_container_ports() -> dict[str, str]:
    rc, out = await run_command(
        "docker", "ps", "--format", "{{.Names}}\t{{.Ports}}", timeout=5
    )
    if rc != 0:
        raise RuntimeError(out.strip() or "docker ps failed")
    return parse_running_container_ports(out)


def parse_running_container_ports(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            raise RuntimeError(f"unexpected docker ps ports output: {line!r}")
        parts = line.split("\t", 1)
        if len(parts) != 2 or not parts[0].strip():
            raise RuntimeError(f"unexpected docker ps ports output: {line!r}")
        name = parts[0].strip()
        if name in result:
            raise RuntimeError(f"duplicate container in docker ps ports output: {name!r}")
        result[name] = parts[1].strip()
    return result


def parse_docker_image_rows(output: str) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in output.splitlines():
        if not line.strip():
            raise RuntimeError(f"unexpected docker image output: {line!r}")
        parts = tuple(part.strip() for part in line.split("\t"))
        if len(parts) != 4 or not all(parts):
            raise RuntimeError(f"unexpected docker image output: {line!r}")
        identity = (parts[0], parts[1])
        if identity in seen:
            raise RuntimeError(f"duplicate docker image output: {line!r}")
        seen.add(identity)
        rows.append(parts)
    return rows


def format_gpu_bar(gpus: list[GpuInfo], bar_width: int = 8) -> str:
    """GPU info 를 rich markup progress bar 로. 두 backend 가 동일 표기."""
    if not gpus:
        return "[dim]No GPUs detected[/dim]"
    parts: list[str] = []
    for g in gpus:
        try:
            used = parse_gpu_reading(g.memory_used, "memory used")
        except ValueError:
            used = None
        try:
            total = parse_gpu_reading(g.memory_total, "memory total")
        except ValueError:
            total = None
        ratio = (
            max(0.0, min(1.0, used / total))
            if used is not None and total is not None and total > 0
            else None
        )
        if ratio is None:
            bar = f"[dim]{'░' * bar_width}[/dim]"
        else:
            filled = round(ratio * bar_width)
            bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_width - filled)}[/dim]"
        used_text = "—" if used is None else f"{used / 1024:.1f}"
        total_text = "—" if total is None or total <= 0 else f"{total / 1024:.1f}"
        mem = f"{used_text}/{total_text}GB"
        try:
            utilization = f"{parse_gpu_reading(g.utilization, 'utilization'):g}%"
        except ValueError:
            utilization = "—"
        try:
            temperature = f"{parse_gpu_reading(g.temperature, 'temperature'):g}°C"
        except ValueError:
            temperature = "—"
        parts.append(
            f"[bold]GPU{g.index}[/bold] {bar}  {mem}  {utilization}  {temperature}"
        )
    return "  [dim]│[/dim]  ".join(parts)


GPU_WILDCARD = "*"


def parse_gpu_ids(raw: str) -> set[str]:
    """Parse comma-separated ids; `all` and `-1` map to the wildcard token."""
    if not raw:
        return set()
    out: set[str] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        lower = token.lower()
        if lower in {"all", "-1"}:
            out.add(GPU_WILDCARD)
        else:
            out.add(token)
    return out


def gpu_sets_overlap(a: set[str], b: set[str]) -> set[str]:
    """Return GPU overlap, expanding a wildcard against the other set."""
    if GPU_WILDCARD in a and GPU_WILDCARD in b:
        return {GPU_WILDCARD}
    if GPU_WILDCARD in a:
        return set(b) if b else {GPU_WILDCARD}
    if GPU_WILDCARD in b:
        return set(a) if a else {GPU_WILDCARD}
    return a & b


async def get_disk_usage(path: str) -> tuple[str, str, str]:
    """Return `(used, available, percent)` from `df -h`."""
    rc, out = await run_command("df", "-h", path, timeout=5)
    if rc != 0:
        raise RuntimeError(out.strip() or f"df failed for {path}")
    lines = out.strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"unexpected df output for {path}")
    parts = lines[1].split()
    # Filesystem  Size  Used Avail Use% Mounted
    if len(parts) < 5:
        raise RuntimeError(f"unexpected df output for {path}")
    return parts[2], parts[3], parts[4]
