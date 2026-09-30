from __future__ import annotations

from tui.common import docker, profile_store
from tui.common.adapter import DashboardRow
from tui.common.conflicts import external_port_conflicts, gpu_conflicts, port_conflicts


async def plan_switch(backend: str, name: str) -> tuple[profile_store.StoredProfile, list[profile_store.StoredProfile]]:
    profiles = [p for owner in ("vllm", "llamacpp") for p in profile_store.list_profiles(owner)]
    target = next((p for p in profiles if (p.backend, p.name) == (backend, name)), None)
    if target is None:
        raise RuntimeError(f"profile not found: {backend}/{name}")
    running = await docker.running_container_names()
    rows = [DashboardRow(
        backend=p.backend, profile_name=p.name, container_name=p.container_name or p.name,
        port=p.port, running=(p.container_name or p.name) in running,
        gpu_id=p.gpu_id, model="", detail="", raw=p,
    ) for p in profiles]
    row = next(r for r in rows if r.raw is target)
    if row.running:
        raise RuntimeError(f"{name} is already running")
    sources = [r for r in rows if gpu_conflicts(row, [r])]
    remaining = [r for r in rows if r not in sources]
    errors = port_conflicts(row, remaining)
    errors += external_port_conflicts(row, rows, await docker.running_container_ports())
    if errors:
        raise RuntimeError("\n".join(errors))
    return target, [r.raw for r in sources]


async def stop_switch_sources(
    target: profile_store.StoredProfile, approved: list[profile_store.StoredProfile],
) -> None:
    current_target, current_sources = await plan_switch(target.backend, target.name)
    if current_target != target or any(p not in approved for p in current_sources):
        raise RuntimeError("Profiles or running models changed; confirm the switch again.")
    for source in current_sources:
        if profile_store.load_profile(source.name, source.backend) != source:
            raise RuntimeError(f"Profile {source.name} changed; confirm the switch again.")
        if source.backend == "vllm":
            from tui.backends.vllm.backend_runtime import container_down
        else:
            from tui.backends.llamacpp.backend_runtime import container_down
        rc, message = await container_down(source.name)
        if rc != 0:
            raise RuntimeError(f"Could not stop {source.name}: {message}")
        running = await docker.running_container_names()
        if (source.container_name or source.name) in running:
            raise RuntimeError(f"{source.name} is still running; the new model was not started.")
    final_target, remaining = await plan_switch(target.backend, target.name)
    if final_target != target or remaining:
        raise RuntimeError("Profiles or running models changed; the new model was not started.")
