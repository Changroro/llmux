from __future__ import annotations

import json

from tui.common import profile_store
from tui.common.docker import image_identity, run_command


async def profile_target(backend: str, name: str) -> tuple[str, str]:
    profile = profile_store.load_profile(name, backend)
    if profile is None:
        raise RuntimeError(f"profile not found: {backend}/{name}")
    container = profile.container_name or name
    rc, output = await run_command(
        "docker", "container", "inspect", container, "--format", "{{json .}}",
    )
    if rc == 0:
        try:
            state = json.loads(output)
            running = state["State"]["Running"]
            image = state["Image"]
            if type(running) is not bool or not isinstance(image, str) or not image:
                raise ValueError("missing container state or image ID")
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"invalid container inspection for {container}") from exc
        if running:
            return image, container
    elif not any(message in output.lower() for message in ("no such container", "no such object")):
        raise RuntimeError(output.strip() or f"could not inspect {container}")

    if backend == "vllm":
        from tui.backends.vllm.backend_runtime import _resolve_prepare_image
        from tui.backends.vllm.backend_storage import load_profile

        image, error = await _resolve_prepare_image(load_profile(name))
        if error:
            raise RuntimeError(error)
    else:
        from tui.backends.llamacpp.backend import load_profile
        from tui.backends.llamacpp.backend_runtime import _resolve_runtime_image

        image = _resolve_runtime_image(load_profile(name))
    return image, ""


async def config_target(backend: str, config_name: str, profile_name: str = "") -> tuple[str, str]:
    if profile_name:
        return await profile_target(backend, profile_name)
    profiles = [
        profile for profile in profile_store.list_profiles(backend)
        if profile_store.effective_config_name(profile) == config_name
    ] if config_name else []
    targets = [await profile_target(backend, profile.name) for profile in profiles]
    identities = {
        image if container else (await image_identity(image) or image)
        for image, container in targets
    } if len(targets) > 1 else set()
    if len(identities) > 1:
        raise RuntimeError("config uses multiple images; open it from the target profile")
    return targets[0] if targets else ("", "")
