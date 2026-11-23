from __future__ import annotations

import asyncio
import errno
import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tui.backends.llamacpp import backend as llamacpp_backend
from tui.backends.llamacpp import backend_runtime as llamacpp_runtime
from tui.backends.vllm import backend_common as vllm_common
from tui.backends.vllm import backend_runtime as vllm_runtime
from tui.backends.vllm.backend_common import Profile as VllmProfile
from tui.common import prepare
from tui.common import docker as common_docker
from tui.common.env import validate_common_env


async def _events(stream) -> tuple[list[str], int]:
    logs: list[str] = []
    rc = -1
    async for kind, value in stream:
        if kind == "log":
            logs.append(str(value))
        elif kind == "rc":
            rc = int(value)
    return logs, rc


async def _validation_result(stream) -> tuple[bool, list[str]]:
    async for event in stream:
        if event[0] == "result":
            return bool(event[1]), list(event[2])
    raise AssertionError("validation stream ended without a result")


def _load_renderer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "llamacpp" / "render-override.py"
    spec = importlib.util.spec_from_file_location("audit_r2_render_override", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_downloader_image_validation_is_operation_specific(tmp_path: Path) -> None:
    common = tmp_path / ".env.common"
    common.write_text(
        "HF_CACHE_PATH=/tmp/llmux-cache\n"
        "VLLM_USE_V2_MODEL_RUNNER=1\n"
    )
    common.chmod(0o600)

    ok, messages = validate_common_env(common)
    assert ok, messages

    ok, messages = validate_common_env(common, require_downloader_image=True)
    assert not ok
    assert "PREPARE_DOWNLOADER_IMAGE" in "\n".join(messages)


def test_env_validation_preserves_success_warnings(tmp_path: Path) -> None:
    common = tmp_path / ".env.common"
    common.write_text("HF_CACHE_PATH=/tmp/llmux-cache\n")
    common.chmod(0o600)

    ok, messages = validate_common_env(common)

    assert ok
    assert any(message.startswith("Warning:") for message in messages)


def test_env_validation_rejects_group_or_other_permissions(tmp_path: Path) -> None:
    common = tmp_path / ".env.common"
    common.write_text("HF_CACHE_PATH=/tmp/llmux-cache\n")
    common.chmod(0o644)

    ok, messages = validate_common_env(common)

    assert not ok
    assert "owner-only" in "\n".join(messages)
    assert "0644" in "\n".join(messages)


@pytest.mark.asyncio
async def test_llamacpp_prepare_requires_downloader_during_preflight() -> None:
    validator = AsyncMock()
    validation_calls: list[bool] = []

    def validate(_path: Path, *, require_downloader_image: bool = False):
        validation_calls.append(require_downloader_image)
        return False, ["stop"]

    with (
        patch.object(
            llamacpp_runtime.profile_store,
            "load_profile",
            return_value=SimpleNamespace(name="p", backend="llamacpp"),
        ),
        patch.object(llamacpp_runtime, "validate_common_env", side_effect=validate),
        patch.object(llamacpp_runtime, "_render_override", validator),
    ):
        _logs, rc = await _events(llamacpp_runtime.stream_container_prepare("p"))

    assert rc == 1
    assert validation_calls == [True]
    validator.assert_not_awaited()


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--port", "9999"],
        ["--webui"],
        "--host=127.0.0.2",
        "--metrics off",
        ["--no-webui"],
    ],
)
def test_llamacpp_extra_args_reject_managed_flags(extra_args: str | list[str]) -> None:
    renderer = _load_renderer()

    with pytest.raises(ValueError, match="managed"):
        renderer.render_command(
            {"extra-args": extra_args},
            hf_repo="org/repo",
            hf_file="model.gguf",
        )


def test_llamacpp_extra_args_only_reject_exact_managed_tokens() -> None:
    renderer = _load_renderer()

    command = renderer.render_command(
        {"extra-args": ["--hostname", "worker"]},
        hf_repo="org/repo",
        hf_file="model.gguf",
    )

    assert command[-2:] == ["--hostname", "worker"]


def test_llamacpp_metrics_is_managed_on_not_a_template_boolean() -> None:
    renderer = _load_renderer()
    example = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "llamacpp"
        / "example.yaml"
    ).read_text()

    command = renderer.render_command(
        {"metrics": False},
        hf_repo="org/repo",
        hf_file="model.gguf",
    )

    assert command.count("--metrics") == 1
    assert "metrics:" not in example


@pytest.mark.asyncio
@pytest.mark.parametrize("ready", [False, True])
async def test_vllm_success_message_follows_readiness(ready: bool) -> None:
    profile = VllmProfile(
        name="audit",
        container_name="audit",
        config_name="audit",
    )

    async def compose(*_args, **_kwargs):
        yield "rc", 0

    async def validation(*_args, **_kwargs):
        yield "log", "readiness checked"
        yield "result", ready, [] if ready else ["Error: not ready"]

    with (
        patch.object(vllm_runtime.profile_store, "load_profile", return_value=object()),
        patch.object(
            vllm_runtime.profile_store,
            "render_env_for_profile",
            return_value=Path("audit.env"),
            create=True,
        ),
        patch.object(vllm_runtime.profile_store, "render_env", return_value=Path("audit.env")),
        patch.object(vllm_runtime, "load_profile", return_value=profile),
        patch.object(vllm_runtime, "check_port_conflict", AsyncMock(return_value=None)),
        patch.object(vllm_runtime, "_ensure_common_env", return_value=(True, [])),
        patch.object(vllm_runtime, "_ensure_profile_config", return_value=(True, [])),
        patch.object(vllm_runtime, "_gpu_conflict_messages", AsyncMock(return_value=[])),
        patch.object(
            vllm_runtime.dev_build,
            "image_exists_locally",
            AsyncMock(return_value=True),
        ),
        patch.object(vllm_runtime, "_dev_image_matches", AsyncMock(return_value=True)),
        patch.object(vllm_runtime, "_compose_env", return_value={}),
        patch.object(vllm_runtime, "stream_command", side_effect=compose),
        patch.object(vllm_runtime, "_post_start_validation", side_effect=validation),
    ):
        logs, rc = await _events(
            vllm_runtime.stream_container_up("audit", use_dev=True)
        )

    success_positions = [
        index for index, line in enumerate(logs) if "started successfully" in line
    ]
    if ready:
        assert rc == 0
        assert success_positions
        assert success_positions[0] > logs.index("readiness checked")
    else:
        assert rc == 1
        assert not success_positions


@pytest.mark.asyncio
async def test_vllm_rejects_credential_repo_before_build_output() -> None:
    secret = "R2_REPO_SECRET"
    profile = VllmProfile(name="audit", container_name="audit", config_name="audit")
    image_probe = AsyncMock(side_effect=AssertionError("image probe must not run"))
    with (
        patch.object(vllm_runtime.profile_store, "load_profile", return_value=object()),
        patch.object(
            vllm_runtime.profile_store,
            "render_env_for_profile",
            return_value=Path("audit.env"),
        ),
        patch.object(vllm_runtime, "load_profile", return_value=profile),
        patch.object(vllm_runtime, "check_port_conflict", AsyncMock(return_value=None)),
        patch.object(vllm_runtime, "_ensure_common_env", return_value=(True, [])),
        patch.object(vllm_runtime, "_ensure_profile_config", return_value=(True, [])),
        patch.object(vllm_runtime, "_gpu_conflict_messages", AsyncMock(return_value=[])),
        patch.object(vllm_runtime.dev_build, "image_exists_locally", image_probe),
    ):
        logs, rc = await _events(
            vllm_runtime.stream_container_up(
                "audit",
                use_dev=True,
                repo_url=f"https://builder:{secret}@git.example/repo.git",
            )
        )

    assert rc == 1
    assert secret not in "\n".join(logs)
    assert "invalid repository URL" in "\n".join(logs)
    image_probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_llamacpp_rejects_credential_repo_before_build_output() -> None:
    secret = "R2_REPO_SECRET"
    stored = SimpleNamespace(name="audit", backend="llamacpp", config_name="audit")
    profile = llamacpp_backend.Profile(
        name="audit", container_name="audit", config_name="audit"
    )
    image_probe = AsyncMock(side_effect=AssertionError("image probe must not run"))
    with (
        patch.object(llamacpp_runtime.profile_store, "load_profile", return_value=stored),
        patch.object(llamacpp_runtime, "validate_common_env", return_value=(True, [])),
        patch.object(llamacpp_runtime, "load_profile", return_value=profile),
        patch.object(
            llamacpp_runtime, "check_port_conflict", AsyncMock(return_value=None)
        ),
        patch.object(
            llamacpp_runtime, "_gpu_conflict_messages", AsyncMock(return_value=[])
        ),
        patch.object(llamacpp_runtime, "_ensure_profile_config", return_value=(True, [])),
        patch.object(llamacpp_runtime.dev_build, "image_exists_locally", image_probe),
    ):
        logs, rc = await _events(
            llamacpp_runtime.stream_container_up(
                "audit",
                use_dev=True,
                repo_url=f"https://builder:{secret}@git.example/repo.git",
            )
        )

    assert rc == 1
    assert secret not in "\n".join(logs)
    assert "invalid repository URL" in "\n".join(logs)
    image_probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_vllm_rejects_credential_profile_image_before_runtime_output() -> None:
    secret = "R2_IMAGE_SECRET"
    profile = VllmProfile(
        name="audit",
        container_name="audit",
        config_name="audit",
        image_tag=f"registry.example/image:v1?token={secret}",
    )
    with (
        patch.object(vllm_runtime.profile_store, "load_profile", return_value=object()),
        patch.object(vllm_runtime, "load_profile", return_value=profile),
    ):
        logs, rc = await _events(vllm_runtime.stream_container_up("audit"))

    assert rc == 1
    assert secret not in "\n".join(logs)
    assert "invalid profile image reference" in "\n".join(logs)


@pytest.mark.asyncio
async def test_llamacpp_rejects_credential_profile_image_before_runtime_output() -> None:
    secret = "R2_IMAGE_SECRET"
    stored = SimpleNamespace(name="audit", backend="llamacpp", config_name="audit")
    profile = llamacpp_backend.Profile(
        name="audit",
        container_name="audit",
        config_name="audit",
        image_tag=f"registry.example/image:v1?token={secret}",
    )
    with (
        patch.object(llamacpp_runtime.profile_store, "load_profile", return_value=stored),
        patch.object(llamacpp_runtime, "validate_common_env", return_value=(True, [])),
        patch.object(llamacpp_runtime, "load_profile", return_value=profile),
        patch.object(
            llamacpp_runtime, "check_port_conflict", AsyncMock(return_value=None)
        ),
        patch.object(
            llamacpp_runtime, "_gpu_conflict_messages", AsyncMock(return_value=[])
        ),
        patch.object(llamacpp_runtime, "_ensure_profile_config", return_value=(True, [])),
    ):
        logs, rc = await _events(llamacpp_runtime.stream_container_up("audit"))

    assert rc == 1
    assert secret not in "\n".join(logs)
    assert "invalid runtime image reference" in "\n".join(logs)


def test_vllm_profile_env_cannot_override_host_control_plane(tmp_path: Path) -> None:
    common = tmp_path / ".env.common"
    common.write_text("HF_CACHE_PATH=/cache\nHF_TOKEN=common-token\n")
    profile_path = tmp_path / "audit.env"
    profile_path.write_text(
        "CONTAINER_NAME=audit\n"
        "VLLM_PORT=18000\n"
        "GPU_ID=0\n"
        "TENSOR_PARALLEL_SIZE=1\n"
        "DOCKER_HOST=tcp://attacker:2375\n"
        "PATH=/attacker\n"
        "HF_TOKEN=profile-token\n"
        "CUSTOM_CONTAINER_VALUE=kept-in-env-file\n"
    )
    profile = VllmProfile(name="audit", config_name="audit")

    with (
        patch.object(vllm_runtime, "COMMON_ENV", common),
        patch.object(vllm_common, "RUNTIME_DIR", tmp_path),
        patch.dict(
            os.environ,
            {"DOCKER_HOST": "unix:///trusted.sock", "PATH": "/trusted"},
            clear=True,
        ),
    ):
        env = vllm_runtime._compose_env(profile, use_dev=False)

    assert env["DOCKER_HOST"] == "unix:///trusted.sock"
    assert env["PATH"] == "/trusted"
    assert env["HF_TOKEN"] == "common-token"
    assert "CUSTOM_CONTAINER_VALUE" not in env
    assert env["CONTAINER_NAME"] == "audit"
    assert env["VLLM_PORT"] == "18000"


def test_llamacpp_profile_env_cannot_override_host_control_plane(tmp_path: Path) -> None:
    common = tmp_path / ".env.common"
    common.write_text("HF_CACHE_PATH=/cache\nHF_TOKEN=common-token\n")
    profile_path = tmp_path / "audit.env"
    profile_path.write_text(
        "CONTAINER_NAME=audit\n"
        "LLAMA_PORT=18000\n"
        "GPU_ID=0\n"
        "CONFIG_NAME=audit\n"
        "DOCKER_CONTEXT=attacker\n"
        "HF_TOKEN=profile-token\n"
        "CUSTOM_CONTAINER_VALUE=kept-in-env-file\n"
    )
    profile = llamacpp_backend.Profile(name="audit", config_name="audit")

    with (
        patch.object(llamacpp_runtime, "COMMON_ENV", common),
        patch.object(llamacpp_backend, "RUNTIME_DIR", tmp_path),
        patch.dict(os.environ, {"DOCKER_CONTEXT": "trusted"}, clear=True),
    ):
        env = llamacpp_runtime._compose_env(profile)

    assert env["DOCKER_CONTEXT"] == "trusted"
    assert env["HF_TOKEN"] == "common-token"
    assert "CUSTOM_CONTAINER_VALUE" not in env
    assert env["CONTAINER_NAME"] == "audit"
    assert env["LLAMA_PORT"] == "18000"


@pytest.mark.asyncio
async def test_vllm_status_preserves_restarting_and_unknown_health() -> None:
    profile = VllmProfile(
        name="audit", container_name="audit", config_name="audit"
    )
    snapshot = common_docker.ContainerSnapshot(
        name="audit",
        lifecycle=common_docker.ContainerLifecycle.RESTARTING,
        health=common_docker.ContainerHealth.UNKNOWN,
        raw_state="restarting",
        raw_status="Restarting (1) 2 seconds ago",
    )
    scan = AsyncMock(return_value={"audit": snapshot})
    with (
        patch.object(vllm_runtime.common_docker, "container_snapshots", scan),
        patch.object(vllm_runtime, "list_profile_names", return_value=["audit"]),
        patch.object(vllm_runtime, "load_profile", return_value=profile),
        patch.object(
            vllm_runtime,
            "load_config",
            return_value=SimpleNamespace(model="org/model"),
        ),
    ):
        statuses = await vllm_runtime.get_container_statuses()

    scan.assert_awaited_once_with(include_stopped=True)
    assert statuses[0].running
    assert statuses[0].status_text == "restarting"
    assert statuses[0].health == "unknown"


@pytest.mark.asyncio
async def test_llamacpp_status_preserves_restarting_and_unknown_health() -> None:
    profile = llamacpp_backend.Profile(
        name="audit", container_name="audit", config_name="audit"
    )
    snapshot = common_docker.ContainerSnapshot(
        name="audit",
        lifecycle=common_docker.ContainerLifecycle.RESTARTING,
        health=common_docker.ContainerHealth.UNKNOWN,
        raw_state="restarting",
        raw_status="Restarting (1) 2 seconds ago",
    )
    scan = AsyncMock(return_value={"audit": snapshot})
    with (
        patch.object(llamacpp_runtime.common_docker, "container_snapshots", scan),
        patch.object(llamacpp_runtime, "list_profile_names", return_value=["audit"]),
        patch.object(llamacpp_runtime, "load_profile", return_value=profile),
        patch.object(
            llamacpp_runtime,
            "load_config",
            return_value=llamacpp_backend.Config(name="audit"),
        ),
    ):
        statuses = await llamacpp_runtime.get_container_statuses()

    scan.assert_awaited_once_with(include_stopped=True)
    assert statuses[0].running
    assert statuses[0].status_text == "restarting"
    assert statuses[0].health == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "profile"),
    [
        (
            vllm_runtime,
            VllmProfile(name="p", container_name="p", config_name="p", port="18000"),
        ),
        (
            llamacpp_runtime,
            llamacpp_backend.Profile(
                name="p", container_name="p", config_name="p", port=18000
            ),
        ),
    ],
)
async def test_non_address_in_use_bind_error_is_a_probe_failure(runtime, profile) -> None:
    socket_instance = SimpleNamespace(
        setsockopt=lambda *_args: None,
        bind=lambda *_args: (_ for _ in ()).throw(PermissionError(errno.EACCES, "denied")),
        close=lambda: None,
    )
    run_name = "run_command" if runtime is vllm_runtime else "_docker_run"

    with (
        patch.object(runtime, run_name, AsyncMock(return_value=(0, ""))),
        patch.object(runtime.socket, "socket", return_value=socket_instance),
    ):
        with pytest.raises(RuntimeError, match="bind probe failed"):
            await runtime.check_port_conflict(profile)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "profile"),
    [
        (
            vllm_runtime,
            VllmProfile(name="p", container_name="p", config_name="p", port="18000"),
        ),
        (
            llamacpp_runtime,
            llamacpp_backend.Profile(
                name="p", container_name="p", config_name="p", port=18000
            ),
        ),
    ],
)
async def test_address_in_use_bind_error_is_a_conflict(runtime, profile) -> None:
    socket_instance = SimpleNamespace(
        setsockopt=lambda *_args: None,
        bind=lambda *_args: (_ for _ in ()).throw(
            OSError(errno.EADDRINUSE, "already in use")
        ),
        close=lambda: None,
    )
    run_name = "run_command" if runtime is vllm_runtime else "_docker_run"

    with (
        patch.object(runtime, run_name, AsyncMock(return_value=(0, ""))),
        patch.object(runtime.socket, "socket", return_value=socket_instance),
    ):
        conflict = await runtime.check_port_conflict(profile)

    assert conflict is not None
    assert "18000" in conflict


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "run_name"),
    [(vllm_runtime, "run_command"), (llamacpp_runtime, "_docker_run")],
)
async def test_du_failure_is_not_folded_into_missing_progress(runtime, run_name) -> None:
    with patch.object(
        runtime,
        run_name,
        AsyncMock(return_value=(1, "du: permission denied")),
    ):
        with pytest.raises(RuntimeError, match="permission denied"):
            await runtime._dir_size_bytes("/cache")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "profile", "run_name"),
    [
        (
            vllm_runtime,
            VllmProfile(name="p", container_name="p", config_name="p", port="18000"),
            "run_command",
        ),
        (
            llamacpp_runtime,
            llamacpp_backend.Profile(
                name="p", container_name="p", config_name="p", port=18000
            ),
            "_docker_run",
        ),
    ],
)
async def test_progress_log_failure_ends_readiness_with_explicit_error(
    runtime, profile, run_name
) -> None:
    async def run(*args, **_kwargs):
        if args[:2] == ("docker", "inspect"):
            return 0, "running\tstarting"
        if args[:2] == ("docker", "logs"):
            return 1, "daemon unavailable"
        return 0, ""

    with (
        patch.object(runtime, run_name, side_effect=run),
        patch.object(runtime, "_models_endpoint_ready", AsyncMock(return_value=False)),
    ):
        ok, messages = await _validation_result(
            runtime._post_start_validation(
                profile,
                timeout=0.1,
                poll_interval=0.01,
                hf_cache_path=None,
            )
        )

    assert not ok
    assert "daemon unavailable" in "\n".join(messages)
    assert "progress probe" in "\n".join(messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("download", ["vllm", "llamacpp"])
async def test_prepare_refuses_to_start_when_stale_cleanup_fails(
    download: str, tmp_path: Path
) -> None:
    run = AsyncMock(return_value=(1, "permission denied"))

    async def stream_ok(*_args, **_kwargs):
        yield "rc", 0

    with (
        patch.object(prepare, "_run", run),
        patch.object(prepare, "stream_lines", side_effect=stream_ok),
    ):
        if download == "vllm":
            logs, rc = await _events(
                prepare.stream_vllm_download(
                    image_ref="vllm/vllm-openai:v1",
                    model_id="org/model",
                    cache_path=str(tmp_path),
                    token="",
                    container_name="prepare-p",
                )
            )
        else:
            with (
                patch.object(prepare, "downloader_image", return_value="repo/image:v1"),
                patch.object(prepare, "image_present", AsyncMock(return_value=True)),
            ):
                logs, rc = await _events(
                    prepare.stream_llamacpp_download(
                        hf_repo="org/repo",
                        hf_file="model.gguf",
                        cache_path=str(tmp_path),
                        token="",
                        container_name="prepare-p",
                    )
                )

    assert rc == 1
    assert "cleanup" in "\n".join(logs).lower()
    assert "permission denied" in "\n".join(logs)


@pytest.mark.asyncio
async def test_missing_prepare_container_is_a_successful_cleanup() -> None:
    with patch.object(
        prepare,
        "_run",
        AsyncMock(return_value=(1, "Error response from daemon: No such container: p")),
    ):
        rc, message = await prepare._remove_prepare_container("p")

    assert rc == 0
    assert message == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("download", ["vllm", "llamacpp"])
async def test_cancelled_prepare_surfaces_cleanup_failure(
    download: str, tmp_path: Path
) -> None:
    async def cancelled_stream(*_args, **_kwargs):
        raise asyncio.CancelledError
        yield

    cleanup = AsyncMock(side_effect=[(0, ""), (1, "daemon denied")])
    with (
        patch.object(prepare, "_remove_prepare_container", cleanup),
        patch.object(prepare, "stream_lines", side_effect=cancelled_stream),
    ):
        if download == "vllm":
            stream = prepare.stream_vllm_download(
                image_ref="vllm/vllm-openai:v1",
                model_id="org/model",
                cache_path=str(tmp_path),
                token="",
                container_name="prepare-p",
            )
        else:
            with (
                patch.object(prepare, "downloader_image", return_value="repo/image:v1"),
                patch.object(prepare, "image_present", AsyncMock(return_value=True)),
            ):
                stream = prepare.stream_llamacpp_download(
                    hf_repo="org/repo",
                    hf_file="model.gguf",
                    cache_path=str(tmp_path),
                    token="",
                    container_name="prepare-p",
                )
                with pytest.raises(RuntimeError, match="cleanup failed"):
                    await _events(stream)
                return

        with pytest.raises(RuntimeError, match="cleanup failed"):
            await _events(stream)


@pytest.mark.asyncio
async def test_vllm_down_uses_manual_cleanup_when_compose_env_fails() -> None:
    profile = VllmProfile(name="p", container_name="p", config_name="p")
    commands: list[tuple[str, ...]] = []

    async def run(*args, **_kwargs):
        commands.append(tuple(args))
        if args[:2] == ("docker", "inspect"):
            return 0, "vllm/vllm-openai:v1"
        return 0, ""

    with (
        patch.object(vllm_runtime.profile_store, "load_profile", return_value=object()),
        patch.object(
            vllm_runtime.profile_store,
            "render_env_for_profile",
            return_value=Path("p.env"),
        ),
        patch.object(vllm_runtime, "load_profile", return_value=profile),
        patch.object(vllm_runtime, "_container_exists", AsyncMock(return_value=True)),
        patch.object(vllm_runtime, "_compose_env", side_effect=OSError("env denied")),
        patch.object(vllm_runtime, "run_command", side_effect=run),
        patch.object(
            vllm_runtime, "_remove_compose_network", AsyncMock(return_value=(0, ""))
        ),
    ):
        rc, message = await vllm_runtime.container_down("p")

    assert rc == 0
    assert any(command[:2] == ("docker", "stop") for command in commands)
    assert any(command[:2] == ("docker", "rm") for command in commands)
    assert "env denied" in message


@pytest.mark.asyncio
async def test_llamacpp_down_uses_manual_cleanup_when_compose_env_fails(
    tmp_path: Path,
) -> None:
    profile = llamacpp_backend.Profile(name="p", container_name="p", config_name="p")
    override = tmp_path / "override-p.yaml"
    override.write_text("services: {}\n")
    rm = AsyncMock(return_value=(0, ""))

    with (
        patch.object(
            llamacpp_runtime.profile_store,
            "load_profile",
            return_value=SimpleNamespace(name="p", backend="llamacpp"),
        ),
        patch.object(
            llamacpp_runtime.profile_store,
            "render_env_for_profile",
            return_value=tmp_path / "p.env",
            create=True,
        ),
        patch.object(llamacpp_runtime, "load_profile", return_value=profile),
        patch.object(llamacpp_runtime, "_override_path", return_value=override),
        patch.object(llamacpp_runtime, "_compose_env", side_effect=OSError("env denied")),
        patch.object(llamacpp_runtime, "_container_exists", AsyncMock(return_value=True)),
        patch.object(llamacpp_runtime, "_docker_run", rm),
        patch.object(
            llamacpp_runtime, "_remove_compose_network", AsyncMock(return_value=(0, ""))
        ),
    ):
        rc, message = await llamacpp_runtime.container_down("p")

    assert rc == 0
    rm.assert_awaited_once()
    assert "env denied" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", [vllm_runtime, llamacpp_runtime])
async def test_down_skips_compose_when_runtime_env_render_fails(
    runtime, tmp_path: Path
) -> None:
    if runtime is vllm_runtime:
        profile = VllmProfile(name="p", container_name="p", config_name="p")
        run_name = "run_command"
        render_message = "could not render runtime env"
    else:
        profile = llamacpp_backend.Profile(
            name="p", container_name="p", config_name="p"
        )
        run_name = "_docker_run"
        render_message = "env 렌더 실패"

    async def run(*args, **_kwargs):
        if args[:2] == ("docker", "inspect"):
            return 0, "repo/image:v1"
        return 0, ""

    override = tmp_path / "override-p.yaml"
    override.write_text("services: {}\n")
    with (
        patch.object(
            runtime.profile_store,
            "load_profile",
            return_value=SimpleNamespace(name="p", backend="llamacpp"),
        ),
        patch.object(
            runtime.profile_store,
            "render_env_for_profile",
            side_effect=OSError("artifact denied"),
        ) as render,
        patch.object(runtime, "load_profile", return_value=profile),
        patch.object(runtime, "_container_exists", AsyncMock(return_value=True)),
        patch.object(runtime, "_compose_env") as compose_env,
        patch.object(runtime, run_name, side_effect=run),
        patch.object(
            runtime, "_remove_compose_network", AsyncMock(return_value=(0, ""))
        ),
        patch.object(
            llamacpp_runtime,
            "_override_path",
            return_value=override,
        ),
    ):
        rc, message = await runtime.container_down("p")

    assert rc == 0
    render.assert_called_once()
    compose_env.assert_not_called()
    assert render_message in message
    assert "artifact denied" in message


@pytest.mark.asyncio
async def test_vllm_down_skips_compose_when_image_probe_fails() -> None:
    profile = VllmProfile(name="p", container_name="p", config_name="p")
    commands: list[tuple[str, ...]] = []

    async def run(*args, **_kwargs):
        commands.append(tuple(args))
        if args[:2] == ("docker", "inspect"):
            return 1, "daemon unavailable"
        return 0, ""

    with (
        patch.object(vllm_runtime.profile_store, "load_profile", return_value=object()),
        patch.object(
            vllm_runtime.profile_store,
            "render_env_for_profile",
            return_value=Path("p.env"),
        ),
        patch.object(vllm_runtime, "load_profile", return_value=profile),
        patch.object(vllm_runtime, "_container_exists", AsyncMock(return_value=True)),
        patch.object(vllm_runtime, "_compose_env") as compose_env,
        patch.object(vllm_runtime, "run_command", side_effect=run),
        patch.object(
            vllm_runtime, "_remove_compose_network", AsyncMock(return_value=(0, ""))
        ),
    ):
        rc, message = await vllm_runtime.container_down("p")

    assert rc == 0
    compose_env.assert_not_called()
    assert any(command[:2] == ("docker", "stop") for command in commands)
    assert "image probe failed" in message
    assert "daemon unavailable" in message


def test_llamacpp_override_is_atomic_owner_only_and_transactional(
    tmp_path: Path,
) -> None:
    renderer = _load_renderer()
    config_dir = tmp_path / "config"
    runtime_dir = tmp_path / "runtime"
    config_dir.mkdir()
    runtime_dir.mkdir()
    (config_dir / "p.yaml").write_text("api-key: secret\n")
    target = runtime_dir / "override-p.yaml"
    target.write_text("old\n")
    target.chmod(0o644)
    entered: list[bool] = []

    @contextmanager
    def transaction():
        entered.append(True)
        yield

    stored = SimpleNamespace(
        config_name="p",
        hf_repo="org/repo",
        hf_file="model.gguf",
    )
    with (
        patch.object(renderer, "CONFIG_DIR", config_dir),
        patch.object(renderer, "RUNTIME_DIR", runtime_dir),
        patch.object(renderer.profile_store, "load_profile", return_value=stored),
            patch.object(
                renderer.profile_store,
                "storage_transaction",
                transaction,
                create=True,
            ),
            patch.object(renderer.profile_store, "render_env", return_value=Path("p.env")),
            patch.object(renderer.sys, "argv", ["render-override.py", "p"]),
    ):
        assert renderer.main() == 0

    assert entered == [True]
    assert target.stat().st_mode & 0o777 == 0o600
    assert "secret" in target.read_text()
