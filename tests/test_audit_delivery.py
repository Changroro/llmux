from __future__ import annotations

import asyncio
import builtins
import contextlib
import re
import ssl
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import certifi
import pytest
from textual.widgets import Input
from typer.testing import CliRunner

from tui.app import LlmuxApp
from tui.common.adapter import DashboardRow
from tui.screens.dashboard import DashboardScreen
from tui.screens.too_narrow import TooNarrowScreen


WIDE = (120, 30)
NARROW = (60, 30)


def _row(backend: str, name: str, *, running: bool = False, gpu_id: str = "0"):
    return DashboardRow(
        backend=backend,
        profile_name=name,
        container_name=name,
        port=8000 if backend == "vllm" else 8080,
        running=running,
        model=f"org/{name}",
        detail="",
        gpu_id=gpu_id,
    )


@contextlib.contextmanager
def _dashboard_patches():
    with (
        patch(
            "tui.common.docker.running_container_names",
            AsyncMock(return_value=set()),
        ),
        patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[])),
        patch("tui.screens.dashboard.VllmAdapter.rows", return_value=[]),
        patch("tui.screens.dashboard.LlamacppAdapter.rows", return_value=[]),
    ):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "action"),
    (("vllm", "clone"), ("llamacpp", "clone-profile")),
)
async def test_dashboard_clone_action_runs_callback_for_both_backends(
    backend: str, action: str
) -> None:
    source = _row(backend, f"{backend}-source")
    clone = SimpleNamespace(name=f"{backend}-copy")
    with (
        _dashboard_patches(),
        patch("tui.screens.dashboard.clone_profile", return_value=clone, create=True) as copy,
    ):
        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            dashboard = next(
                screen
                for screen in app.screen_stack
                if isinstance(screen, DashboardScreen)
            )
            dashboard._after_mutation = MagicMock()
            if backend == "vllm":
                dashboard._dispatch_vllm(action, source)
            else:
                dashboard._dispatch_llamacpp(
                    action,
                    source,
                    SimpleNamespace(name=source.profile_name),
                )
            await pilot.pause()
            app.screen.query_one("#prompt-input", Input).value = clone.name
            app.screen._submit()
            await pilot.pause()

            copy.assert_called_once_with(source.profile_name, clone.name, backend)
            dashboard._after_mutation.assert_called_once_with(clone.name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "action"),
    (("vllm", "render_env"), ("llamacpp", "render-env")),
)
async def test_dashboard_render_env_action_targets_selected_backend(
    tmp_path: Path, backend: str, action: str
) -> None:
    row = _row(backend, f"{backend}-profile")
    profile = SimpleNamespace(name=row.profile_name, backend=backend)
    rendered = tmp_path / ".runtime" / backend / f"{row.profile_name}.env"
    with (
        _dashboard_patches(),
        patch("tui.screens.dashboard.load_profile", return_value=profile, create=True) as load,
        patch(
            "tui.screens.dashboard.render_env_for_profile", return_value=rendered
        ) as render,
    ):
        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            dashboard = next(
                screen
                for screen in app.screen_stack
                if isinstance(screen, DashboardScreen)
            )
            dashboard.notify = MagicMock()
            if backend == "vllm":
                dashboard._dispatch_vllm(action, row)
            else:
                dashboard._dispatch_llamacpp(action, row, profile)

            load.assert_called_once_with(row.profile_name, backend)
            render.assert_called_once_with(profile.name, profile.backend)
            assert str(rendered) in str(dashboard.notify.call_args.args[0])


@pytest.mark.asyncio
async def test_partial_backend_scan_preserves_rows_and_blocks_start() -> None:
    vllm = _row("vllm", "target", gpu_id="0")
    llama = _row("llamacpp", "last-known", running=True, gpu_id="1")
    llama_rows = MagicMock(return_value=[llama])
    with (
        patch(
            "tui.common.docker.running_container_names",
            AsyncMock(return_value={"last-known"}),
        ),
        patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[])),
        patch("tui.common.docker.running_container_ports", AsyncMock(return_value={})),
        patch("tui.screens.dashboard.VllmAdapter.rows", return_value=[vllm]),
        patch("tui.screens.dashboard.LlamacppAdapter.rows", llama_rows),
    ):
        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            dashboard = next(
                screen
                for screen in app.screen_stack
                if isinstance(screen, DashboardScreen)
            )
            llama_rows.side_effect = RuntimeError("profile scan failed")
            dashboard._reload()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert {(row.backend, row.profile_name) for row in dashboard._rows} == {
                ("vllm", "target"),
                ("llamacpp", "last-known"),
            }
            assert dashboard._scan_errors == {"llamacpp"}

            launch = MagicMock()
            dashboard._check_and_confirm(vllm, launch)
            await app.workers.wait_for_complete()
            await pilot.pause()

            launch.assert_not_called()
            assert type(app.screen).__name__ == "ConfirmModal"
            assert app.screen._variant == "error"


@pytest.mark.asyncio
async def test_gpu_probe_failure_preserves_last_verified_state() -> None:
    gpu = SimpleNamespace(index=0)
    gpu_probe = AsyncMock(return_value=[gpu])
    with (
        patch("tui.common.docker.running_container_names", AsyncMock(return_value=set())),
        patch("tui.common.docker.get_gpu_info", gpu_probe),
        patch("tui.common.docker.format_gpu_bar", return_value="GPU0 verified"),
        patch("tui.screens.dashboard.VllmAdapter.rows", return_value=[]),
        patch("tui.screens.dashboard.LlamacppAdapter.rows", return_value=[]),
    ):
        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            dashboard = next(
                screen
                for screen in app.screen_stack
                if isinstance(screen, DashboardScreen)
            )
            assert dashboard._gpus == [gpu]

            gpu_probe.side_effect = RuntimeError("nvidia-smi query failed")
            dashboard._poll_gpu()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert dashboard._gpus == [gpu]
            assert dashboard._gpu_scan_error == "nvidia-smi query failed"
            gpu_bar = str(dashboard.query_one("#gpu-bar").render())
            assert "GPU status unavailable" in gpu_bar
            assert "GPU0 verified" in gpu_bar


@pytest.mark.asyncio
async def test_vllm_model_discovery_failure_is_reported_by_benchmark() -> None:
    row = _row("vllm", "bench-target")
    with (
        _dashboard_patches(),
        patch("tui.common.http.profile_api_key", AsyncMock(return_value="")),
        patch(
            "tui.screens.dashboard.list_served_models",
            AsyncMock(side_effect=RuntimeError("invalid /v1/models response")),
        ),
        patch("tui.screens.dashboard.run_bench", AsyncMock()) as bench,
    ):
        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            dashboard = next(
                screen
                for screen in app.screen_stack
                if isinstance(screen, DashboardScreen)
            )
            dashboard.notify = MagicMock()
            dashboard._run_vllm_bench(row)
            await app.workers.wait_for_complete()

            messages = " ".join(
                str(call.args[0]) for call in dashboard.notify.call_args_list
            )
            assert "invalid /v1/models response" in messages
            bench.assert_not_awaited()


def test_apply_update_reports_refresh_failures_as_partial_failures() -> None:
    from tui.common import version_check as vc

    with patch.object(vc, "_git", return_value=(0, "")), patch.object(
        vc.shutil, "which", return_value=None
    ):
        ok, message = vc.apply_update("v9.9.9")
    assert ok is False
    assert "checkout" in message.lower()
    assert "uv" in message

    failed = SimpleNamespace(returncode=1, stdout="", stderr="refresh failed")
    with (
        patch.object(vc, "_git", return_value=(0, "")),
        patch.object(vc.shutil, "which", return_value="/usr/bin/uv"),
        patch.object(vc.subprocess, "run", return_value=failed),
    ):
        ok, message = vc.apply_update("v9.9.9")
    assert ok is False
    assert "checkout" in message.lower()
    assert "refresh" in message.lower()


def test_cli_update_refresh_failure_exits_one() -> None:
    from tui.cli import app
    from tui.common import version_check as vc

    status = vc.UpdateStatus(vc.BEHIND, tag="v9.9.9", url="u", local_version="2.8.0")
    with (
        patch.object(vc, "resolve_status", return_value=status),
        patch.object(vc, "update_blocked_reason", return_value=""),
        patch.object(
            vc,
            "apply_update",
            return_value=(False, "Checkout updated, but tool refresh failed."),
        ),
    ):
        result = CliRunner().invoke(app, ["update", "--yes"])

    assert result.exit_code == 1
    assert "refresh failed" in result.output


@pytest.mark.asyncio
async def test_tui_update_refresh_failure_does_not_exit() -> None:
    from tui.common import version_check as vc

    with (
        _dashboard_patches(),
        patch.object(vc, "apply_update", return_value=(False, "failed")),
    ):
        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            dashboard = next(
                screen
                for screen in app.screen_stack
                if isinstance(screen, DashboardScreen)
            )
            dashboard.notify = MagicMock()
            with patch.object(app, "exit") as exit_app:
                dashboard._apply_update("v9.9.9")
                await app.workers.wait_for_complete()
            exit_app.assert_not_called()


def test_source_checkout_version_does_not_require_tomllib(tmp_path: Path) -> None:
    from tui import cli
    from tui.common import profile_store

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "llmux"\nversion = "9.8.7"\n'
    )
    real_import = builtins.__import__

    def import_without_tomllib(name, *args, **kwargs):
        if name == "tomllib":
            raise ModuleNotFoundError("tomllib is unavailable on Python 3.10")
        return real_import(name, *args, **kwargs)

    with (
        patch("importlib.metadata.version", side_effect=PackageNotFoundError),
        patch.object(profile_store, "PROJECT_ROOT", tmp_path),
        patch.object(builtins, "__import__", side_effect=import_without_tomllib),
    ):
        assert cli.llmux_version() == "9.8.7"


def test_invalid_existing_env_still_needs_onboarding(tmp_path: Path) -> None:
    from tui.common import onboarding

    env_path = tmp_path / ".env.common"
    env_path.write_text("HF_CACHE_PATH=relative\n")
    with patch.object(onboarding, "COMMON_ENV", env_path):
        assert onboarding.needs_onboarding() is True


def test_onboarding_surfaces_cleanup_failure_and_retries(tmp_path: Path) -> None:
    from tui.common import onboarding

    env_path = tmp_path / ".env.common"
    example = tmp_path / ".env.common.example"
    example.write_text("HF_CACHE_PATH=\nMODEL_DIR=\nHF_TOKEN=\n")
    console = MagicMock()
    answers = [str(tmp_path / "cache"), str(tmp_path / "models"), ""]

    def write_invalid(_content: str) -> None:
        env_path.write_text("HF_CACHE_PATH=relative\n")

    with (
        patch.object(onboarding, "COMMON_ENV", env_path),
        patch.object(onboarding, "COMMON_ENV_EXAMPLE", example),
        patch.object(onboarding, "_write_common_env", side_effect=write_invalid),
        patch.object(
            onboarding,
            "validate_common_env",
            return_value=(False, ["Error: invalid generated config"]),
        ),
        patch("rich.console.Console", return_value=console),
        patch("rich.prompt.Prompt.ask", side_effect=answers),
        patch.object(Path, "unlink", side_effect=PermissionError("cleanup denied")),
    ):
        assert onboarding.run_onboarding() is False
        assert onboarding.needs_onboarding() is True

    output = " ".join(str(call.args[0]) for call in console.print.call_args_list)
    assert "cleanup denied" in output
    assert env_path.exists()


@pytest.mark.asyncio
async def test_late_update_modal_cannot_cover_width_guard() -> None:
    from tui.common import version_check as vc

    started = asyncio.Event()
    release = asyncio.Event()
    status = vc.UpdateStatus(vc.BEHIND, tag="v9.9.9", url="u")

    async def delayed_to_thread(function, *args, **kwargs):
        started.set()
        await release.wait()
        return function(*args, **kwargs)

    with (
        _dashboard_patches(),
        patch.object(vc, "resolve_status", return_value=status),
        patch.object(vc, "update_blocked_reason", return_value=""),
        patch("tui.screens.dashboard.asyncio.to_thread", side_effect=delayed_to_thread),
    ):
        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            dashboard = next(
                screen
                for screen in app.screen_stack
                if isinstance(screen, DashboardScreen)
            )
            dashboard.action_check_update()
            await started.wait()
            await pilot.resize_terminal(*NARROW)
            await pilot.pause()
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert isinstance(app.screen, TooNarrowScreen)
            assert "ConfirmModal" not in {
                type(screen).__name__ for screen in app.screen_stack
            }


def _pem_certificates(bundle: str) -> list[str]:
    return [
        match + "\n"
        for match in re.findall(
            r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
            bundle,
            re.DOTALL,
        )
    ]


def test_ssl_context_combines_system_and_certifi_ca_files(tmp_path: Path) -> None:
    from tui.common import ssl_ctx

    certificates = _pem_certificates(Path(certifi.where()).read_text())
    assert len(certificates) >= 2
    system_ca = tmp_path / "system-ca.pem"
    certifi_ca = tmp_path / "certifi-ca.pem"
    system_ca.write_text(certificates[0])
    certifi_ca.write_text(certificates[1])
    empty_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    empty_context.check_hostname = True
    empty_context.verify_mode = ssl.CERT_REQUIRED

    with (
        patch.object(ssl_ctx, "_cached", None),
        patch.object(ssl_ctx, "_CA_CANDIDATES", (str(system_ca),)),
        patch.object(ssl_ctx.ssl, "create_default_context", return_value=empty_context),
        patch.object(certifi, "where", return_value=str(certifi_ca)),
    ):
        context = ssl_ctx.get_ssl_context()

    loaded = set(context.get_ca_certs(binary_form=True))
    expected = {
        ssl.PEM_cert_to_DER_cert(certificates[0]),
        ssl.PEM_cert_to_DER_cert(certificates[1]),
    }
    assert expected <= loaded


def test_docs_and_ci_publish_current_delivery_contracts() -> None:
    root = Path(__file__).parents[1]
    vllm_docs = (root / "docs/backends/vllm.html").read_text()
    profiles_docs = (root / "docs/guide/profiles.html").read_text()
    cli_docs = (root / "docs/reference/cli.html").read_text()
    tui_docs = (root / "docs/guide/tui.html").read_text()
    ci = (root / ".github/workflows/ci.yml").read_text()

    assert "./config/vllm:/config/:ro" in vllm_docs
    assert "^[a-z0-9][a-z0-9_-]*$" in profiles_docs
    assert "<code>prepare</code>" in cli_docs
    assert "<code>update</code>" in cli_docs
    assert "<code>--max-workers</code>" in cli_docs
    assert "<code>llmux bench</code>" in cli_docs
    assert "<code>llmux stats</code>" in cli_docs
    assert "Profile names are globally unique" in cli_docs
    assert "Clone Profile" in tui_docs
    assert "Render Runtime Env" in tui_docs
    assert "warmup" in tui_docs and "3 measured" in tui_docs
    assert "export CONTAINER_NAME=test CONFIG_NAME=test" in ci
    assert 'python-version: ["3.10", "3.13"]' in ci
    assert 'env -u "$variable" docker compose' in ci
    assert len(re.findall(r"^\s+expect_missing ", ci, re.MULTILINE)) == 16
    for variable in (
        "CONTAINER_NAME",
        "LLAMA_PORT",
        "LLAMACPP_DEV_TAG",
        "VLLM_PORT",
        "VLLM_DEV_TAG",
        "HF_CACHE_PATH",
        "CONFIG_NAME",
        "GPU_ID",
        "PROFILE_PATH",
        "TENSOR_PARALLEL_SIZE",
        "LORA_BASE_PATH",
    ):
        assert re.search(
            rf"^\s+expect_missing \S+:{variable} {variable}(?: |$)",
            ci,
            re.MULTILINE,
        )
