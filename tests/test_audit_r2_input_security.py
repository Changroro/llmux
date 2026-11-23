from __future__ import annotations

import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner


class ReferenceValidationTests(unittest.TestCase):
    def test_repo_url_error_rejects_credential_components_without_echoing_them(self) -> None:
        from tui.common import dev_build

        secret = "LLMUX_R2_GIT_SENTINEL"
        rejected = (
            f"https://builder:{secret}@git.example/org/repo.git",
            f"https://git.example/org/repo.git?token={secret}",
            f"https://git.example/org/repo.git#{secret}",
        )
        for value in rejected:
            with self.subTest(value=value):
                error = dev_build.repo_url_error(value)
                self.assertTrue(error)
                self.assertNotIn(secret, error)

        self.assertEqual(
            dev_build.repo_url_error("https://git.example/org/repo.git"), ""
        )
        self.assertEqual(
            dev_build.repo_url_error("ssh://git@git.example/org/repo.git"), ""
        )
        self.assertEqual(dev_build.repo_url_error("git@git.example:org/repo.git"), "")

    def test_image_tag_error_rejects_credential_components_without_echoing_them(self) -> None:
        from tui.common import dev_build

        secret = "LLMUX_R2_IMAGE_SENTINEL"
        rejected = (
            f"https://builder:{secret}@registry.example/org/image:v1",
            f"builder:{secret}@registry.example/org/image:v1",
            f"registry.example/org/image:v1?token={secret}",
            f"registry.example/org/image:v1#{secret}",
        )
        for value in rejected:
            with self.subTest(value=value):
                error = dev_build.image_tag_error(value)
                self.assertTrue(error)
                self.assertNotIn(secret, error)

        self.assertEqual(
            dev_build.image_tag_error("registry.example/org/image:v1"), ""
        )


class CliReferenceBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_image_commands_reject_credentials_before_runtime_calls(self) -> None:
        from tui.cli import image

        secret = "LLMUX_R2_IMAGE_CLI_SENTINEL"
        cases = (
            ["list", "--repo", f"builder:{secret}@registry.example/org/image"],
            ["pull", "v1", "--repo", f"builder:{secret}@registry.example/org/image"],
            ["remove", f"builder:{secret}@registry.example/org/image:v1"],
        )
        def reject_runtime(awaitable):
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise AssertionError("runtime called")

        for args in cases:
            with self.subTest(args=args), patch.object(
                image, "run_async", side_effect=reject_runtime
            ):
                result = self.runner.invoke(image.app, args)
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertNotIn(secret, result.output)

    def test_build_dev_rejects_credentials_before_backend_stream(self) -> None:
        from tui.cli import image

        secret = "LLMUX_R2_BUILD_CLI_SENTINEL"
        with patch.object(
            image, "stream_async", side_effect=AssertionError("backend stream called")
        ):
            result = self.runner.invoke(
                image.app,
                [
                    "build-dev",
                    "--backend",
                    "vllm",
                    "--repo-url",
                    f"https://builder:{secret}@git.example/org/repo.git",
                ],
            )

        self.assertEqual(result.exit_code, 2, result.output)
        self.assertNotIn(secret, result.output)

    def test_container_up_rejects_credentials_before_profile_or_docker_probes(self) -> None:
        from tui.cli import container

        secret = "LLMUX_R2_UP_CLI_SENTINEL"
        with patch.object(
            container, "detect_backend", side_effect=AssertionError("profile probe called")
        ):
            result = self.runner.invoke(
                container.app,
                [
                    "up",
                    "audit",
                    "--dev",
                    "--repo-url",
                    f"https://git.example/org/repo.git?token={secret}",
                ],
            )

        self.assertEqual(result.exit_code, 2, result.output)
        self.assertNotIn(secret, result.output)


class StatsMetricsContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_profile_metrics_failures_are_reported_and_exit_nonzero(self) -> None:
        from tui.backends.vllm import backend_runtime
        from tui.cli import container
        from tui.common import metrics

        statuses = [
            SimpleNamespace(running=True, port=8101, profile_name="alpha"),
            SimpleNamespace(running=True, port=8102, profile_name="beta"),
        ]

        async def unavailable(port):
            raise metrics.MetricsUnavailableError(f"endpoint {port} unavailable")

        with patch.object(
            backend_runtime,
            "get_container_statuses",
            AsyncMock(return_value=statuses),
        ), patch.object(metrics, "fetch_token_counters", side_effect=unavailable):
            result = self.runner.invoke(
                container.app,
                ["stats", "--backend", "vllm", "--once", "--interval", "0.001"],
            )

        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("vllm/alpha", result.output)
        self.assertIn("vllm/beta", result.output)
        self.assertIn("endpoint 8101 unavailable", result.output)
        self.assertIn("endpoint 8102 unavailable", result.output)

    def test_missing_counter_family_remains_successful_na(self) -> None:
        from tui.backends.vllm import backend_runtime
        from tui.cli import container
        from tui.common import metrics

        statuses = [SimpleNamespace(running=True, port=8101, profile_name="alpha")]
        with patch.object(
            backend_runtime,
            "get_container_statuses",
            AsyncMock(return_value=statuses),
        ), patch.object(
            metrics, "fetch_token_counters", AsyncMock(return_value=None)
        ):
            result = self.runner.invoke(
                container.app,
                ["stats", "--backend", "vllm", "--once", "--interval", "0.001"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("alpha (vllm, port 8101): n/a", result.output)

    def test_zero_counters_remain_real_zero_rates(self) -> None:
        from tui.backends.vllm import backend_runtime
        from tui.cli import container
        from tui.common import metrics

        statuses = [SimpleNamespace(running=True, port=8101, profile_name="alpha")]
        with patch.object(
            backend_runtime,
            "get_container_statuses",
            AsyncMock(return_value=statuses),
        ), patch.object(
            metrics, "fetch_token_counters", AsyncMock(return_value=(0.0, 0.0))
        ):
            result = self.runner.invoke(
                container.app,
                ["stats", "--backend", "vllm", "--once", "--interval", "0.001"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("prompt 0.0 tok/s", result.output)
        self.assertIn("gen 0.0 tok/s", result.output)


class ContainerStatusConsumerTests(unittest.TestCase):
    def test_ps_preserves_restarting_status_and_running_flag(self) -> None:
        from tui.backends.vllm import backend_runtime
        from tui.cli import container

        status = SimpleNamespace(
            profile_name="alpha",
            container_name="alpha-container",
            status_text="restarting",
            running=True,
            health="",
            port="8101",
            gpu_id="0",
            model="model",
        )
        with patch.object(
            backend_runtime,
            "get_container_statuses",
            AsyncMock(return_value=[status]),
        ), patch.object(
            container.common_docker,
            "container_images",
            AsyncMock(return_value={"alpha-container": "vllm/vllm-openai:v1"}),
        ), patch("tui.common.readiness.probe_served", AsyncMock(return_value=("loading", [], ""))):
            result = CliRunner().invoke(
                container.app, ["ps", "--backend", "vllm", "--json"]
            )

        self.assertEqual(result.exit_code, 0, result.output)
        rows = json.loads(result.output)
        self.assertEqual(rows[0]["status"], "restarting")
        self.assertIs(rows[0]["running"], True)
        self.assertEqual(rows[0]["image"], "vllm/vllm-openai:v1")
        self.assertEqual(rows[0]["port"], 8101)
        self.assertIs(rows[0]["ready"], False)


class RenderEnvSnapshotContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_single_profile_uses_latest_profile_renderer(self) -> None:
        from tui.cli import container

        output = Path("/runtime/vllm/alpha.env")
        with patch.object(
            container, "detect_backend", return_value="vllm"
        ), patch.object(
            container.profile_store,
            "load_profile",
            side_effect=AssertionError("snapshot loaded"),
        ), patch.object(
            container.profile_store,
            "render_env",
            side_effect=AssertionError("snapshot rendered"),
        ), patch.object(
            container.profile_store,
            "render_env_for_profile",
            return_value=output,
        ) as latest_renderer:
            result = self.runner.invoke(container.app, ["render-env", "alpha"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.strip(), str(output))
        latest_renderer.assert_called_once_with("alpha", "vllm")

    def test_batch_render_uses_latest_profile_renderer(self) -> None:
        from tui.cli import container

        profiles = [SimpleNamespace(name="alpha", backend="vllm")]
        output = Path("/runtime/vllm/alpha.env")
        with patch.object(
            container.profile_store, "list_profiles", return_value=profiles
        ), patch.object(
            container.profile_store,
            "render_env",
            side_effect=AssertionError("snapshot rendered"),
        ), patch.object(
            container.profile_store,
            "render_env_for_profile",
            return_value=output,
        ) as latest_renderer:
            result = self.runner.invoke(
                container.app, ["render-env", "--backend", "vllm"]
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.strip(), str(output))
        latest_renderer.assert_called_once_with("alpha", "vllm")


class WidgetReferenceBoundaryTests(unittest.TestCase):
    def test_dev_build_modal_does_not_store_credential_bearing_default(self) -> None:
        from tui.common.widgets import DevBuildPromptModal

        secret = "LLMUX_R2_WIDGET_SENTINEL"
        modal = DevBuildPromptModal(
            "vllm",
            f"https://builder:{secret}@git.example/org/repo.git",
            "main",
        )

        self.assertNotIn(secret, repr(modal.__dict__))
        self.assertEqual(modal._repo_url, "")
        self.assertTrue(modal._repo_url_error)


class WidgetReferencePilotTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_clears_rejected_repository_value(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Input, Static

        from tui.common.widgets import DevBuildPromptModal

        secret = "LLMUX_R2_WIDGET_SUBMIT_SENTINEL"
        modal = DevBuildPromptModal(
            "vllm", "https://git.example/org/repo.git", "main"
        )

        class Host(App):
            def compose(self) -> ComposeResult:
                yield Static()

            def on_mount(self) -> None:
                self.push_screen(modal)

        async with Host().run_test() as pilot:
            repo_input = modal.query_one("#build-repo-input", Input)
            repo_input.value = (
                f"https://git.example/org/repo.git?token={secret}"
            )
            await pilot.click("#build-submit")

            self.assertEqual(repo_input.value, "")
            self.assertIs(modal.app.screen, modal)


class VersionComparisonFailureTests(unittest.TestCase):
    def test_cat_file_operational_failures_are_unknown(self) -> None:
        from tui.common import version_check as vc

        cases = (
            (-1, ""),
            (1, "permission denied"),
            (128, "fatal: unable to read object database"),
        )
        for result in cases:
            with self.subTest(result=result), patch.object(
                vc,
                "_git",
                side_effect=(result, (0, "false\n")),
            ):
                self.assertIsNone(vc._is_behind("deadbeef"))

    def test_missing_commit_is_unknown_when_shallow_probe_fails(self) -> None:
        from tui.common import version_check as vc

        probes = ((-1, ""), (1, "permission denied"), (0, ""))
        for shallow_probe in probes:
            with self.subTest(shallow_probe=shallow_probe), patch.object(
                vc,
                "_git",
                side_effect=(
                    (128, "fatal: Not a valid object name deadbeef^{commit}"),
                    shallow_probe,
                ),
            ):
                self.assertIsNone(vc._is_behind("deadbeef"))

    def test_missing_commit_in_complete_history_is_behind(self) -> None:
        from tui.common import version_check as vc

        with patch.object(
            vc,
            "_git",
            side_effect=(
                (128, "fatal: Not a valid object name deadbeef^{commit}"),
                (0, "false\n"),
            ),
        ):
            self.assertIs(vc._is_behind("deadbeef"), True)

    def test_resolve_status_exposes_unknown_comparison_to_consumers(self) -> None:
        from tui.common import version_check as vc

        with (
            patch.object(vc, "_local_version", return_value="1.0.0"),
            patch.object(vc, "_is_git_checkout", return_value=True),
            patch.object(vc, "_cooldown_active", return_value=False),
            patch.object(vc, "_repo_slug", return_value="owner/repo"),
            patch.object(vc, "_latest_release", return_value=("v2.0.0", "release-url")),
            patch.object(vc, "_clear_failure"),
            patch.object(vc, "_release_commit", return_value="deadbeef"),
            patch.object(vc, "_is_behind", return_value=None),
        ):
            status = vc.resolve_status()

        self.assertEqual(status.state, vc.UNKNOWN)
        self.assertIn("could not compare", status.detail)


if __name__ == "__main__":
    unittest.main()
