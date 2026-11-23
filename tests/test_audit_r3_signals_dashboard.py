from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class GpuProbeContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_nvidia_smi_is_not_an_empty_gpu_list(self) -> None:
        from tui.common.docker import get_gpu_info

        with patch("tui.common.docker.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "nvidia-smi"):
                await get_gpu_info()

    async def test_pcie_command_and_malformed_rows_are_errors(self) -> None:
        from tui.common.docker import get_pcie_stats

        with patch(
            "tui.common.docker.shutil.which", return_value="/usr/bin/nvidia-smi"
        ), patch(
            "tui.common.docker.run_command",
            AsyncMock(return_value=(1, "driver communication failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "driver communication failed"):
                await get_pcie_stats()

        with patch(
            "tui.common.docker.shutil.which", return_value="/usr/bin/nvidia-smi"
        ), patch(
            "tui.common.docker.run_command",
            AsyncMock(return_value=(0, "0 malformed\n")),
        ):
            with self.assertRaisesRegex(RuntimeError, "nvidia-smi dmon"):
                await get_pcie_stats()

    async def test_plain_poll_reports_pcie_failure(self) -> None:
        from tui.common import plain_monitor

        with patch.object(
            plain_monitor.common_docker,
            "get_gpu_info",
            AsyncMock(return_value=[]),
        ), patch.object(
            plain_monitor.common_docker,
            "get_pcie_stats",
            AsyncMock(side_effect=RuntimeError("pcie unavailable")),
        ), patch.object(
            plain_monitor,
            "sample_entries",
            AsyncMock(return_value=([], [])),
        ):
            poll = await plain_monitor.poll_monitor(None, {}, [])

        self.assertEqual(poll.pcie, {})
        self.assertTrue(any("PCIe scan failed" in notice for notice in poll.notices))

    async def test_textual_monitor_reports_pcie_failure(self) -> None:
        from tui.screens import monitor

        screen = monitor.MonitorScreen()
        screen._repaint = MagicMock()
        with patch.object(
            monitor.common_docker,
            "get_gpu_info",
            AsyncMock(return_value=[]),
        ), patch.object(
            monitor.common_docker,
            "get_pcie_stats",
            AsyncMock(side_effect=RuntimeError("pcie unavailable")),
        ), patch.object(
            monitor,
            "sample_entries",
            AsyncMock(return_value=([], [])),
        ):
            await screen._poll()

        self.assertTrue(
            any("PCIe scan failed" in notice for notice in screen._last["notices"])
        )


class GpuRendererContractTests(unittest.TestCase):
    def test_partial_gpu_readings_keep_the_monitor_row(self) -> None:
        from rich.console import Console

        from tui.common.docker import GpuInfo
        from tui.common.monitor_render import _gpu_panel

        output = io.StringIO()
        gpu = GpuInfo(
            "0",
            "Partial GPU",
            "[N/A]",
            "24576",
            "[N/A]",
            "NaN",
            "[N/A]",
        )
        Console(file=output, width=100).print(_gpu_panel([gpu], {}))
        rendered = output.getvalue()

        self.assertIn("GPU0", rendered)
        self.assertIn("Partial GPU", rendered)
        self.assertIn("—", rendered)

    def test_partial_gpu_readings_keep_the_dashboard_bar_entry(self) -> None:
        from tui.common.docker import GpuInfo, format_gpu_bar

        rendered = format_gpu_bar(
            [GpuInfo("0", "Partial GPU", "[N/A]", "24576", "[N/A]", "NaN")]
        )

        self.assertIn("GPU0", rendered)
        self.assertIn("—", rendered)


class MemoryFitContractTests(unittest.TestCase):
    def _invoke(self, gpus):
        from typer.testing import CliRunner

        from tui.cli import system

        with patch(
            "tui.common.mem.estimate_model_memory",
            AsyncMock(return_value="estimated ~4.0GB"),
        ), patch(
            "tui.common.docker.get_gpu_info",
            AsyncMock(return_value=gpus),
        ):
            return CliRunner().invoke(
                system.app, ["mem-estimate", "org/model", "--json"]
            )

    def test_no_gpu_is_an_unknown_fit_and_nonzero_exit(self) -> None:
        result = self._invoke([])

        self.assertEqual(result.exit_code, 1)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "error")
        self.assertIsNone(payload["any_over"])

    def test_nonpositive_gpu_memory_is_an_unknown_fit(self) -> None:
        from tui.common.docker import GpuInfo

        result = self._invoke([GpuInfo("0", "GPU", "0", "0", "0", "0")])

        self.assertEqual(result.exit_code, 1)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "error")
        self.assertIsNone(payload["gpus"][0]["ratio"])
        self.assertIsNone(payload["gpus"][0]["over"])


class HfMemContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_multi_gguf_result_requires_explicit_file_selection(self) -> None:
        from tui.common.mem import estimate_model_memory

        hf_mem = SimpleNamespace(
            arun=AsyncMock(
                return_value=SimpleNamespace(
                    memory={"model-q4.gguf": 4 * 1024**3, "model-q8.gguf": 8 * 1024**3},
                    kv_cache=0,
                    total_memory=None,
                )
            )
        )
        with patch.dict(sys.modules, {"hf_mem": hf_mem}):
            result = await estimate_model_memory("org/multi-gguf", hf_token="")

        self.assertIn("multiple GGUF", result)
        self.assertIn("select", result.lower())
        self.assertNotIn("TypeError", result)


class DashboardWorkerContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_llamacpp_benchmark_config_error_is_notified(self) -> None:
        from tui.screens import dashboard

        screen = dashboard.DashboardScreen()
        screen.notify = MagicMock()
        profile = SimpleNamespace(config_name="bad", name="p", port=8000)
        with patch.object(
            dashboard.lbackend,
            "load_config",
            side_effect=ValueError("malformed config yaml"),
        ):
            await dashboard.DashboardScreen._run_llamacpp_bench.__wrapped__(
                screen, profile
            )

        self.assertTrue(screen.notify.called)
        self.assertIn("benchmark failed", screen.notify.call_args.args[0])
        self.assertEqual(screen.notify.call_args.kwargs["severity"], "error")

    async def test_gpu_renderer_error_sets_error_state_and_notifies(self) -> None:
        from tui.common.docker import GpuInfo
        from tui.screens import dashboard

        screen = dashboard.DashboardScreen()
        previous = GpuInfo("0", "old", "1", "2", "3", "4")
        current = GpuInfo("0", "new", "1", "2", "3", "4")
        screen._gpus = [previous]
        screen.notify = MagicMock()
        widget = MagicMock()
        screen.query_one = MagicMock(return_value=widget)
        with patch.object(
            dashboard.common_docker,
            "get_gpu_info",
            AsyncMock(return_value=[current]),
        ), patch.object(
            dashboard.common_docker,
            "format_gpu_bar",
            side_effect=ValueError("render failed"),
        ):
            await dashboard.DashboardScreen._poll_gpu.__wrapped__(screen)

        self.assertIn("render failed", screen._gpu_scan_error)
        self.assertEqual(screen._gpus, [previous])
        self.assertTrue(screen.notify.called)
        self.assertEqual(screen.notify.call_args.kwargs["severity"], "error")
        widget.update.assert_called_once()

    async def test_dashboard_marks_no_gpu_fit_unknown(self) -> None:
        from tui.screens import dashboard

        screen = dashboard.DashboardScreen()
        screen._gpus = []
        result_bar = MagicMock()
        screen.query_one = MagicMock(return_value=result_bar)
        with patch.object(
            dashboard,
            "estimate_model_memory",
            AsyncMock(return_value="estimated ~4.0GB"),
        ):
            await dashboard.DashboardScreen._do_mem_estimate.__wrapped__(
                screen, "org/model"
            )

        rendered = result_bar.update.call_args_list[-1].args[0]
        self.assertIn("fit unknown", rendered.lower())

    async def test_dashboard_marks_cached_gpu_fit_unknown_after_scan_failure(self) -> None:
        from tui.common.docker import GpuInfo
        from tui.screens import dashboard

        screen = dashboard.DashboardScreen()
        screen._gpus = [GpuInfo("0", "GPU", "0", "24576", "0", "0")]
        screen._gpu_scan_error = "nvidia-smi query failed"
        result_bar = MagicMock()
        screen.query_one = MagicMock(return_value=result_bar)
        with patch.object(
            dashboard,
            "estimate_model_memory",
            AsyncMock(return_value="estimated ~4.0GB"),
        ):
            await dashboard.DashboardScreen._do_mem_estimate.__wrapped__(
                screen, "org/model"
            )

        rendered = result_bar.update.call_args_list[-1].args[0]
        self.assertIn("fit unknown", rendered.lower())
        self.assertIn("nvidia-smi query failed", rendered)
        self.assertNotIn("17%", rendered)

    async def test_dashboard_marks_nonpositive_memory_unknown(self) -> None:
        from tui.common.docker import GpuInfo
        from tui.screens import dashboard

        screen = dashboard.DashboardScreen()
        screen._gpus = [GpuInfo("0", "GPU", "0", "0", "0", "0")]
        result_bar = MagicMock()
        screen.query_one = MagicMock(return_value=result_bar)
        with patch.object(
            dashboard,
            "estimate_model_memory",
            AsyncMock(return_value="estimated ~4.0GB"),
        ):
            await dashboard.DashboardScreen._do_mem_estimate.__wrapped__(
                screen, "org/model"
            )

        rendered = result_bar.update.call_args_list[-1].args[0]
        self.assertIn("UNKNOWN", rendered)
        self.assertNotIn("0%", rendered)


class PrometheusGrammarContractTests(unittest.TestCase):
    def test_optional_timestamp_does_not_replace_sample_value(self) -> None:
        from tui.common.metrics import parse_token_counters

        counters = parse_token_counters(
            "vllm:prompt_tokens_total 10 1700000000000\n"
            "vllm:generation_tokens_total 20 1700000000000\n"
        )

        self.assertEqual(counters, (10.0, 20.0))

    def test_labels_and_timestamp_follow_exact_grammar(self) -> None:
        from tui.common.metrics import parse_snapshot

        valid = parse_snapshot(
            'vllm:kv_cache_usage_perc{gpu="0",worker="a\\n\\\"b"} 0.5 7\n'
        )
        malformed_label = parse_snapshot(
            'vllm:kv_cache_usage_perc{gpu="0" 0.5\n'
        )
        malformed_timestamp = parse_snapshot(
            "vllm:kv_cache_usage_perc 0.5 1.5\n"
        )

        self.assertEqual(valid.kv_cache_usage, 0.5)
        self.assertIsNone(malformed_label.kv_cache_usage)
        self.assertIsNone(malformed_timestamp.kv_cache_usage)

    def test_counter_and_histogram_overflow_are_unavailable(self) -> None:
        from tui.common.metrics import parse_snapshot, parse_token_counters

        counters = parse_token_counters(
            "vllm:prompt_tokens_total 1e308\n"
            "vllm:prompt_tokens_total 1e308\n"
            "vllm:generation_tokens_total 1\n"
        )
        snapshot = parse_snapshot(
            "vllm:time_to_first_token_seconds_sum 1e308\n"
            "vllm:time_to_first_token_seconds_sum 1e308\n"
            "vllm:time_to_first_token_seconds_count 2\n"
        )

        self.assertIsNone(counters)
        self.assertIsNone(snapshot.ttft)


class DockerPortGrammarContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_ports_are_distinct_from_malformed_and_duplicate_rows(self) -> None:
        from tui.common.docker import running_container_ports

        with patch(
            "tui.common.docker.run_command", AsyncMock(return_value=(0, ""))
        ):
            self.assertEqual(await running_container_ports(), {})

        for output in (
            "malformed\n",
            "same\t8000/tcp\nsame\t8001/tcp\n",
        ):
            with self.subTest(output=output), patch(
                "tui.common.docker.run_command", AsyncMock(return_value=(0, output))
            ):
                with self.assertRaisesRegex(RuntimeError, "docker ps"):
                    await running_container_ports()

    def test_published_ranges_are_fully_checked(self) -> None:
        from tui.common.adapter import DashboardRow
        from tui.common.conflicts import external_port_conflicts

        target = DashboardRow("vllm", "p", "p", 8001, False, "m", "")
        messages = external_port_conflicts(
            target,
            [target],
            {"external": "0.0.0.0:8000-8002->9000-9002/tcp"},
        )

        self.assertEqual(len(messages), 1)
        self.assertIn("external", messages[0])

    def test_non_tcp_publications_do_not_conflict_with_http_port(self) -> None:
        from tui.common.adapter import DashboardRow
        from tui.common.conflicts import external_port_conflicts

        target = DashboardRow("vllm", "p", "p", 8000, False, "m", "")
        for protocol in ("udp", "sctp"):
            with self.subTest(protocol=protocol):
                messages = external_port_conflicts(
                    target,
                    [target],
                    {"external": f"0.0.0.0:8000->8000/{protocol}"},
                )

                self.assertEqual(messages, [])

    def test_malformed_published_port_syntax_is_an_error(self) -> None:
        from tui.common.adapter import DashboardRow
        from tui.common.conflicts import external_port_conflicts

        target = DashboardRow("vllm", "p", "p", 8001, False, "m", "")
        for value in (
            "0.0.0.0:bad->9000/tcp",
            "0.0.0.0:8000-8002->9000-9001/tcp",
            "0.0.0.0:0->9000/tcp",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "Docker port"):
                    external_port_conflicts(target, [target], {"external": value})

    async def test_backend_http_probe_ignores_udp_and_sctp_publications(self) -> None:
        from tui.backends.llamacpp import backend as llamacpp_backend
        from tui.backends.llamacpp import backend_runtime as llamacpp_runtime
        from tui.backends.vllm import backend_runtime as vllm_runtime
        from tui.backends.vllm.backend_common import Profile as VllmProfile

        cases = (
            (
                vllm_runtime,
                "run_command",
                VllmProfile(name="p", container_name="p", port="8000"),
            ),
            (
                llamacpp_runtime,
                "_docker_run",
                llamacpp_backend.Profile(name="p", container_name="p", port=8000),
            ),
        )
        for runtime, run_name, profile in cases:
            for protocol in ("udp", "sctp"):
                with self.subTest(runtime=runtime.__name__, protocol=protocol):
                    sock = MagicMock()
                    with patch.object(
                        runtime,
                        run_name,
                        AsyncMock(
                            return_value=(
                                0,
                                f"external\t0.0.0.0:8000->8000/{protocol}\n",
                            )
                        ),
                    ), patch("socket.socket", return_value=sock):
                        conflict = await runtime.check_port_conflict(profile)

                    self.assertIsNone(conflict)
                    sock.bind.assert_called_once_with(("127.0.0.1", 8000))

    async def test_backend_http_probe_rejects_malformed_docker_rows(self) -> None:
        from tui.backends.llamacpp import backend as llamacpp_backend
        from tui.backends.llamacpp import backend_runtime as llamacpp_runtime
        from tui.backends.vllm import backend_runtime as vllm_runtime
        from tui.backends.vllm.backend_common import Profile as VllmProfile

        cases = (
            (
                vllm_runtime,
                "run_command",
                VllmProfile(name="p", container_name="p", port="8000"),
            ),
            (
                llamacpp_runtime,
                "_docker_run",
                llamacpp_backend.Profile(name="p", container_name="p", port=8000),
            ),
        )
        for runtime, run_name, profile in cases:
            for output in ("malformed\n", "external\t0.0.0.0:bad->8000/tcp\n"):
                with self.subTest(runtime=runtime.__name__, output=output), patch.object(
                    runtime, run_name, AsyncMock(return_value=(0, output))
                ), patch("socket.socket") as socket_factory:
                    with self.assertRaisesRegex(RuntimeError, "docker ps|Docker port"):
                        await runtime.check_port_conflict(profile)

                socket_factory.assert_not_called()


class SystemInventoryWorkerContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_inventory_errors_are_rendered_inside_worker(self) -> None:
        from tui.backends.llamacpp.screens import system as llamacpp_system
        from tui.backends.vllm.screens import system as vllm_system

        for module in (vllm_system, llamacpp_system):
            for failure in (ValueError("bad profile yaml"), OSError("profile unreadable")):
                with self.subTest(module=module.__name__, failure=type(failure).__name__):
                    screen = module.SystemScreen()
                    log = MagicMock()
                    screen.query_one = MagicMock(return_value=log)
                    screen.notify = MagicMock()
                    with patch.object(
                        module, "list_profile_names", side_effect=failure
                    ), patch.object(
                        module, "container_snapshots", AsyncMock()
                    ) as snapshots:
                        await module.SystemScreen._refresh_containers.__wrapped__(screen)

                    snapshots.assert_not_awaited()
                    self.assertTrue(log.write.called)
                    self.assertIn(
                        str(failure),
                        " ".join(str(call.args[0]) for call in log.write.call_args_list),
                    )
                    self.assertEqual(
                        screen.notify.call_args.kwargs["severity"], "error"
                    )

    async def test_profile_load_errors_are_rendered_inside_worker(self) -> None:
        from tui.backends.llamacpp.screens import system as llamacpp_system
        from tui.backends.vllm.screens import system as vllm_system

        for module in (vllm_system, llamacpp_system):
            screen = module.SystemScreen()
            log = MagicMock()
            screen.query_one = MagicMock(return_value=log)
            screen.notify = MagicMock()
            with patch.object(
                module, "list_profile_names", return_value=["broken"]
            ), patch.object(
                module, "load_profile", side_effect=ValueError("invalid profile")
            ), patch.object(
                module, "container_snapshots", AsyncMock()
            ) as snapshots:
                await module.SystemScreen._refresh_containers.__wrapped__(screen)

            snapshots.assert_not_awaited()
            self.assertIn(
                "invalid profile",
                " ".join(str(call.args[0]) for call in log.write.call_args_list),
            )
            self.assertEqual(screen.notify.call_args.kwargs["severity"], "error")


class DockerImageGrammarContractTests(unittest.IsolatedAsyncioTestCase):
    def test_image_rows_distinguish_empty_malformed_and_duplicate(self) -> None:
        from tui.common.docker import parse_docker_image_rows

        self.assertEqual(parse_docker_image_rows(""), [])
        self.assertEqual(
            parse_docker_image_rows("repo\tv1\t1GB\t2 days ago\n"),
            [("repo", "v1", "1GB", "2 days ago")],
        )
        with self.assertRaisesRegex(RuntimeError, "docker image"):
            parse_docker_image_rows("malformed\n")
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            parse_docker_image_rows(
                "repo\tv1\t1GB\t2 days ago\nrepo\tv1\t1GB\t2 days ago\n"
            )

    async def test_image_identity_rejects_credentials_before_subprocess(self) -> None:
        from tui.common.docker import image_identity

        with patch(
            "tui.common.dev_build.image_reference_credential_error",
            return_value="image reference must not contain credentials",
        ), patch(
            "tui.common.docker.run_command", AsyncMock()
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "must not contain credentials"):
                await image_identity("registry.example/user:secret/image:v1")

        run.assert_not_awaited()
