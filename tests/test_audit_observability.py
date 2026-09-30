from __future__ import annotations

import contextlib
import inspect
import io
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class DockerGpuProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_nvidia_smi_is_an_explicit_probe_failure(self) -> None:
        from tui.common import docker

        with patch("shutil.which", return_value=None), patch.object(
            docker, "run_command", AsyncMock()
        ) as command:
            with self.assertRaisesRegex(RuntimeError, "nvidia-smi"):
                await docker.get_gpu_info()

        command.assert_not_awaited()

    async def test_installed_nvidia_smi_query_failure_raises(self) -> None:
        from tui.common import docker

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch.object(
            docker,
            "run_command",
            AsyncMock(return_value=(1, "unsupported query field")),
        ):
            with self.assertRaisesRegex(RuntimeError, "unsupported query field"):
                await docker.get_gpu_info()

    async def test_successful_empty_gpu_query_stays_empty(self) -> None:
        from tui.common import docker

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch.object(
            docker, "run_command", AsyncMock(return_value=(0, ""))
        ):
            self.assertEqual(await docker.get_gpu_info(), [])

    async def test_malformed_gpu_query_is_not_reported_as_no_gpus(self) -> None:
        from tui.common import docker

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch.object(
            docker, "run_command", AsyncMock(return_value=(0, "not,csv"))
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected nvidia-smi output"):
                await docker.get_gpu_info()

    def test_gpu_cli_failure_has_json_error_and_nonzero_status(self) -> None:
        import typer

        from tui.cli import system
        from tui.common import docker

        stdout = io.StringIO()
        with patch.object(
            docker, "get_gpu_info", AsyncMock(side_effect=RuntimeError("query failed"))
        ), contextlib.redirect_stdout(stdout):
            with self.assertRaises(typer.Exit) as caught:
                system.gpu(json_out=True)

        self.assertEqual(caught.exception.exit_code, 1)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {"status": "error", "error": "query failed"},
        )

    def test_mem_estimate_gpu_failure_has_json_error_and_nonzero_status(self) -> None:
        import typer

        from tui.cli import system
        from tui.common import docker, mem

        stdout = io.StringIO()
        with patch.object(
            mem, "estimate_model_memory", AsyncMock(return_value="~1.0GB")
        ), patch.object(
            docker, "get_gpu_info", AsyncMock(side_effect=RuntimeError("query failed"))
        ), contextlib.redirect_stdout(stdout):
            with self.assertRaises(typer.Exit) as caught:
                system.mem_estimate("org/model", json_out=True)

        self.assertEqual(caught.exception.exit_code, 1)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "status": "error",
                "model_id": "org/model",
                "error": "query failed",
            },
        )


class HttpProbeTests(unittest.IsolatedAsyncioTestCase):
    class _Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self) -> bytes:
            return self.payload

    async def test_model_discovery_preserves_successful_empty_response(self) -> None:
        from tui.common import http

        response = self._Response(b'{"data": []}')
        with patch.object(http, "open_url", return_value=response):
            self.assertEqual(await http.list_served_models(8000), [])

    async def test_model_discovery_failure_raises(self) -> None:
        from tui.common import http

        with patch.object(
            http, "open_url", side_effect=OSError("connection refused")
        ):
            with self.assertRaisesRegex(RuntimeError, "model discovery failed"):
                await http.list_served_models(8000)

    async def test_completion_tokens_must_be_a_bounded_integer(self) -> None:
        from tui.common import http

        for value in (True, 1.5, "2", -1, 201):
            with self.subTest(value=value), patch.object(
                http,
                "chat_completion_bench",
                AsyncMock(return_value={"elapsed": 1.0, "usage": {"completion_tokens": value}}),
            ):
                with self.assertRaisesRegex(RuntimeError, "completion_tokens"):
                    await http.run_bench(
                        8000, "model", max_tokens=200, runs=1, warmup=0
                    )

    async def test_completion_tokens_accepts_zero_and_maximum(self) -> None:
        from tui.common import http

        for value in (0, 200):
            with self.subTest(value=value), patch.object(
                http,
                "chat_completion_bench",
                AsyncMock(return_value={"elapsed": 1.0, "usage": {"completion_tokens": value}}),
            ):
                result = await http.run_bench(
                    8000, "model", max_tokens=200, runs=1, warmup=0
                )
                self.assertEqual(result["runs"][0]["tokens"], value)


class MetricsMissingValueTests(unittest.TestCase):
    def test_partial_token_counters_are_unavailable(self) -> None:
        from tui.common.metrics import parse_snapshot, parse_token_counters

        prompt_only = "vllm:prompt_tokens_total 10\n"
        generation_only = "vllm:generation_tokens_total 20\n"

        self.assertIsNone(parse_token_counters(prompt_only))
        self.assertIsNone(parse_token_counters(generation_only))
        self.assertIsNone(parse_snapshot(prompt_only).token_counters())
        self.assertIsNone(parse_snapshot(generation_only).token_counters())

    def test_partial_sample_breaks_the_rate_baseline(self) -> None:
        from tui.common.metrics import MetricsSnapshot
        from tui.common.monitor_render import MonitorState

        state = MonitorState()
        state.update(MetricsSnapshot(prompt_tokens=10, generation_tokens=20), 1.0, 0.0)
        partial = state.update(MetricsSnapshot(prompt_tokens=11), 2.0, 0.0)
        resumed = state.update(
            MetricsSnapshot(prompt_tokens=12, generation_tokens=24), 3.0, 0.0
        )
        measured = state.update(
            MetricsSnapshot(prompt_tokens=14, generation_tokens=30), 4.0, 0.0
        )

        self.assertIsNone(partial.gen_tps)
        self.assertIsNone(resumed.gen_tps)
        self.assertEqual(measured.prompt_tps, 2.0)
        self.assertEqual(measured.gen_tps, 6.0)

    def test_missing_monitor_values_stay_missing_in_history_and_render(self) -> None:
        from rich.console import Console

        from tui.common.adapter import DashboardRow
        from tui.common.monitor_render import ModelEntry, MonitorState, render_dashboard

        state = MonitorState()
        derived = state.update(None, 1.0, 0.0)
        entry = ModelEntry(
            DashboardRow("vllm", "p", "p", 8000, True, "m", ""),
            None,
            state,
            derived,
        )
        output = io.StringIO()
        Console(width=110, file=output).print(
            render_dashboard([entry], [], {}, 110)
        )
        rendered = output.getvalue()

        self.assertEqual(list(state.gen), [None])
        self.assertEqual(list(state.prompt), [None])
        self.assertEqual(list(state.kv), [None])
        self.assertIsNone(state.peak_gen)
        self.assertIsNone(state.peak_prompt)
        self.assertIsNone(state.peak_kv)
        self.assertIn("▲—", rendered)
        self.assertIn("scale — tok/s", rendered)
        self.assertIn("▲peak —", rendered)

    def test_measured_zero_is_distinct_from_missing(self) -> None:
        from tui.common.metrics import MetricsSnapshot
        from tui.common.monitor_render import MonitorState

        state = MonitorState()
        snapshot = MetricsSnapshot(
            prompt_tokens=0.0, generation_tokens=0.0, kv_cache_usage=0.0
        )
        state.update(snapshot, 1.0, 0.0)
        measured = state.update(snapshot, 2.0, 0.0)

        self.assertEqual(measured.prompt_tps, 0.0)
        self.assertEqual(measured.gen_tps, 0.0)
        self.assertEqual(state.gen[-1], 0.0)
        self.assertEqual(state.kv[-1], 0.0)
        self.assertEqual(state.peak_gen, 0.0)
        self.assertEqual(state.peak_kv, 0.0)


class TerminalRestoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_restore_failure_raises_when_monitor_would_exit_cleanly(self) -> None:
        from tui.common import plain_monitor

        class _Live:
            def __init__(self, **_kwargs) -> None:
                pass

            def __enter__(self):
                raise KeyboardInterrupt

            def __exit__(self, *_args) -> None:
                return None

        def fail_restore(*_args) -> None:
            raise OSError("restore denied")

        termios = SimpleNamespace(
            TCSADRAIN=1,
            tcgetattr=lambda _fd: ["saved"],
            tcsetattr=fail_restore,
        )
        tty = SimpleNamespace(setcbreak=lambda _fd: None)
        with patch.dict(
            sys.modules,
            {
                "termios": termios,
                "tty": tty,
                "rich.live": SimpleNamespace(Live=_Live),
            },
        ), patch.object(sys.stdin, "fileno", return_value=7):
            with self.assertRaisesRegex(
                plain_monitor.TerminalRestoreError, "restore denied"
            ):
                await plain_monitor.run_plain_monitor()

    async def test_body_error_is_preserved_when_restore_also_fails(self) -> None:
        from tui.common import plain_monitor

        class _Live:
            def __init__(self, **_kwargs) -> None:
                pass

            def __enter__(self):
                raise ValueError("body failed")

            def __exit__(self, *_args) -> None:
                return None

        def fail_restore(*_args) -> None:
            raise OSError("restore denied")

        termios = SimpleNamespace(
            TCSADRAIN=1,
            tcgetattr=lambda _fd: ["saved"],
            tcsetattr=fail_restore,
        )
        tty = SimpleNamespace(setcbreak=lambda _fd: None)
        stderr = io.StringIO()
        with patch.dict(
            sys.modules,
            {
                "termios": termios,
                "tty": tty,
                "rich.live": SimpleNamespace(Live=_Live),
            },
        ), patch.object(sys.stdin, "fileno", return_value=7), contextlib.redirect_stderr(
            stderr
        ):
            with self.assertRaisesRegex(ValueError, "body failed"):
                await plain_monitor.run_plain_monitor()

        self.assertIn("terminal restore failed", stderr.getvalue())
        self.assertIn("restore denied", stderr.getvalue())

    def test_plain_monitor_cli_reports_restore_failure_as_nonzero(self) -> None:
        from tui.common import plain_monitor

        def fail(coro):
            coro.close()
            raise plain_monitor.TerminalRestoreError("restore denied")

        stderr = io.StringIO()
        with patch.object(
            plain_monitor.asyncio,
            "run",
            side_effect=fail,
        ), contextlib.redirect_stderr(stderr):
            rc = plain_monitor.run_cli()

        self.assertEqual(rc, 1)
        self.assertIn("restore denied", stderr.getvalue())


class DiskErrorBoundaryTests(unittest.TestCase):
    def test_vllm_env_inventory_failure_has_json_error_schema(self) -> None:
        import typer

        from tui.cli import system
        from tui.common import env, profile_store

        common = MagicMock()
        common.exists.return_value = True
        root = MagicMock()
        root.__truediv__.return_value = common
        stdout = io.StringIO()
        with patch.object(profile_store, "PROJECT_ROOT", root), patch.object(
            env, "parse_env_file", side_effect=OSError("env inventory failed")
        ), contextlib.redirect_stdout(stdout):
            with self.assertRaises(typer.Exit) as caught:
                system.disk(backend="vllm", json_out=True)

        self.assertEqual(caught.exception.exit_code, 1)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "status": "error",
                "backend": "vllm",
                "error": "env inventory failed",
            },
        )

    def test_llamacpp_inventory_failure_has_json_error_schema(self) -> None:
        import typer

        from tui.backends.llamacpp import backend
        from tui.cli import system

        stdout = io.StringIO()
        with patch.object(
            backend, "_get_model_dir", side_effect=OSError("inventory failed")
        ), contextlib.redirect_stdout(stdout):
            with self.assertRaises(typer.Exit) as caught:
                system.disk(backend="llamacpp", json_out=True)

        self.assertEqual(caught.exception.exit_code, 1)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "status": "error",
                "backend": "llamacpp",
                "error": "inventory failed",
            },
        )

    def test_llamacpp_missing_cache_config_is_a_human_error(self) -> None:
        import typer

        from tui.backends.llamacpp import backend
        from tui.cli import system

        stderr = io.StringIO()
        with patch.object(backend, "_get_model_dir", return_value=SimpleNamespace()), patch.object(
            backend, "_get_hf_cache_dir", side_effect=ValueError("HF_CACHE_PATH is required")
        ), contextlib.redirect_stderr(stderr):
            with self.assertRaises(typer.Exit) as caught:
                system.disk(backend="llamacpp", json_out=False)

        self.assertEqual(caught.exception.exit_code, 1)
        self.assertIn("HF_CACHE_PATH is required", stderr.getvalue())


class ConflictProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_docker_probe_failure_aborts_conflict_scan(self) -> None:
        from tui.common import conflicts

        with patch.object(
            conflicts, "run_command", AsyncMock(return_value=(1, "daemon unavailable"))
        ):
            with self.assertRaisesRegex(RuntimeError, "daemon unavailable"):
                await conflicts.gpu_conflict_messages(
                    profile_name="p",
                    container_name="container-p",
                    profile_gpu_id="0",
                    backend="vllm",
                )

    async def test_profile_scan_failure_propagates(self) -> None:
        from tui.common import conflicts

        with patch.object(
            conflicts, "run_command", AsyncMock(return_value=(0, "other\n"))
        ), patch(
            "tui.common.profile_store.list_profiles",
            side_effect=ValueError("profiles invalid"),
        ):
            with self.assertRaisesRegex(ValueError, "profiles invalid"):
                await conflicts.gpu_conflict_messages(
                    profile_name="p",
                    container_name="container-p",
                    profile_gpu_id="0",
                    backend="vllm",
                )

    async def test_target_container_is_not_compared_with_itself(self) -> None:
        from tui.common import conflicts

        other = SimpleNamespace(
            name="legacy-alias", container_name="container-p", gpu_id="0"
        )
        with patch.object(
            conflicts,
            "run_command",
            AsyncMock(return_value=(0, "container-p\n")),
        ), patch(
            "tui.common.profile_store.list_profiles",
            side_effect=lambda name: [other] if name == "llamacpp" else [],
        ):
            messages = await conflicts.gpu_conflict_messages(
                profile_name="p",
                container_name="container-p",
                profile_gpu_id="0",
                backend="vllm",
            )

        self.assertEqual(messages, [])

    async def test_cli_conflict_preflight_propagates_docker_failure(self) -> None:
        from tui.cli import _runtime
        from tui.common import docker

        with patch.object(
            docker,
            "running_container_names",
            AsyncMock(side_effect=RuntimeError("daemon unavailable")),
        ):
            with self.assertRaisesRegex(RuntimeError, "daemon unavailable"):
                await _runtime.gather_conflict_warnings("p", "vllm")


class MemoryFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_hf_mem_import_failure_is_identified(self) -> None:
        from tui.common.mem import estimate_model_memory

        with patch.dict(sys.modules, {"hf_mem": None}):
            result = await estimate_model_memory("org/model", hf_token="")

        self.assertIn("hf-mem import failed", result)

    async def test_hf_mem_network_failure_is_identified(self) -> None:
        from tui.common.mem import estimate_model_memory

        async def fail(**_kwargs):
            raise ConnectionError("offline")

        with patch.dict(sys.modules, {"hf_mem": SimpleNamespace(arun=fail)}):
            result = await estimate_model_memory("org/model", hf_token="")

        self.assertIn("hf-mem network failed", result)
        self.assertIn("offline", result)

    async def test_hf_mem_runtime_failure_is_identified(self) -> None:
        from tui.common.mem import estimate_model_memory

        async def fail(**_kwargs):
            raise ValueError("schema changed")

        with patch.dict(sys.modules, {"hf_mem": SimpleNamespace(arun=fail)}):
            result = await estimate_model_memory("org/model", hf_token="")

        self.assertIn("hf-mem runtime failed", result)
        self.assertIn("schema changed", result)


class BenchmarkDefaultTests(unittest.TestCase):
    def test_cli_uses_common_benchmark_defaults(self) -> None:
        from tui.cli.container import benchmark
        from tui.common.http import (
            BENCH_MAX_TOKENS,
            BENCH_PROMPT,
            BENCH_RUNS,
            BENCH_WARMUP,
        )

        parameters = inspect.signature(benchmark).parameters
        self.assertEqual(parameters["prompt"].default.default, BENCH_PROMPT)
        self.assertEqual(parameters["max_tokens"].default.default, BENCH_MAX_TOKENS)
        self.assertEqual(parameters["runs"].default.default, BENCH_RUNS)
        self.assertEqual(parameters["warmup"].default.default, BENCH_WARMUP)


class SystemScreenErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_monitor_preserves_gpu_state_and_adds_error_notice(self) -> None:
        from tui.screens import monitor

        screen = SimpleNamespace(
            _paused=False,
            _polling=False,
            _focus=None,
            _states={},
            _last={
                "entries": [],
                "gpus": ["last-known"],
                "pcie": {},
                "lag": 0.0,
                "ready": True,
                "notices": [],
            },
            _repaint=MagicMock(),
        )
        with patch.object(
            monitor.common_docker,
            "get_gpu_info",
            AsyncMock(side_effect=RuntimeError("query failed")),
        ), patch.object(
            monitor.common_docker, "get_pcie_stats", AsyncMock(return_value={})
        ), patch.object(
            monitor, "sample_entries", AsyncMock(return_value=([], []))
        ):
            await monitor.MonitorScreen._poll(screen)

        self.assertEqual(screen._last["gpus"], ["last-known"])
        self.assertTrue(
            any("GPU scan failed" in notice for notice in screen._last["notices"])
        )
        screen._repaint.assert_called_once()

    async def test_backend_system_screens_render_gpu_query_failure(self) -> None:
        from tui.backends.llamacpp.screens import system as llama_system
        from tui.backends.vllm.screens import system as vllm_system

        for module in (vllm_system, llama_system):
            with self.subTest(module=module.__name__):
                table = MagicMock()
                screen = SimpleNamespace(
                    query_one=MagicMock(return_value=table),
                    notify=MagicMock(),
                )
                screen._update_gpu_table = lambda gpus: module.SystemScreen._update_gpu_table(
                    screen, gpus
                )
                screen._update_gpu_error = lambda exc: module.SystemScreen._update_gpu_error(
                    screen, exc
                )
                with patch.object(
                    module,
                    "get_gpu_info",
                    AsyncMock(side_effect=RuntimeError("query failed")),
                ):
                    await module.SystemScreen._refresh_gpu.__wrapped__(screen)

                table.clear.assert_called_once()
                table.add_row.assert_called_once()
                screen.notify.assert_called_once()
                self.assertIn("query failed", screen.notify.call_args.args[0])

    async def test_llamacpp_system_disk_inventory_failure_is_rendered(self) -> None:
        from tui.backends.llamacpp.screens import system as llama_system

        log = MagicMock()
        screen = SimpleNamespace(
            query_one=MagicMock(return_value=log),
            notify=MagicMock(),
        )
        with patch.object(
            llama_system,
            "_get_model_dir",
            side_effect=OSError("inventory failed"),
        ):
            await llama_system.SystemScreen._refresh_disk.__wrapped__(screen)

        log.clear.assert_called_once()
        self.assertTrue(
            any("inventory failed" in str(call.args[0]) for call in log.write.call_args_list)
        )
        screen.notify.assert_called_once()

    async def test_vllm_system_disk_inventory_failure_is_rendered(self) -> None:
        from tui.backends.vllm.screens import system as vllm_system

        log = MagicMock()
        screen = SimpleNamespace(
            query_one=MagicMock(return_value=log),
            notify=MagicMock(),
        )
        with patch.object(
            vllm_system,
            "parse_env_file",
            side_effect=OSError("inventory failed"),
        ):
            await vllm_system.SystemScreen._refresh_disk.__wrapped__(screen)

        log.clear.assert_called_once()
        self.assertTrue(
            any("inventory failed" in str(call.args[0]) for call in log.write.call_args_list)
        )
        screen.notify.assert_called_once()
