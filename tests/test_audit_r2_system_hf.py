from __future__ import annotations

import contextlib
import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch


class MemoryEstimateAuditTests(unittest.TestCase):
    def test_malformed_gpu_memory_is_unknown_and_fails_json_result(self) -> None:
        import typer

        from tui.cli import system
        from tui.common import docker, mem

        stdout = io.StringIO()
        gpu = docker.GpuInfo("0", "Synthetic", "1", "not-a-number", "10", "20")
        with (
            patch.object(
                mem,
                "estimate_model_memory",
                AsyncMock(return_value="~4.0GB"),
            ),
            patch.object(docker, "get_gpu_info", AsyncMock(return_value=[gpu])),
            contextlib.redirect_stdout(stdout),
        ):
            with self.assertRaises(typer.Exit) as caught:
                system.mem_estimate("org/model", profile=None, backend=None, json_out=True)

        self.assertEqual(caught.exception.exit_code, 1)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["any_over"])
        self.assertIsNone(result["gpus"][0]["total_gb"])
        self.assertIsNone(result["gpus"][0]["ratio"])
        self.assertIsNone(result["gpus"][0]["over"])
        self.assertIn("memory", result["gpus"][0]["error"])


class MemoryRedactionAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_hf_mem_exception_does_not_repeat_token(self) -> None:
        from tui.common.mem import estimate_model_memory

        secret = "hf_synthetic_secret_123"

        async def fail(**_kwargs):
            raise ValueError(
                f"request failed: Authorization: Bearer {secret}; token={secret}"
            )

        with patch.dict(sys.modules, {"hf_mem": SimpleNamespace(arun=fail)}):
            result = await estimate_model_memory("org/model", hf_token=secret)

        self.assertNotIn(secret, result)
        self.assertIn("<redacted>", result)
        self.assertIn("hf-mem runtime failed", result)


class VllmDiskAuditTests(unittest.TestCase):
    def test_nonexistent_cache_uses_nearest_existing_parent_for_df(self) -> None:
        from tui.cli import system
        from tui.common import docker, env, profile_store

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache = root / "future" / "hf-cache"
            (root / ".env.common").write_text("HF_CACHE_PATH=synthetic\n")
            stdout = io.StringIO()
            usage = AsyncMock(return_value=("1G", "9G", "10%"))
            with (
                patch.object(profile_store, "PROJECT_ROOT", root),
                patch.object(env, "parse_env_file", return_value={"HF_CACHE_PATH": str(cache)}),
                patch.object(docker, "get_disk_usage", usage),
                contextlib.redirect_stdout(stdout),
            ):
                system.disk(backend="vllm", json_out=True)

        usage.assert_awaited_once_with(str(root))
        result = json.loads(stdout.getvalue())
        self.assertFalse(result["hf_cache_exists"])
        self.assertEqual(result["df_target"], str(root))

    def test_vllm_screen_uses_nearest_existing_cache_parent_for_df(self) -> None:
        from tui.backends.vllm.screens import system

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache = root / "future" / "hf-cache"
            log = MagicMock()
            screen = SimpleNamespace(
                query_one=MagicMock(return_value=log),
                notify=MagicMock(),
            )
            usage = AsyncMock(return_value=("1G", "9G", "10%"))
            with (
                patch.object(system, "parse_env_file", return_value={"HF_CACHE_PATH": str(cache)}),
                patch.object(system, "get_disk_usage", usage),
                patch.object(system, "host_expand", side_effect=lambda value: value),
                patch.object(system, "COMMON_ENV", root / ".env.common"),
            ):
                awaitable = system.SystemScreen._refresh_disk.__wrapped__(screen)
                import asyncio

                asyncio.run(awaitable)

        usage.assert_awaited_once_with(str(root))
        screen.notify.assert_not_called()


class EnvCheckWarningAuditTests(unittest.TestCase):
    def test_ok_warning_is_preserved_in_json_with_zero_exit(self) -> None:
        import typer

        from tui.cli import system
        from tui.common import env, profile_store

        warning = "Warning: synthetic default is active"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env.common").write_text("HF_CACHE_PATH=/tmp/cache\n")
            stdout = io.StringIO()
            with (
                patch.object(profile_store, "PROJECT_ROOT", root),
                patch.object(env, "validate_common_env", return_value=(True, [warning])),
                patch.object(env, "parse_env_file", return_value={"HF_CACHE_PATH": "/tmp/cache"}),
                contextlib.redirect_stdout(stdout),
            ):
                with self.assertRaises(typer.Exit) as caught:
                    system.env_check(json_out=True)

        self.assertEqual(caught.exception.exit_code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["warnings"], [warning])
        self.assertEqual(result["issues"], [])


class SystemRenderTransactionAuditTests(unittest.TestCase):
    def test_backend_render_reloads_each_profile_inside_storage_transaction(self) -> None:
        from tui.common import profile_store, system_operations

        stale = profile_store.StoredProfile(
            name="p",
            backend="vllm",
            model_id="org/stale",
        )
        invalid = profile_store.StoredProfile(
            name="bad",
            backend="vllm",
            model_id="org/stale",
        )
        rendered_path = Path("/synthetic/runtime/p.env")

        def render_latest(name: str, backend: str) -> Path:
            if name == "bad":
                raise ValueError("invalid config")
            self.assertEqual(backend, "vllm")
            return rendered_path

        render = MagicMock(side_effect=render_latest)
        with (
            patch.object(profile_store, "list_profiles", return_value=[stale, invalid]),
            patch.object(
                profile_store,
                "render_env",
                side_effect=AssertionError("stale snapshot must not be rendered"),
            ),
            patch.object(
                profile_store,
                "render_env_for_profile",
                render,
            ),
        ):
            rendered, failures = system_operations.render_backend_envs("vllm")

        self.assertEqual(rendered, [rendered_path])
        self.assertEqual(failures, ["bad: invalid config"])
        self.assertEqual(
            render.call_args_list,
            [
                call("p", "vllm"),
                call("bad", "vllm"),
            ],
        )


class GpuSystemScreenAuditTests(unittest.TestCase):
    def test_malformed_gpu_metrics_render_unknown_in_both_screens(self) -> None:
        from tui.backends.llamacpp.screens import system as llama_system
        from tui.backends.vllm.screens import system as vllm_system
        from tui.common.docker import GpuInfo

        gpu = GpuInfo("0", "Synthetic", "1", "2", "N/A", "broken")
        for module in (vllm_system, llama_system):
            with self.subTest(module=module.__name__):
                table = MagicMock()
                screen = SimpleNamespace(query_one=MagicMock(return_value=table))
                module.SystemScreen._update_gpu_table(screen, [gpu])
                row = table.add_row.call_args.args
                self.assertEqual(row[4], "--")
                self.assertEqual(row[5], "--")


class ContainerSystemScreenAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_container_state_is_preserved_in_both_system_screens(self) -> None:
        from tui.backends.llamacpp.screens import system as llama_system
        from tui.backends.vllm.screens import system as vllm_system
        from tui.common.docker import (
            ContainerHealth,
            ContainerLifecycle,
            ContainerSnapshot,
        )

        snapshot = ContainerSnapshot(
            name="profile-container",
            lifecycle=ContainerLifecycle.UNKNOWN,
            health=ContainerHealth.UNKNOWN,
            raw_state="future-state",
            raw_status="future status payload",
        )
        for module in (vllm_system, llama_system):
            with self.subTest(module=module.__name__):
                log = MagicMock()
                screen = SimpleNamespace(
                    query_one=MagicMock(return_value=log),
                    notify=MagicMock(),
                )
                with (
                    patch.object(module, "list_profile_names", return_value=["p"]),
                    patch.object(
                        module,
                        "load_profile",
                        return_value=SimpleNamespace(container_name="profile-container"),
                    ),
                    patch.object(module.profile_store, "list_profiles", return_value=[]),
                    patch.object(
                        module,
                        "container_snapshots",
                        AsyncMock(return_value={"profile-container": snapshot}),
                    ),
                ):
                    await module.SystemScreen._refresh_containers.__wrapped__(screen)

                rendered = " ".join(
                    str(call.args[0]) for call in log.write.call_args_list
                )
                self.assertIn("future-state", rendered)
                self.assertIn("unknown", rendered)


class LlamacppCacheDedupeAuditTests(unittest.TestCase):
    def test_same_blob_in_multiple_revisions_is_counted_once(self) -> None:
        from tui.backends.llamacpp import backend

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            repo = cache / "hub" / "models--org--repo"
            blob = repo / "blobs" / "blob-id"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"synthetic")
            for revision in ("rev-a", "rev-b"):
                snapshot = repo / "snapshots" / revision
                snapshot.mkdir(parents=True)
                (snapshot / "model.gguf").symlink_to("../../blobs/blob-id")

            with patch.object(backend, "_get_hf_cache_dir", return_value=cache):
                inventory = backend.list_cached_gguf()

        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0]["size_bytes"], len(b"synthetic"))


class _Response:
    def __init__(self, payload, link: str = "") -> None:
        self.payload = payload
        self.headers = {"Link": link} if link else {}

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


async def _list_hf_inline(backend, repo: str):
    loop = asyncio.get_running_loop()

    def run_inline(_executor, function):
        future = loop.create_future()
        try:
            future.set_result(function())
        except BaseException as exc:
            future.set_exception(exc)
        return future

    with patch.object(loop, "run_in_executor", side_effect=run_inline):
        return await backend.list_hf_repo_files(repo)


class HfListingBoundaryAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_cross_origin_next_page_is_rejected(self) -> None:
        from tui.backends.llamacpp import backend

        base = "https://hf.example/api/models/org/repo/tree/main?recursive=true"
        page2 = "https://cdn.example/tree/page2"
        pages = {
            base: _Response(
                [{"type": "file", "path": "a.gguf"}],
                f'<{page2}>; rel="next"',
            ),
            page2: _Response([{"type": "file", "path": "b.gguf"}]),
        }
        requested: list[tuple[str, str | None]] = []

        def open_url(req, *, timeout):
            requested.append((req.full_url, req.get_header("Authorization")))
            return pages[req.full_url]

        with (
            patch.dict(os.environ, {"HF_ENDPOINT": "https://hf.example"}),
            patch.object(backend, "_parse_env_file", return_value={"HF_TOKEN": "secret"}),
            patch.object(backend, "open_url", open_url),
        ):
            with self.assertRaisesRegex(
                backend.HfListingUnavailable, "off-origin"
            ):
                await _list_hf_inline(backend, "org/repo")

        self.assertEqual(requested, [(base, "Bearer secret")])

    async def test_non_list_intermediate_page_fails_instead_of_returning_partial(self) -> None:
        from tui.backends.llamacpp import backend

        base = "https://huggingface.co/api/models/org/repo/tree/main?recursive=true"
        page2 = f"{base}&cursor=next"
        pages = {
            base: _Response(
                [{"type": "file", "path": "a.gguf"}],
                f'<{page2}>; rel="next"',
            ),
            page2: _Response({"error": "schema changed"}),
        }

        with patch.object(backend, "open_url", lambda req, *, timeout: pages[req.full_url]):
            with self.assertRaisesRegex(backend.HfListingUnavailable, "JSON list"):
                await _list_hf_inline(backend, "org/repo")

    async def test_page_cap_fails_instead_of_returning_truncated_entries(self) -> None:
        from tui.backends.llamacpp import backend

        base = "https://huggingface.co/api/models/org/repo/tree/main?recursive=true"
        page2 = f"{base}&cursor=2"
        page3 = f"{base}&cursor=3"
        pages = {
            base: _Response(
                [{"type": "file", "path": "a.gguf"}],
                f'<{page2}>; rel="next"',
            ),
            page2: _Response(
                [{"type": "file", "path": "b.gguf"}],
                f'<{page3}>; rel="next"',
            ),
        }

        with (
            patch.object(backend, "_HF_TREE_PAGE_CAP", 2),
            patch.object(backend, "open_url", lambda req, *, timeout: pages[req.full_url]),
        ):
            with self.assertRaisesRegex(backend.HfListingUnavailable, "page limit"):
                await _list_hf_inline(backend, "org/repo")


class RedirectBoundaryAuditTests(unittest.TestCase):
    def test_sensitive_headers_are_removed_only_for_cross_origin_redirects(self) -> None:
        from tui.common.ssl_ctx import _SameOriginRedirectHandler

        handler = _SameOriginRedirectHandler()
        request = urllib.request.Request(
            "https://hf.example/private",
            headers={"Authorization": "Bearer secret"},
        )

        same = handler.redirect_request(
            request, None, 302, "Found", {}, "https://hf.example/next"
        )
        cross = handler.redirect_request(
            request, None, 302, "Found", {}, "https://other.example/next"
        )

        self.assertEqual(same.get_header("Authorization"), "Bearer secret")
        self.assertIsNone(cross)


class SystemImagePullBoundaryAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_credential_bearing_image_is_rejected_before_pull_without_echo(self) -> None:
        from tui.backends.llamacpp.screens import system as llama_system
        from tui.backends.vllm.screens import system as vllm_system

        raw = "https://user:synthetic-secret@registry.example/repo:v1"
        for module in (vllm_system, llama_system):
            with self.subTest(module=module.__name__):
                log = MagicMock()
                screen = SimpleNamespace(
                    query_one=MagicMock(return_value=log),
                    notify=MagicMock(),
                    _refresh_images=MagicMock(),
                )
                pull = AsyncMock()
                with patch.object(module.system_operations, "pull_image", pull):
                    await module.SystemScreen._pull_image.__wrapped__(screen, raw)

                pull.assert_not_awaited()
                rendered = " ".join(
                    str(call.args[0]) for call in log.write.call_args_list
                )
                rendered += " " + " ".join(
                    str(call.args[0]) for call in screen.notify.call_args_list
                )
                self.assertNotIn("synthetic-secret", rendered)


class LlamacppConfigTransactionAuditTests(unittest.TestCase):
    def test_loaded_profile_save_uses_compare_and_swap_snapshot(self) -> None:
        from tui.backends.llamacpp import backend

        snapshot = backend.profile_store.StoredProfile(
            name="p",
            backend="llamacpp",
            port=8080,
        )
        replacement = backend.profile_store.StoredProfile(
            name="p",
            backend="llamacpp",
            port=9090,
        )
        replace = MagicMock(return_value=replacement)
        with (
            patch.object(backend.profile_store, "load_profile", return_value=snapshot),
            patch.object(backend.profile_store, "replace_profile", replace),
        ):
            profile = backend.load_profile("p")
            profile.port = 9090
            backend.save_profile(profile)

        replace.assert_called_once()
        self.assertEqual(replace.call_args.args[0], "p")
        self.assertEqual(replace.call_args.kwargs["expected"], snapshot)
        self.assertEqual(profile._stored_snapshot, replacement)

    def test_config_save_and_delete_hold_shared_storage_transaction(self) -> None:
        from tui.backends.llamacpp import backend

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            path = config_dir / "cfg.yaml"
            path.write_text("ctx-size: 1024\n")
            active = False
            writes: list[Path] = []

            @contextmanager
            def transaction():
                nonlocal active
                self.assertFalse(active)
                active = True
                try:
                    yield
                finally:
                    active = False

            def atomic_write(target: Path, _text: str) -> None:
                self.assertTrue(active)
                writes.append(target)

            with (
                patch.object(backend, "CONFIG_DIR", config_dir),
                patch.object(backend.profile_store, "storage_transaction", transaction),
                patch.object(backend.profile_store, "_atomic_write", atomic_write),
            ):
                backend.save_config(backend.Config(name="cfg", params={"ctx-size": 2048}))
                self.assertFalse(active)
                self.assertEqual(writes, [path])
                backend.delete_config("cfg")
                self.assertFalse(active)
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
