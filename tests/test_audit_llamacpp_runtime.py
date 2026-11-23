from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from tui.backends.llamacpp import backend
from tui.backends.llamacpp import backend_runtime


class LlamacppFlagCacheAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_string_cache_schema_is_rejected_without_character_flags(self) -> None:
        image = "custom/llama:v1"
        identity = "sha256:cache"
        cache_key = hashlib.sha256(f"{image}@{identity}".encode()).hexdigest()[:16]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            (cache_dir / f".llamacpp-params-{cache_key}.json").write_text(
                json.dumps("ctx-size")
            )
            create = AsyncMock()
            with (
                patch.object(backend, "CONFIG_DIR", cache_dir),
                patch(
                    "tui.common.docker.image_identity",
                    AsyncMock(return_value=identity),
                ),
                patch("asyncio.create_subprocess_exec", create),
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid llama.cpp flag cache"):
                    await backend.extract_llama_server_flags(image)

        create.assert_not_awaited()


class LlamacppProbeAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_port_probe_failure_is_not_treated_as_no_conflict(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port=19876)
        with patch.object(
            backend_runtime,
            "_docker_run",
            AsyncMock(return_value=(1, "daemon unavailable")),
        ):
            with self.assertRaisesRegex(RuntimeError, "docker ps"):
                await backend_runtime.check_port_conflict(profile)

    async def test_start_screen_surfaces_gpu_probe_failure(self) -> None:
        from tui.backends.llamacpp.screens import container

        with (
            patch.object(
                container.backend,
                "load_profile",
                return_value=backend.Profile(name="p"),
            ),
            patch.object(
                container,
                "get_dev_build_defaults",
                return_value=("https://example.test/repo.git", "main"),
            ),
        ):
            screen = container.ContainerUpScreen("p")
        gpu_bar = MagicMock()
        screen.query_one = MagicMock(return_value=gpu_bar)
        screen.notify = MagicMock()

        with patch.object(
            container,
            "get_gpu_info",
            AsyncMock(side_effect=RuntimeError("malformed nvidia-smi output")),
        ):
            await container.ContainerUpScreen._fetch_gpu_info.__wrapped__(screen)

        screen.notify.assert_called_once()
        self.assertIn("malformed nvidia-smi output", screen.notify.call_args.args[0])
        gpu_bar.update.assert_called_once()

    async def test_gpu_conflict_probe_failure_is_a_clean_start_error(self) -> None:
        stored = SimpleNamespace(name="p", backend="llamacpp")

        class ProfileStore:
            @staticmethod
            def load_profile(_name: str, _backend: str):
                return stored

        async def no_port_conflict(_profile):
            return None

        async def failed_gpu_probe(_profile):
            raise RuntimeError("GPU conflict scan failed")

        globals_patch = {
            "profile_store": ProfileStore,
            "validate_common_env": lambda _path: (True, []),
            "load_profile": lambda _name: backend.Profile(name="p"),
            "check_port_conflict": no_port_conflict,
            "_gpu_conflict_messages": failed_gpu_probe,
        }
        with patch.dict(backend_runtime.stream_container_up.__globals__, globals_patch):
            events = [
                event async for event in backend_runtime.stream_container_up("p")
            ]

        self.assertEqual(events[-1], ("rc", 5))  # exit_codes.UNREACHABLE
        self.assertIn("GPU conflict scan failed", events[0][1])


class LlamacppSplitInventoryAuditTests(unittest.TestCase):
    @staticmethod
    def _snapshot(cache: Path) -> Path:
        snapshot = (
            cache
            / "hub"
            / "models--org--repo"
            / "snapshots"
            / "revision"
        )
        snapshot.mkdir(parents=True)
        return snapshot

    def test_split_gguf_is_complete_only_after_every_shard_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            snapshot = self._snapshot(cache)
            first = snapshot / "model-00001-of-00002.gguf"
            second = snapshot / "model-00002-of-00002.gguf"
            first.write_bytes(b"a" * 5)

            with patch.object(backend, "_get_hf_cache_dir", return_value=cache):
                self.assertIsNone(
                    backend.find_cached_gguf(
                        "org/repo", "model-00001-of-00002.gguf"
                    )
                )
                self.assertEqual(backend.list_cached_gguf(), [])

                second.write_bytes(b"b" * 7)

                self.assertEqual(
                    backend.find_cached_gguf(
                        "org/repo", "model-00001-of-00002.gguf"
                    ),
                    first,
                )
                inventory = backend.list_cached_gguf()

            self.assertEqual(len(inventory), 1)
            self.assertEqual(inventory[0]["name"], first.name)
            self.assertEqual(inventory[0]["size_bytes"], 12)

    def test_malformed_split_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            snapshot = self._snapshot(cache)
            malformed = "model-00001-of-00000.gguf"
            (snapshot / malformed).write_bytes(b"bad")

            with patch.object(backend, "_get_hf_cache_dir", return_value=cache):
                with self.assertRaisesRegex(ValueError, "split GGUF"):
                    backend.find_cached_gguf("org/repo", malformed)
                with self.assertRaisesRegex(ValueError, "split GGUF"):
                    backend.list_cached_gguf()


class LlamacppDevImageAuditTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, *, exists: bool, matches: bool) -> list[tuple[str, object]]:
        stored = SimpleNamespace(
            name="p",
            backend="llamacpp",
            container_name="p",
            image_tag="",
        )
        profile = backend.Profile(name="p", container_name="p", port=8080)
        builds: list[tuple[tuple, dict]] = []
        match_calls: list[tuple] = []

        class ProfileStore:
            @staticmethod
            def load_profile(_name: str, _backend: str):
                return stored

            @staticmethod
            def render_env_for_profile(_name: str, _backend: str):
                return None

            @staticmethod
            @contextmanager
            def storage_transaction():
                yield

            @staticmethod
            def render_env(_stored):
                return None

        async def image_matches(_spec, tag: str, repo: str, branch: str) -> bool:
            match_calls.append((tag, repo, branch))
            return matches

        async def no_conflict(_profile):
            return None

        async def no_gpu(_profile):
            return []

        async def build(*args, **kwargs):
            builds.append((args, kwargs))
            yield ("rc", 0)

        async def stop_after_image(_profile, _image):
            return 1, "stop"

        globals_patch = {
            "profile_store": ProfileStore,
            "validate_common_env": lambda _path: (True, []),
            "load_profile": lambda _name: profile,
            "check_port_conflict": no_conflict,
            "_gpu_conflict_messages": no_gpu,
            "_ensure_profile_config": lambda _stored, _profile: (True, []),
            "get_dev_build_defaults": lambda: ("https://example.test/repo.git", "main"),
            "_stream_build_dev_image": build,
            "_render_override": stop_after_image,
            "_resolve_pending_quant": AsyncMock(return_value=""),
            "_docker_run": AsyncMock(return_value=(0, "")),
        }
        with (
            patch.dict(backend_runtime.stream_container_up.__globals__, globals_patch),
            patch.object(
                backend_runtime.dev_build,
                "image_exists_locally",
                AsyncMock(return_value=exists),
            ),
            patch.object(
                backend_runtime.dev_build,
                "image_matches",
                side_effect=image_matches,
            ),
        ):
            events = [
                event
                async for event in backend_runtime.stream_container_up(
                    "p", use_dev=True, tag="audit-tag"
                )
            ]

        self.builds = builds
        self.match_calls = match_calls
        return events

    async def test_explicit_missing_dev_tag_triggers_one_off_build(self) -> None:
        events = await self._run(exists=False, matches=False)
        self.assertEqual(len(self.builds), 1)
        self.assertEqual(
            [event for event in events if event[0] == "rc"],
            [("rc", 1)],
        )

    async def test_explicit_existing_dev_tag_checks_source_label(self) -> None:
        await self._run(exists=True, matches=False)
        self.assertEqual(
            self.match_calls,
            [("audit-tag", "https://example.test/repo.git", "main")],
        )
        self.assertEqual(len(self.builds), 1)


class LlamacppRendererAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "llamacpp"
            / "render-override.py"
        )
        spec = importlib.util.spec_from_file_location("audit_llamacpp_renderer", script)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_special_multi_value_keys_reject_non_string_or_list_values(self) -> None:
        for key in ("extra-args", "override-tensors"):
            for value in (1, True, {"bad": "shape"}):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        self.module.render_command(
                            {key: value}, hf_repo="org/repo", hf_file="m.gguf"
                        )

    def test_main_maps_invalid_special_value_to_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "cfg.yaml").write_text("extra-args: 1\n")
            stderr = StringIO()
            stored = SimpleNamespace(
                config_name="cfg",
                hf_repo="org/repo",
                hf_file="m.gguf",
            )
            with (
                patch.object(self.module, "CONFIG_DIR", config_dir),
                patch.object(self.module, "RUNTIME_DIR", root / "runtime"),
                patch.object(
                    self.module.profile_store,
                    "load_profile",
                    return_value=stored,
                ),
                patch.object(sys, "argv", ["render-override.py", "p"]),
                redirect_stderr(stderr),
            ):
                rc = self.module.main()

        self.assertEqual(rc, 1)
        self.assertIn("config 렌더 실패", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


class LlamacppDiskAuditTests(unittest.TestCase):
    def test_hf_cache_path_is_required_by_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(backend, "ROOT", Path(tmpdir)):
                with self.assertRaisesRegex(ValueError, "HF_CACHE_PATH"):
                    backend._get_hf_cache_dir()

    def test_hf_cache_path_must_resolve_to_an_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env.common").write_text("HF_CACHE_PATH=relative/cache\n")
            with patch.object(backend, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "absolute path"):
                    backend._get_hf_cache_dir()

    def test_inventory_os_error_has_backend_error_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir)
            self._make_cache_entry(cache)
            with (
                patch.object(backend, "_get_hf_cache_dir", return_value=cache),
                patch.object(Path, "stat", side_effect=OSError("inventory failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "inventory failed"):
                    backend.list_cached_gguf()

    @staticmethod
    def _make_cache_entry(cache: Path) -> None:
        snapshot = (
            cache / "hub" / "models--org--repo" / "snapshots" / "revision"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model.gguf").write_bytes(b"data")


class LlamacppCascadeAuditTests(unittest.TestCase):
    def test_cascade_uses_profile_name_for_implicit_config_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "p.yaml"
            config_path.write_text("ctx-size: 1024\n")
            profiles_path = root / "profiles.yaml"
            profiles_path.write_text(
                "version: 1\nprofiles:\n  - name: p\n    backend: llamacpp\n"
            )

            with (
                patch.object(backend, "CONFIG_DIR", config_dir),
                patch.object(
                    backend.profile_store,
                    "PROFILES_YAML",
                    profiles_path,
                ),
                patch.object(
                    backend.profile_store,
                    "RUNTIME_DIR",
                    root / "runtime",
                ),
            ):
                backend.delete_profile("p", delete_config_too=True)

            self.assertFalse(config_path.exists())
            self.assertEqual(backend.yaml.safe_load(profiles_path.read_text())["profiles"], [])


class LlamacppComposeAuditTests(unittest.TestCase):
    def test_required_runtime_variables_fail_compose_config(self) -> None:
        compose = (
            Path(__file__).resolve().parents[1]
            / "compose"
            / "llamacpp"
            / "docker-compose.yaml"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.env"
            profile_path.write_text("")
            required = {
                "CONTAINER_NAME": "profile",
                "LLAMA_PORT": "8080",
                "GPU_ID": "0",
                "HF_CACHE_PATH": str(Path(tmpdir) / "cache"),
                "PROFILE_PATH": str(profile_path),
                "CONFIG_NAME": "cfg",
            }
            for missing in (
                "CONTAINER_NAME",
                "HF_CACHE_PATH",
                "PROFILE_PATH",
                "CONFIG_NAME",
            ):
                with self.subTest(missing=missing):
                    env = {
                        "PATH": os.environ["PATH"],
                        **{key: value for key, value in required.items() if key != missing},
                    }
                    result = subprocess.run(
                        ["docker", "compose", "-f", str(compose), "config"],
                        cwd=Path(tmpdir),
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    output = result.stdout + result.stderr
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn(missing, output)
