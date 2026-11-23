from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tui.backends.vllm import backend_runtime
from tui.backends.vllm.backend_common import Profile
from tui.common import dev_build, prepare, system_operations
from tui.common.env import validate_common_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT_WRAPPER = PROJECT_ROOT / "scripts" / "vllm" / "entrypoint-wrapper.sh"


async def _events(stream) -> tuple[list[str], int]:
    logs: list[str] = []
    rc = -1
    async for kind, value in stream:
        if kind == "rc":
            rc = int(value)
        elif kind == "log":
            logs.append(str(value))
    return logs, rc


class VllmDevComposeAuditTests(unittest.TestCase):
    def _environment(self, profile_path: Path) -> dict[str, str]:
        env = os.environ.copy()
        for key in (
            "CONFIG_NAME",
            "CONTAINER_NAME",
            "GPU_ID",
            "HF_CACHE_PATH",
            "PROFILE_PATH",
            "TENSOR_PARALLEL_SIZE",
            "VLLM_DEV_TAG",
            "VLLM_PORT",
        ):
            env.pop(key, None)
        env.update(
            {
                "CONFIG_NAME": "audit",
                "CONTAINER_NAME": "audit",
                "GPU_ID": "0",
                "HF_CACHE_PATH": "/tmp/llmux-audit-cache",
                "PROFILE_PATH": str(profile_path),
                "TENSOR_PARALLEL_SIZE": "1",
                "VLLM_DEV_TAG": "audit",
                "VLLM_PORT": "18000",
            }
        )
        return env

    def _render(self, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        profile = Profile(name="audit", config_name="audit")
        compose_args = backend_runtime._compose_files(profile, use_dev=True)
        self.assertEqual(
            compose_args[:4],
            [
                "-f",
                str(PROJECT_ROOT / "compose" / "vllm" / "docker-compose.yaml"),
                "-f",
                str(PROJECT_ROOT / "compose" / "vllm" / "docker-compose.dev.yaml"),
            ],
        )
        return subprocess.run(
            ["docker", "compose", *compose_args, "config", "-q"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def test_runtime_dev_compose_renders_base_then_dev(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "audit.env"
            profile_path.write_text("")
            result = self._render(self._environment(profile_path))
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_required_runtime_values_fail_compose_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "audit.env"
            profile_path.write_text("")
            baseline = self._environment(profile_path)
            for key in ("CONFIG_NAME", "TENSOR_PARALLEL_SIZE", "PROFILE_PATH"):
                with self.subTest(key=key):
                    env = baseline.copy()
                    env.pop(key)
                    result = self._render(env)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(key, result.stderr + result.stdout)


class CredentialBoundaryAuditTests(unittest.IsolatedAsyncioTestCase):
    def test_ssh_username_is_not_treated_as_a_url_password(self) -> None:
        repo_url = "ssh://git@git.example/org/repo.git"
        transport_url, env, temp_dir = dev_build._git_transport(repo_url)
        self.assertEqual(transport_url, repo_url)
        self.assertIsNone(env)
        self.assertIsNone(temp_dir)

    async def test_extra_pip_credentials_are_rejected_before_start_logs(self) -> None:
        secret = "LLMUX_EXTRA_PIP_SENTINEL"
        profile = Profile(
            name="audit",
            container_name="audit",
            config_name="audit",
            extra_pip_packages=f"pkg @ https://user:{secret}@packages.example/pkg.whl",
        )

        async def no_conflict(_profile):
            return None

        async def no_gpu_conflicts(_profile):
            return []

        async def stop_build(*_args, **_kwargs):
            yield "rc", 1

        with (
            patch.object(backend_runtime.profile_store, "load_profile", return_value=object()),
            patch.object(backend_runtime.profile_store, "render_env", return_value=Path("audit.env")),
            patch.object(backend_runtime, "load_profile", return_value=profile),
            patch.object(backend_runtime, "check_port_conflict", side_effect=no_conflict),
            patch.object(backend_runtime, "_ensure_common_env", return_value=(True, [])),
            patch.object(backend_runtime, "_ensure_profile_config", return_value=(True, [])),
            patch.object(
                backend_runtime,
                "_render_profile_snapshot",
                return_value=(profile, Path("audit.env")),
            ),
            patch.object(backend_runtime, "_gpu_conflict_messages", side_effect=no_gpu_conflicts),
            patch.object(backend_runtime.dev_build, "image_exists_locally", AsyncMock(return_value=False)),
            patch.object(backend_runtime, "_stream_build_dev_image", side_effect=stop_build),
        ):
            logs, rc = await _events(
                backend_runtime.stream_container_up("audit", use_dev=True)
            )

        output = "\n".join(logs)
        self.assertEqual(rc, 1)
        self.assertIn("credential", output.lower())
        self.assertNotIn(secret, output)

    async def test_git_credentials_are_rejected_before_subprocess(self) -> None:
        secret = "LLMUX_GIT_SENTINEL"
        repo_url = f"https://builder:{secret}@git.example/org/repo.git"
        clean_url = "https://git.example/org/repo.git"
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            spec = dev_build.DevBuildSpec(
                backend="audit",
                image_prefix="audit",
                src_dir=source,
                default_repo_url=clean_url,
            )
            with patch.object(
                dev_build,
                "_stream",
                side_effect=AssertionError("git must not be started"),
            ), patch.object(
                dev_build,
                "_run",
                side_effect=AssertionError("git must not be started"),
            ):
                logs, rc = await _events(
                    dev_build.clone_or_update(spec, repo_url, "main")
                )

            self.assertEqual(rc, 1)
            self.assertIn("credential", "\n".join(logs).lower())
            self.assertNotIn(secret, "\n".join(logs))

    async def test_git_failure_output_redacts_repository_credentials(self) -> None:
        secret = "LLMUX_GIT_ERROR_SENTINEL"
        repo_url = f"https://builder:{secret}@git.example/org/repo.git"
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            spec = dev_build.DevBuildSpec(
                backend="audit",
                image_prefix="audit",
                src_dir=source,
                default_repo_url="https://git.example/org/repo.git",
            )

            async def fake_stream(args, *, cwd=None, env=None):
                (source / ".git").mkdir(parents=True)
                yield "rc", 0

            async def fake_run(*args, cwd=None, timeout=30, env=None):
                if args[1] == "fetch":
                    return 1, f"fatal: authentication failed for {repo_url}"
                return 0, ""

            with patch.object(dev_build, "_stream", side_effect=fake_stream), patch.object(
                dev_build, "_run", side_effect=fake_run
            ):
                logs, rc = await _events(dev_build.clone_or_update(spec, repo_url, "main"))

        self.assertEqual(rc, 1)
        self.assertNotIn(secret, "\n".join(logs))

    async def test_git_query_credentials_are_rejected_before_subprocess(self) -> None:
        secret = "LLMUX_GIT_QUERY_SENTINEL"
        repo_url = f"https://git.example/org/repo.git?access_token={secret}"
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = dev_build.DevBuildSpec(
                backend="audit",
                image_prefix="audit",
                src_dir=Path(tmpdir) / "source",
                default_repo_url="https://git.example/org/repo.git",
            )
            with patch.object(
                dev_build,
                "_stream",
                side_effect=AssertionError("git must not be started"),
            ):
                logs, rc = await _events(
                    dev_build.clone_or_update(spec, repo_url, "main")
                )

        self.assertEqual(rc, 1)
        self.assertNotIn(secret, "\n".join(logs))

    async def test_existing_origin_credentials_are_removed(self) -> None:
        secret = "LLMUX_GIT_CONFIG_SENTINEL"
        repo_url = f"https://builder:{secret}@git.example/org/repo.git"
        clean_url = "https://git.example/org/repo.git"
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            (source / ".git").mkdir(parents=True)
            config_path = source / ".git" / "config"
            config_path.write_text(f"url = {repo_url}\n")
            spec = dev_build.DevBuildSpec(
                backend="audit",
                image_prefix="audit",
                src_dir=source,
                default_repo_url=clean_url,
            )

            async def fake_run(*args, cwd=None, timeout=30, env=None):
                if args[1:4] == ("remote", "get-url", "origin"):
                    return 0, repo_url
                if args[1:4] == ("remote", "set-url", "origin"):
                    config_path.write_text(f"url = {args[4]}\n")
                    return 0, ""
                if args[1] == "rev-parse":
                    return 0, "abcdef\n"
                return 0, ""

            with patch.object(dev_build, "_run", side_effect=fake_run):
                await _events(dev_build.clone_or_update(spec, clean_url, "main"))

            config = config_path.read_text()
            self.assertIn(clean_url, config)
            self.assertNotIn(secret, config)

    def test_entrypoint_rejects_credentials_without_logging_or_argv(self) -> None:
        secret = "LLMUX_ENTRYPOINT_SENTINEL"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            captured = tmp / "pip-argv"
            for command in ("pip", "vllm"):
                executable = tmp / command
                executable.write_text(
                    "#!/bin/sh\n"
                    + (f"printf '%s\\n' \"$@\" > {captured}\n" if command == "pip" else "")
                    + "exit 0\n"
                )
                executable.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{tmp}:{env['PATH']}"
            env["EXTRA_PIP_PACKAGES"] = (
                f"pkg @ https://user:{secret}@packages.example/pkg.whl"
            )
            result = subprocess.run(
                ["bash", str(ENTRYPOINT_WRAPPER)],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            output = result.stdout + result.stderr
            captured_argv = captured.read_text() if captured.exists() else ""
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(secret, output)
        self.assertNotIn(secret, captured_argv)

    def test_entrypoint_passes_public_packages_through_requirements_file(self) -> None:
        package = "audit-package==1.2.3"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            captured = tmp / "pip-argv"
            for command in ("pip", "vllm"):
                executable = tmp / command
                executable.write_text(
                    "#!/bin/sh\n"
                    + (f"printf '%s\\n' \"$@\" > {captured}\n" if command == "pip" else "")
                    + "exit 0\n"
                )
                executable.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{tmp}:{env['PATH']}"
            env["EXTRA_PIP_PACKAGES"] = package
            result = subprocess.run(
                ["bash", str(ENTRYPOINT_WRAPPER)],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            captured_argv = captured.read_text()

        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertNotIn(package, result.stdout + result.stderr)
        self.assertNotIn(package, captured_argv)
        self.assertIn("-r", captured_argv)


class PrepareContractAuditTests(unittest.IsolatedAsyncioTestCase):
    def _write_common(self, path: Path, downloader: str | None) -> None:
        lines = ["HF_CACHE_PATH=/tmp/llmux-cache", "VLLM_USE_V2_MODEL_RUNNER=1"]
        if downloader is not None:
            lines.append(f"PREPARE_DOWNLOADER_IMAGE={downloader}")
        path.write_text("\n".join(lines) + "\n")
        path.chmod(0o600)

    def test_downloader_image_is_required_and_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            common = Path(tmpdir) / ".env.common"
            for image_ref in (None, "vllm/vllm-openai", "vllm/vllm-openai:latest"):
                with self.subTest(image_ref=image_ref):
                    self._write_common(common, image_ref)
                    ok, messages = validate_common_env(
                        common, require_downloader_image=True
                    )
                    self.assertFalse(ok)
                    self.assertIn("PREPARE_DOWNLOADER_IMAGE", "\n".join(messages))

            self._write_common(common, "vllm/vllm-openai:v0.27.1")
            ok, messages = validate_common_env(
                common, require_downloader_image=True
            )
            self.assertTrue(ok, messages)
            self.assertEqual(messages, [])

    async def test_prepare_rejects_invalid_downloader_before_docker(self) -> None:
        probe = AsyncMock(side_effect=AssertionError("docker must not be called"))
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            prepare, "downloader_image", return_value="vllm/vllm-openai"
        ), patch.object(prepare, "image_present", probe):
            logs, rc = await _events(
                prepare.stream_llamacpp_download(
                    hf_repo="o/r",
                    hf_file="m.gguf",
                    cache_path=tmpdir,
                    token="",
                    container_name="audit-prepare",
                )
            )

        self.assertEqual(rc, 1)
        self.assertIn("PREPARE_DOWNLOADER_IMAGE", "\n".join(logs))
        probe.assert_not_awaited()

    async def test_cached_prepare_still_validates_required_downloader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = (
                Path(tmpdir)
                / "hub"
                / "models--o--r"
                / "snapshots"
                / "revision"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "m.gguf").write_text("gguf")
            with patch.object(
                prepare, "downloader_image", return_value="vllm/vllm-openai:latest"
            ):
                logs, rc = await _events(
                    prepare.stream_llamacpp_download(
                        hf_repo="o/r",
                        hf_file="m.gguf",
                        cache_path=tmpdir,
                        token="",
                        container_name="audit-prepare",
                    )
                )

        self.assertEqual(rc, 1)
        self.assertIn("PREPARE_DOWNLOADER_IMAGE", "\n".join(logs))

    def test_invalid_split_shard_coordinates_raise(self) -> None:
        for hf_file in (
            "m-00001-of-00000.gguf",
            "m-00000-of-00001.gguf",
            "m-00003-of-00002.gguf",
        ):
            with self.subTest(hf_file=hf_file), self.assertRaisesRegex(
                ValueError, "invalid split GGUF shard"
            ):
                prepare.gguf_shard_names(hf_file)

        self.assertEqual(
            prepare.gguf_shard_names("m-00002-of-00003.gguf"),
            [
                "m-00001-of-00003.gguf",
                "m-00002-of-00003.gguf",
                "m-00003-of-00003.gguf",
            ],
        )

    async def test_invalid_split_shard_is_a_stable_prepare_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshots = Path(tmpdir) / "hub" / "models--o--r" / "snapshots"
            snapshots.mkdir(parents=True)
            logs, rc = await _events(
                prepare.stream_llamacpp_download(
                    hf_repo="o/r",
                    hf_file="m-00001-of-00000.gguf",
                    cache_path=tmpdir,
                    token="",
                    container_name="audit-prepare",
                )
            )

        self.assertEqual(rc, 1)
        self.assertIn("invalid split GGUF shard", "\n".join(logs))


class DevBuildFailureAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_git_and_docker_are_stable_events(self) -> None:
        for executable in ("git", "docker"):
            with self.subTest(executable=executable), patch(
                "tui.common.dev_build.asyncio.create_subprocess_exec",
                side_effect=FileNotFoundError(executable),
            ):
                logs, rc = await _events(dev_build._stream([executable, "version"]))
            self.assertNotEqual(rc, 0)
            self.assertIn(executable, "\n".join(logs))

    async def test_subprocess_start_failure_is_a_stable_event(self) -> None:
        with patch(
            "tui.common.dev_build.asyncio.create_subprocess_exec",
            side_effect=PermissionError("execution denied"),
        ):
            logs, rc = await _events(dev_build._stream(["git", "fetch"]))

        self.assertEqual(rc, 126)
        self.assertIn("execution denied", "\n".join(logs))

    async def test_system_operation_collects_subprocess_failure(self) -> None:
        async def failed_build(*_args, **_kwargs):
            yield "log", "docker build failed"
            yield "rc", 125

        with patch(
            "tui.backends.vllm.backend_runtime._stream_build_dev_image",
            side_effect=failed_build,
        ):
            rc, logs = await system_operations.build_dev_image(
                "vllm",
                repo_url="https://git.example/vllm.git",
                branch="main",
                custom_tag="audit",
            )

        self.assertEqual(rc, 125)
        self.assertEqual(logs, ["docker build failed"])


class DevImageIdentityAuditTests(unittest.IsolatedAsyncioTestCase):
    async def _start(self, *, exists: bool, matches: bool):
        profile = Profile(
            name="audit", container_name="audit", config_name="audit"
        )
        builds: list[tuple[tuple, dict]] = []

        async def no_conflict(_profile):
            return None

        async def no_gpu_conflicts(_profile):
            return []

        async def build(*args, **kwargs):
            builds.append((args, kwargs))
            yield "rc", 0

        async def compose(*_args, **_kwargs):
            yield "rc", 1

        with (
            patch.object(backend_runtime.profile_store, "load_profile", return_value=object()),
            patch.object(
                backend_runtime.profile_store,
                "render_env_for_profile",
                return_value=Path("audit.env"),
            ),
            patch.object(backend_runtime, "load_profile", return_value=profile),
            patch.object(backend_runtime, "check_port_conflict", side_effect=no_conflict),
            patch.object(backend_runtime, "_ensure_common_env", return_value=(True, [])),
            patch.object(backend_runtime, "_ensure_profile_config", return_value=(True, [])),
            patch.object(
                backend_runtime,
                "_render_profile_snapshot",
                return_value=(profile, Path("audit.env")),
            ),
            patch.object(backend_runtime, "_gpu_conflict_messages", side_effect=no_gpu_conflicts),
            patch.object(backend_runtime.dev_build, "image_exists_locally", AsyncMock(return_value=exists)),
            patch.object(backend_runtime, "_dev_image_matches", AsyncMock(return_value=matches)),
            patch.object(backend_runtime, "_stream_build_dev_image", side_effect=build),
            patch.object(backend_runtime, "_compose_env", return_value={}),
            patch.object(backend_runtime, "stream_command", side_effect=compose),
        ):
            events = [
                event
                async for event in backend_runtime.stream_container_up(
                    "audit",
                    use_dev=True,
                    tag="trusted",
                    repo_url="https://git.example/vllm.git",
                    branch="release",
                )
            ]
        logs = [str(value) for kind, value in events if kind == "log"]
        rc = next(int(value) for kind, value in events if kind == "rc")
        return logs, rc, builds, events

    async def test_explicit_tag_missing_image_builds_that_tag(self) -> None:
        _logs, _rc, builds, events = await self._start(exists=False, matches=False)
        self.assertEqual(len(builds), 1)
        self.assertEqual(builds[0][1]["custom_tag"], "trusted")
        self.assertEqual([event for event in events if event[0] == "rc"], [("rc", 1)])

    async def test_explicit_tag_source_mismatch_rebuilds_that_tag(self) -> None:
        _logs, _rc, builds, events = await self._start(exists=True, matches=False)
        self.assertEqual(len(builds), 1)
        self.assertEqual(builds[0][1]["custom_tag"], "trusted")
        self.assertEqual([event for event in events if event[0] == "rc"], [("rc", 1)])

    async def test_explicit_tag_matching_source_is_reused(self) -> None:
        _logs, _rc, builds, _events = await self._start(exists=True, matches=True)
        self.assertEqual(builds, [])


class VllmGpuProbeAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_screen_reports_gpu_probe_failure(self) -> None:
        from tui.backends.vllm.screens import container

        updates: list[str] = []
        notifications: list[tuple[str, dict]] = []
        screen = SimpleNamespace(
            _gpu_error_notified=False,
            query_one=lambda *_args, **_kwargs: SimpleNamespace(
                update=lambda value: updates.append(str(value))
            ),
            notify=lambda message, **kwargs: notifications.append((str(message), kwargs)),
        )
        method = container.ContainerUpScreen.__dict__["_fetch_gpu_info"].__wrapped__
        with patch.object(
            container, "get_gpu_info", AsyncMock(side_effect=RuntimeError("nvidia-smi malformed"))
        ):
            await method(screen)

        self.assertTrue(updates)
        self.assertIn("failed", updates[-1].lower())
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0][1]["severity"], "error")


class VllmConflictProbeAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_port_probe_failure_is_not_reported_as_available(self) -> None:
        profile = Profile(name="audit", container_name="audit", config_name="audit")
        with patch.object(
            backend_runtime,
            "run_command",
            AsyncMock(return_value=(1, "docker daemon unavailable")),
        ):
            with self.assertRaisesRegex(RuntimeError, "docker daemon unavailable"):
                await backend_runtime.check_port_conflict(profile)

    async def test_runtime_start_maps_conflict_probe_failure_to_rc(self) -> None:
        profile = Profile(name="audit", container_name="audit", config_name="audit")
        with (
            patch.object(backend_runtime.profile_store, "load_profile", return_value=object()),
            patch.object(backend_runtime, "load_profile", return_value=profile),
            patch.object(
                backend_runtime,
                "check_port_conflict",
                AsyncMock(side_effect=RuntimeError("docker ps unavailable")),
            ),
        ):
            logs, rc = await _events(backend_runtime.stream_container_up("audit"))

        self.assertEqual(rc, 5)  # exit_codes.UNREACHABLE
        self.assertIn("docker ps unavailable", "\n".join(logs))

    async def test_runtime_start_maps_gpu_probe_failure_to_rc(self) -> None:
        profile = Profile(name="audit", container_name="audit", config_name="audit")
        with (
            patch.object(backend_runtime.profile_store, "load_profile", return_value=object()),
            patch.object(
                backend_runtime.profile_store,
                "render_env_for_profile",
                return_value=Path("audit.env"),
            ),
            patch.object(backend_runtime, "load_profile", return_value=profile),
            patch.object(backend_runtime, "check_port_conflict", AsyncMock(return_value=None)),
            patch.object(backend_runtime, "_ensure_common_env", return_value=(True, [])),
            patch.object(backend_runtime, "_ensure_profile_config", return_value=(True, [])),
            patch.object(
                backend_runtime,
                "_render_profile_snapshot",
                return_value=(profile, Path("audit.env")),
            ),
            patch.object(
                backend_runtime,
                "_gpu_conflict_messages",
                AsyncMock(side_effect=RuntimeError("GPU inventory unavailable")),
            ),
        ):
            logs, rc = await _events(backend_runtime.stream_container_up("audit"))

        self.assertEqual(rc, 5)  # exit_codes.UNREACHABLE
        self.assertIn("GPU inventory unavailable", "\n".join(logs))


if __name__ == "__main__":
    unittest.main()
