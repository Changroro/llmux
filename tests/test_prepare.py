"""`prepare` — download + render without starting the server."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tui.common import prepare


async def _drain(agen) -> tuple[list[str], int]:
    logs: list[str] = []
    rc = -1
    async for kind, data in agen:
        if kind == "rc":
            rc = int(data)
        else:
            logs.append(str(data))
    return logs, rc


class ImageProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_image_is_false(self) -> None:
        with patch.object(
            prepare,
            "_run",
            AsyncMock(return_value=(1, "Error response from daemon: No such image: x")),
        ):
            self.assertFalse(await prepare.image_present("x"))

    async def test_docker_probe_failure_is_not_reported_as_missing(self) -> None:
        with patch.object(
            prepare,
            "_run",
            AsyncMock(return_value=(1, "Cannot connect to the Docker daemon")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot connect"):
                await prepare.image_present("x")

    async def test_missing_docker_binary_is_not_reported_as_missing(self) -> None:
        with patch.object(
            prepare,
            "_run",
            AsyncMock(return_value=(-1, "Executable not found: docker")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Executable not found"):
                await prepare.image_present("x")


class StreamLinesTests(unittest.IsolatedAsyncioTestCase):
    async def test_splits_on_newline_and_reports_rc(self) -> None:
        logs, rc = await _drain(
            prepare.stream_lines(["printf", "one\ntwo\n"])
        )
        self.assertEqual(logs, ["one", "two"])
        self.assertEqual(rc, 0)

    async def test_carriage_return_redraws_are_throttled(self) -> None:
        # A progress bar rewrites one line with CR; without throttling every
        # redraw would land in the log.
        logs, rc = await _drain(
            prepare.stream_lines(["printf", "10pct\r20pct\r30pct\rdone\n"])
        )
        self.assertEqual(rc, 0)
        self.assertIn("done", logs)
        self.assertLessEqual(len([line for line in logs if line.endswith("pct")]), 1)

    async def test_missing_binary_is_reported_not_raised(self) -> None:
        logs, rc = await _drain(prepare.stream_lines(["llmux-no-such-binary"]))
        self.assertEqual(rc, -1)
        self.assertTrue(any("not found" in line for line in logs))


class GgufCacheTests(unittest.TestCase):
    def _cache(self, tmp: Path, *, filename: str = "m.gguf") -> Path:
        snap = tmp / "hub" / "models--o--r" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        target = snap / filename
        target.write_text("gguf")
        return target

    def test_finds_completed_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            expected = self._cache(tmp)
            self.assertEqual(
                prepare.gguf_in_cache(str(tmp), "o/r", "m.gguf"), expected
            )

    def test_a_lone_first_shard_is_not_a_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._cache(tmp, filename="m-00001-of-00003.gguf")
            self.assertIsNone(
                prepare.gguf_in_cache(str(tmp), "o/r", "m-00001-of-00003.gguf")
            )

    def test_every_shard_present_is_a_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            first = self._cache(tmp, filename="m-00001-of-00003.gguf")
            snap = first.parent
            (snap / "m-00002-of-00003.gguf").write_text("gguf")
            (snap / "m-00003-of-00003.gguf").write_text("gguf")
            self.assertEqual(
                prepare.gguf_in_cache(str(tmp), "o/r", "m-00001-of-00003.gguf"), first
            )

    def test_missing_file_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(
                prepare.gguf_in_cache(tmpdir, "o/r", "m.gguf")
            )


class PrepareWorkersTests(unittest.TestCase):
    def test_unset_leaves_the_library_default(self) -> None:
        with patch.object(prepare, "_common", lambda: {}):
            self.assertIsNone(prepare.prepare_max_workers())
            self.assertEqual(prepare.workers_env(None), [])

    def test_env_file_value_is_injected(self) -> None:
        with patch.object(prepare, "_common", lambda: {"PREPARE_MAX_WORKERS": "2"}):
            self.assertEqual(prepare.workers_env(None), ["-e", "LLMUX_PREPARE_WORKERS=2"])

    def test_an_explicit_count_wins_over_the_env_file(self) -> None:
        with patch.object(prepare, "_common", lambda: {"PREPARE_MAX_WORKERS": "2"}):
            self.assertEqual(prepare.workers_env(8), ["-e", "LLMUX_PREPARE_WORKERS=8"])

    def test_a_junk_value_is_rejected_loudly(self) -> None:
        with patch.object(prepare, "_common", lambda: {"PREPARE_MAX_WORKERS": "many"}):
            with self.assertRaises(ValueError):
                prepare.prepare_max_workers()


class GgufShardNameTests(unittest.TestCase):
    def test_a_split_shard_expands_to_every_sibling(self) -> None:
        self.assertEqual(
            prepare.gguf_shard_names("UD-Q3_K_XL/m-00002-of-00003.gguf"),
            [
                "UD-Q3_K_XL/m-00001-of-00003.gguf",
                "UD-Q3_K_XL/m-00002-of-00003.gguf",
                "UD-Q3_K_XL/m-00003-of-00003.gguf",
            ],
        )

    def test_an_unsplit_file_is_taken_as_is(self) -> None:
        self.assertEqual(prepare.gguf_shard_names("m.gguf"), ["m.gguf"])


async def _present(_image_ref: str) -> bool:
    return True


class LlamacppDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_cached_gguf_skips_docker_entirely(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            snap = tmp / "hub" / "models--o--r" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            (snap / "m.gguf").write_text("gguf")

            async def fail(*args, **kwargs):
                raise AssertionError("docker must not run for a cached model")

            with patch.object(prepare, "_run", fail), patch.object(
                prepare,
                "downloader_image",
                return_value="vllm/vllm-openai:v0.27.1",
            ):
                logs, rc = await _drain(
                    prepare.stream_llamacpp_download(
                        hf_repo="o/r", hf_file="m.gguf",
                        cache_path=str(tmp), token="", container_name="c-prepare",
                    )
                )
            self.assertEqual(rc, 0)
            self.assertTrue(any("m.gguf" in line for line in logs))

    async def test_every_shard_is_requested_in_one_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            snap = tmp / "hub" / "models--o--r" / "snapshots" / "abc" / "UD-Q3"
            snap.mkdir(parents=True)
            hf_file = "UD-Q3/m-00001-of-00002.gguf"
            seen: list[list[str]] = []
            seen_env: list[dict[str, str] | None] = []

            async def fake_run(*args, **kwargs):
                return 0, ""

            async def fake_stream(args, *, env=None):
                seen.append(args)
                seen_env.append(env)
                yield ("log", "Fetching 2 files")
                (snap / "m-00001-of-00002.gguf").write_text("gguf")
                (snap / "m-00002-of-00002.gguf").write_text("gguf")
                yield ("rc", 0)

            with patch.object(prepare, "_run", fake_run), \
                 patch.object(prepare, "stream_lines", fake_stream), \
                 patch.object(prepare, "downloader_image", lambda: "downloader:img"), \
                 patch.object(prepare, "image_present", _present):
                logs, rc = await _drain(
                    prepare.stream_llamacpp_download(
                        hf_repo="o/r", hf_file=hf_file,
                        cache_path=str(tmp), token="tok", container_name="c-prepare",
                    )
                )

            self.assertEqual(rc, 0)
            self.assertTrue(any("Fetching 2 files" in line for line in logs))
            args = seen[0]
            self.assertIn(
                "LLMUX_PREPARE_ALLOW=UD-Q3/m-00001-of-00002.gguf,UD-Q3/m-00002-of-00002.gguf",
                args,
            )
            self.assertIn("downloader:img", args)
            self.assertNotIn("-hf", args)
            self.assertNotIn("tok", " ".join(args))
            self.assertEqual(seen_env[0]["HF_TOKEN"], "tok")
            self.assertEqual(seen_env[0]["HUGGING_FACE_HUB_TOKEN"], "tok")

    async def test_vllm_token_is_not_exposed_in_docker_arguments(self) -> None:
        seen: list[tuple[list[str], dict[str, str] | None]] = []

        async def fake_run(*args, **kwargs):
            return 0, ""

        async def fake_stream(args, *, env=None):
            seen.append((args, env))
            yield ("rc", 0)

        with patch.object(prepare, "_run", fake_run), patch.object(
            prepare, "stream_lines", fake_stream
        ):
            _, rc = await _drain(
                prepare.stream_vllm_download(
                    image_ref="vllm:test",
                    model_id="o/r",
                    cache_path="/tmp/cache",
                    token="hf_secret",
                    container_name="prepare-test",
                )
            )

        self.assertEqual(rc, 0)
        args, env = seen[0]
        self.assertNotIn("hf_secret", " ".join(args))
        self.assertEqual(env["HF_TOKEN"], "hf_secret")
        self.assertEqual(env["HUGGING_FACE_HUB_TOKEN"], "hf_secret")

    async def test_exit_before_download_completes_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async def fake_run(*args, **kwargs):
                return 0, ""

            async def fake_stream(_args):
                yield ("log", "error: 404 not found")
                yield ("rc", 0)

            with patch.object(prepare, "_run", fake_run), \
                 patch.object(prepare, "stream_lines", fake_stream), \
                 patch.object(prepare, "downloader_image", lambda: "downloader:img"), \
                 patch.object(prepare, "image_present", _present):
                _, rc = await _drain(
                    prepare.stream_llamacpp_download(
                        hf_repo="o/r", hf_file="m.gguf",
                        cache_path=tmpdir, token="", container_name="c-prepare",
                    )
                )
            self.assertEqual(rc, 1)


class PrepareRuntimeGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_vllm_unknown_profile_fails(self) -> None:
        from tui.backends.vllm import backend_runtime as vrt

        with patch.object(vrt.profile_store, "load_profile", lambda *a, **k: None):
            logs, rc = await _drain(vrt.stream_container_prepare("nope"))
        self.assertEqual(rc, 1)
        self.assertTrue(any("nope" in line for line in logs))

    async def test_llamacpp_without_hf_metadata_fails(self) -> None:
        from tui.backends.llamacpp import backend_runtime as lrt
        from tui.common import profile_store as ps

        stored = ps.StoredProfile(
            name="p", backend="llamacpp", container_name="p", port=8080,
            config_name="p",
        )
        profile = lrt.Profile(
            name="p", container_name="p", port=8080, config_name="p"
        )

        def validate_common_env(path, *, require_downloader_image=False):
            self.assertEqual(path, lrt.COMMON_ENV)
            self.assertTrue(require_downloader_image)
            return True, []

        with patch.object(lrt, "validate_common_env", validate_common_env), \
             patch.object(lrt.profile_store, "load_profile", lambda *a, **k: stored), \
             patch.object(lrt, "load_profile", lambda _n: profile), \
             patch.object(lrt, "_ensure_profile_config", lambda *_a: (True, [])):
            logs, rc = await _drain(lrt.stream_container_prepare("p"))
        self.assertEqual(rc, 1)
        self.assertTrue(any("hf_repo" in line for line in logs))


class LlamacppPrepareHappyPathTests(unittest.IsolatedAsyncioTestCase):
    """A cached GGUF still renders + verifies the image, but downloads nothing
    and never starts the server."""

    async def test_cached_model_prepares_without_starting(self) -> None:
        from tui.backends.llamacpp import backend_runtime as lrt
        from tui.common import profile_store as ps

        stored = ps.StoredProfile(
            name="p", backend="llamacpp", container_name="p", port=8080,
            config_name="p", hf_repo="o/r", hf_file="m.gguf",
        )
        profile = lrt.Profile(
            name="p", container_name="p", port=8080, config_name="p",
            hf_repo="o/r", hf_file="m.gguf",
        )

        async def render_ok(_profile, _image):
            return (0, "")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            snap = tmp / "hub" / "models--o--r" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            (snap / "m.gguf").write_text("gguf")

            async def image_present(_ref):
                return True

            async def no_docker(*args, **kwargs):
                raise AssertionError("no container may run for a cached model")

            def validate_common_env(path, *, require_downloader_image=False):
                self.assertEqual(path, lrt.COMMON_ENV)
                self.assertTrue(require_downloader_image)
                return True, []

            def render_env_for_profile(name, backend):
                self.assertEqual((name, backend), ("p", "llamacpp"))
                return Path("p.env")

            with patch.object(lrt, "validate_common_env", validate_common_env), \
                 patch.object(lrt.profile_store, "load_profile", lambda *a, **k: stored), \
                 patch.object(lrt.profile_store, "render_env_for_profile", render_env_for_profile), \
                 patch.object(lrt, "load_profile", lambda _n: profile), \
                 patch.object(lrt, "_ensure_profile_config", lambda *_a: (True, [])), \
                 patch.object(lrt, "_render_override", render_ok), \
                 patch.object(lrt.prepare, "image_present", image_present), \
                 patch.object(lrt.prepare, "hf_cache_path", lambda: str(tmp)), \
                 patch.object(lrt.prepare, "hf_token", lambda: ""), \
                 patch.object(lrt.prepare, "downloader_image", lambda: "vllm/vllm-openai:v0.27.1"), \
                 patch.object(prepare, "_run", no_docker):
                logs, rc = await _drain(lrt.stream_container_prepare("p"))

        self.assertEqual(rc, 0)
        self.assertTrue(any("이미 HF 캐시에" in line or "Already in the HF cache" in line
                            for line in logs))


class PrepareCliWiringTests(unittest.TestCase):
    def test_prepare_is_registered_top_level_and_under_container(self) -> None:
        from tui.cli import app
        from tui.cli import container as cli_container

        names = {c.name for c in app.registered_commands}
        self.assertIn("prepare", names)
        sub = {c.name or c.callback.__name__ for c in cli_container.app.registered_commands}
        self.assertIn("prepare", sub)


if __name__ == "__main__":
    unittest.main()


class UpdateCommandTests(unittest.TestCase):
    """`llmux update` answers on demand — no cache, no TTY requirement."""

    def _invoke(self, status, **kwargs):
        from typer.testing import CliRunner

        from tui.cli import app
        from tui.common import version_check as vc

        args = ["update"] + kwargs.pop("args", [])
        with patch.object(vc, "resolve_status", return_value=status) as resolve, \
             patch.object(vc, "update_blocked_reason", return_value=kwargs.pop("blocked", "")), \
             patch.object(vc, "apply_update", return_value=kwargs.pop("apply", (True, "Updated to v9.9.9."))) as apply_:
            result = CliRunner().invoke(app, args)
        return result, resolve, apply_

    def test_up_to_date_exits_zero_without_touching_the_checkout(self) -> None:
        from tui.common import version_check as vc

        result, resolve, apply_ = self._invoke(
            vc.UpdateStatus(vc.CURRENT, tag="v2.6.0", local_version="2.6.0")
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("up to date", result.output)
        apply_.assert_not_called()
        # The explicit command must never sit behind the failure back-off.
        self.assertEqual(resolve.call_args.kwargs, {"respect_cooldown": False})

    def test_check_only_reports_without_updating(self) -> None:
        from tui.common import version_check as vc

        result, _, apply_ = self._invoke(
            vc.UpdateStatus(vc.BEHIND, tag="v9.9.9", url="u", local_version="2.6.0"),
            args=["--check"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("v9.9.9", result.output)
        apply_.assert_not_called()

    def test_behind_with_yes_applies_the_update(self) -> None:
        from tui.common import version_check as vc

        result, _, apply_ = self._invoke(
            vc.UpdateStatus(vc.BEHIND, tag="v9.9.9", url="u"), args=["--yes"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        apply_.assert_called_once_with("v9.9.9")

    def test_dirty_checkout_refuses_and_exits_nonzero(self) -> None:
        from tui.common import version_check as vc

        result, _, apply_ = self._invoke(
            vc.UpdateStatus(vc.BEHIND, tag="v9.9.9", url="u"),
            args=["--yes"], blocked="checkout has uncommitted changes to tracked files",
        )
        self.assertEqual(result.exit_code, 1)
        apply_.assert_not_called()

    def test_failed_lookup_exits_nonzero(self) -> None:
        from tui.common import version_check as vc

        result, _, _ = self._invoke(
            vc.UpdateStatus(vc.UNKNOWN, detail="offline"), args=["--check"]
        )
        self.assertEqual(result.exit_code, 1)

    def test_json_reports_state(self) -> None:
        import json

        from tui.common import version_check as vc

        result, _, _ = self._invoke(
            vc.UpdateStatus(vc.BEHIND, tag="v9.9.9", url="u", local_version="2.6.0"),
            args=["--check", "--json"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data["state"], "behind")
        self.assertEqual(data["latest_tag"], "v9.9.9")
        self.assertEqual(data["local_version"], "2.6.0")
