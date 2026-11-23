from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tui.backends.llamacpp import backend_runtime


def _load_renderer():
    script = Path(__file__).resolve().parents[1] / "scripts" / "llamacpp" / "render-override.py"
    spec = importlib.util.spec_from_file_location("render_flags_renderer", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load_renderer()
SRC = {"hf_repo": "org/repo", "hf_file": "m.gguf"}


class FalseBoolRenderTests(unittest.TestCase):
    def test_false_renders_negated_flag_when_image_has_it(self) -> None:
        command = R.render_command({"jinja": False}, known_flags={"jinja", "no-jinja"}, **SRC)
        self.assertIn("--no-jinja", command)
        self.assertNotIn("--jinja", command)

    def test_false_without_negated_flag_is_a_hard_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "--no-mlock"):
            R.render_command({"mlock": False}, known_flags={"mlock"}, **SRC)

    def test_false_without_flag_list_is_a_hard_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "flag list"):
            R.render_command({"jinja": False}, **SRC)

    def test_flash_attn_false_still_renders_off_without_flag_list(self) -> None:
        command = R.render_command({"flash-attn": False}, **SRC)
        self.assertEqual(command[command.index("--flash-attn") + 1], "off")

    def test_main_reads_flag_list_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cfg.yaml").write_text("jinja: false\n")
            stored = SimpleNamespace(config_name="cfg", **SRC)
            env = {R.FLAGS_ENV: json.dumps(["jinja", "no-jinja"])}
            with (
                patch.object(R, "CONFIG_DIR", root),
                patch.object(R, "RUNTIME_DIR", root / "runtime"),
                patch.object(R.profile_store, "load_profile", return_value=stored),
                patch.object(R.profile_store, "render_env", return_value=None),
                patch.object(sys, "argv", ["render-override.py", "p"]),
                patch.dict(os.environ, env),
            ):
                rc = R.main()
            self.assertEqual(rc, 0)
            self.assertIn("--no-jinja", (root / "runtime" / "override-p.yaml").read_text())


class ModelSourceBlockTests(unittest.TestCase):
    def test_model_source_config_keys_are_rejected(self) -> None:
        for key in ("model", "m", "hf-repo", "hf", "hf-file", "hff", "models-dir", "model-url"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                R.render_command({key: "x"}, **SRC)

    def test_model_source_extra_args_are_rejected(self) -> None:
        for args in (["-m", "x.gguf"], ["--hf-repo=org/x"], ["-hff", "x"], ["--models-dir", "/m"]):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "extra-args"):
                R.render_command({"extra-args": args}, **SRC)

    def test_ui_keys_are_managed(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            command = R.render_command({"ui": True}, **SRC)
        self.assertNotIn("--ui", command)
        self.assertIn("managed by llmux", stderr.getvalue())
        with self.assertRaisesRegex(ValueError, "--no-ui"):
            R.render_command({"extra-args": ["--no-ui"]}, **SRC)


class RenderFlagFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_flag_list_is_fetched_only_when_config_has_false(self) -> None:
        extract = AsyncMock(return_value={"jinja", "no-jinja"})
        with patch.object(backend_runtime, "extract_llama_server_flags", extract):
            env = await backend_runtime._render_flags_env({"jinja": True}, "img:1")
            self.assertEqual(env, {})
            extract.assert_not_awaited()
            env = await backend_runtime._render_flags_env({"jinja": False}, "img:1")
        extract.assert_awaited_once_with("img:1")
        self.assertEqual(json.loads(env[R.FLAGS_ENV]), ["jinja", "no-jinja"])

    async def test_on_off_flags_need_no_flag_list(self) -> None:
        extract = AsyncMock(side_effect=AssertionError("flash-attn renders on/off without the image"))
        with patch.object(backend_runtime, "extract_llama_server_flags", extract):
            self.assertEqual(await backend_runtime._render_flags_env({"flash-attn": False}, "img:1"), {})


class ModelPresetTests(unittest.TestCase):
    def test_presets_that_pick_a_model_or_port_are_blocked(self) -> None:
        for key in ("embd-gemma-default", "fim-qwen-7b-spec", "gpt-oss-20b-default"):
            with self.assertRaisesRegex(ValueError, key):
                R.render_command({key: True}, **SRC)
            with self.assertRaisesRegex(ValueError, key):
                R.render_command({"extra-args": [f"--{key}"]}, **SRC)

    def test_spec_default_only_tunes_decoding(self) -> None:
        self.assertIn("--spec-default", R.render_command({"spec-default": True}, **SRC))


if __name__ == "__main__":
    unittest.main()
