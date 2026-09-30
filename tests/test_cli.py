"""Smoke tests for the typer CLI surface.

CLI tests run via subprocess so each test gets a fresh interpreter with its
own LLMUX_ROOT — this avoids the profile_store PROJECT_ROOT being captured at
first import (which would otherwise pin to the real repo when other test
modules load first).
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_temp_project() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="llmux-cli-test-"))
    (tmp / "compose").mkdir()
    (tmp / "profiles.example.yaml").write_text("version: 1\n")
    (tmp / "config" / "vllm").mkdir(parents=True)
    (tmp / "config" / "llamacpp").mkdir(parents=True)
    (tmp / ".runtime").mkdir()
    (tmp / "profiles.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "defaults": {
                    "vllm": {
                        "port": 8000, "gpu_id": "0",
                        "tensor_parallel_size": 1, "enable_lora": False,
                    },
                    "llamacpp": {"port": 8080, "gpu_id": "0"},
                },
                "profiles": [
                    {"name": "alpha", "backend": "vllm", "port": 8001,
                     "gpu_id": "1", "model_id": "Qwen/Qwen3-0.6B"},
                    {"name": "bravo", "backend": "vllm", "port": 8002, "gpu_id": "0"},
                    {"name": "lcpp", "backend": "llamacpp", "port": 8080, "gpu_id": "0"},
                ],
            },
            sort_keys=False,
        )
    )
    (tmp / "config" / "vllm" / "alpha.yaml").write_text(
        yaml.safe_dump(
            {"model": "Qwen/Qwen3-0.6B", "gpu-memory-utilization": "0.85", "max-model-len": 4096},
            sort_keys=False,
        )
    )
    return tmp


def _run_cli(tmp: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "LLMUX_ROOT": str(tmp), "NO_COLOR": "1"}
    return subprocess.run(
        [sys.executable, "-m", "tui", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _run_cli_with_env(
    tmp: Path, args: list[str], extra_env: dict[str, str]
) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        **extra_env,
        "LLMUX_ROOT": str(tmp),
        "NO_COLOR": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "tui", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _run_cli_stubbed_hf(
    tmp: Path, gguf_files: list[str], *args: str
) -> subprocess.CompletedProcess:
    """Run the CLI with `list_hf_repo_files` stubbed to return a synthetic
    GGUF file listing. Used by the llama.cpp quick-setup regression test so
    it doesn't depend on huggingface.co reachability (`NameError: run_async`
    would otherwise be masked by network flake).
    """
    env = {**os.environ, "LLMUX_ROOT": str(tmp), "NO_COLOR": "1"}
    # Build the in-process stub: monkey-patch the backend symbol *before*
    # tui.cli is imported, so _quick_setup_llamacpp's local
    # `from ...backend import list_hf_repo_files` picks up the stub.
    stub_src = (
        "import sys\n"
        "import tui.backends.llamacpp.backend as _b\n"
        f"_FAKE_FILES = {[{'type': 'file', 'path': p} for p in gguf_files]!r}\n"
        "async def _stub(*_a, **_k): return _FAKE_FILES\n"
        "_b.list_hf_repo_files = _stub\n"
        "from tui.cli import main\n"
        f"sys.argv = ['llmux', *{list(args)!r}]\n"
        "main()\n"
    )
    return subprocess.run(
        [sys.executable, "-c", stub_src],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _run_cli_stubbed_recipe(tmp: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "LLMUX_ROOT": str(tmp), "NO_COLOR": "1"}
    stub_src = (
        "import sys\n"
        "import tui.common.recipes as _r\n"
        "async def _stub(*_a, **_k):\n"
        "    return _r.Recipe(\n"
        "        model_id='Qwen/Base',\n"
        "        base_args=['--max-model-len', '8192'],\n"
        "        variants=[_r.RecipeVariant(name='fp8', extra_args=['--quantization', 'fp8'])],\n"
        "        features=[_r.RecipeFeature(name='reasoning', args=['--reasoning-parser', 'qwen3'])],\n"
        "    )\n"
        "_r.fetch_recipe = _stub\n"
        "from tui.cli import main\n"
        f"sys.argv = ['llmux', *{list(args)!r}]\n"
        "main()\n"
    )
    return subprocess.run(
        [sys.executable, "-c", stub_src],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


class CliSmokeTests(unittest.TestCase):
    def test_config_list_rejects_empty_non_mapping_yaml(self):
        tmp = _make_temp_project()
        try:
            (tmp / "config" / "vllm" / "bad.yaml").write_text("[]\n")
            result = _run_cli(tmp, "config", "list", "--json")
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("not a mapping", result.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    """Side-effect-free dispatch + read-only command coverage."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = _make_temp_project()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


    def test_help_lists_top_commands(self):
        r = _run_cli(self.tmp, "--help")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        for cmd in ["tui", "up", "prepare", "down", "logs", "ps", "render-env",
                    "update", "container", "profile", "config", "image",
                    "system", "gpu", "env-check"]:
            self.assertIn(cmd, r.stdout)

    def test_subcommand_help(self):
        for sub in ["container", "profile", "config", "image", "system"]:
            r = _run_cli(self.tmp, sub, "--help")
            self.assertEqual(r.returncode, 0, f"{sub}: {r.stderr or r.stdout}")

    def test_config_flags_discovers_both_backends_from_image_cache(self):
        cases = (
            ("vllm", "custom/vllm:test", ".vllm-params-", ["max-model-len", "quantization"]),
            ("llamacpp", "custom/llama:test", ".llamacpp-params-", ["ctx-size", "flash-attn"]),
        )
        for backend, image, prefix, flags in cases:
            with self.subTest(backend=backend):
                identity = f"sha256:{backend}"
                version = "full-help-v2:" if backend == "vllm" else ""
                digest = hashlib.sha256(f"{version}{image}@{identity}".encode()).hexdigest()[:16]
                cache = self.tmp / "config" / backend / f"{prefix}{digest}.json"
                cache.write_text(json.dumps(flags))
                try:
                    with tempfile.TemporaryDirectory() as bindir:
                        docker = Path(bindir) / "docker"
                        docker.write_text(
                            "#!/usr/bin/env python3\n"
                            f"print({identity!r})\n"
                        )
                        docker.chmod(0o755)
                        result = _run_cli_with_env(
                            self.tmp,
                            [
                                "config",
                                "flags",
                                "--backend",
                                backend,
                                "--image",
                                image,
                                "--json",
                            ],
                            {"PATH": f"{bindir}:{os.environ['PATH']}"},
                        )
                    self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
                    data = json.loads(result.stdout)
                    self.assertEqual(data["backend"], backend)
                    self.assertEqual(data["image"], image)
                    self.assertEqual(data["flags"], sorted(flags))
                finally:
                    cache.unlink(missing_ok=True)

    def test_system_disk_supports_vllm_hf_cache(self):
        cache = self.tmp / "hf-cache"
        cache.mkdir()
        env_common = self.tmp / ".env.common"
        env_common.write_text(f"HF_CACHE_PATH={cache}\n")
        try:
            result = _run_cli(
                self.tmp,
                "system",
                "disk",
                "--backend",
                "vllm",
                "--json",
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            data = json.loads(result.stdout)
            self.assertEqual(data["backend"], "vllm")
            self.assertEqual(data["hf_cache_path"], str(cache))
            self.assertTrue(data["hf_cache_exists"])
        finally:
            env_common.unlink(missing_ok=True)

    def test_system_disk_failure_json_identifies_backend(self):
        cache = self.tmp / "hf-cache-failure"
        cache.mkdir()
        env_common = self.tmp / ".env.common"
        env_common.write_text(f"HF_CACHE_PATH={cache}\n")
        try:
            with tempfile.TemporaryDirectory() as bindir:
                df = Path(bindir) / "df"
                df.write_text("#!/bin/sh\necho 'df failed' >&2\nexit 1\n")
                df.chmod(0o755)
                for backend in ("vllm", "llamacpp"):
                    with self.subTest(backend=backend):
                        result = _run_cli_with_env(
                            self.tmp,
                            ["system", "disk", "--backend", backend, "--json"],
                            {"PATH": f"{bindir}:{os.environ['PATH']}"},
                        )
                        self.assertEqual(result.returncode, 1)
                        self.assertEqual(json.loads(result.stdout)["backend"], backend)
        finally:
            env_common.unlink(missing_ok=True)

    def test_image_list_backend_selects_llamacpp_repo(self):
        with tempfile.TemporaryDirectory() as bindir:
            docker = Path(bindir) / "docker"
            docker.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "repo = sys.argv[2]\n"
                "print(f'{repo}\\tserver-cuda\\t1GB\\t1 day ago')\n"
            )
            docker.chmod(0o755)
            result = _run_cli_with_env(
                self.tmp,
                ["image", "list", "--backend", "llamacpp", "--json"],
                {"PATH": f"{bindir}:{os.environ['PATH']}"},
            )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        rows = json.loads(result.stdout)
        self.assertEqual(rows[0]["repository"], "ghcr.io/ggml-org/llama.cpp")
        self.assertEqual(rows[0]["tag"], "server-cuda")

    def test_llamacpp_up_rejects_ambiguous_custom_image_reference(self):
        for image in ("ghcr.io/example/llama:latest", "ghcr.io/example/llama"):
            with self.subTest(image=image):
                result = _run_cli(self.tmp, "up", "lcpp", "--tag", image)
                self.assertEqual(result.returncode, 2, result.stderr or result.stdout)
                self.assertIn("image", (result.stderr + result.stdout).lower())

    def test_vllm_up_rejects_ambiguous_custom_image_reference(self):
        for image in ("custom/vllm:latest", "custom/vllm", "bad tag"):
            with self.subTest(image=image):
                result = _run_cli(self.tmp, "up", "alpha", "--tag", image)
                self.assertEqual(result.returncode, 2, result.stderr or result.stdout)
                self.assertIn("image", (result.stderr + result.stdout).lower())


    def test_profile_list_table(self):
        r = _run_cli(self.tmp, "profile", "list")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        for n in ("alpha", "bravo", "lcpp"):
            self.assertIn(n, r.stdout)

    def test_profile_list_json(self):
        r = _run_cli(self.tmp, "profile", "list", "--json")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        rows = json.loads(r.stdout)
        self.assertEqual(sorted(p["name"] for p in rows), ["alpha", "bravo", "lcpp"])

    def test_profile_list_filtered_by_backend(self):
        r = _run_cli(self.tmp, "profile", "list", "--backend", "llamacpp", "--json")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        rows = json.loads(r.stdout)
        self.assertEqual([p["name"] for p in rows], ["lcpp"])

    def test_profile_show_autodetects_backend(self):
        r = _run_cli(self.tmp, "profile", "show", "alpha", "--json")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        data = json.loads(r.stdout)
        self.assertEqual(data["backend"], "vllm")
        self.assertEqual(data["model_id"], "Qwen/Qwen3-0.6B")

    def test_profile_show_unknown_returns_error(self):
        r = _run_cli(self.tmp, "profile", "show", "no-such-profile")
        self.assertNotEqual(r.returncode, 0)
        # typer prints param errors to stderr by default
        combined = (r.stdout or "") + (r.stderr or "")
        self.assertIn("not found", combined)

    # --- bad env value: isolated to the one profile, not the whole read path -

    def _inject_bad_env_value(self, tmp: Path) -> None:
        """Put a value the .env renderer refuses into profiles.yaml directly —
        the shape a hand-edited or pre-validation file would have."""
        data = yaml.safe_load((tmp / "profiles.yaml").read_text())
        for p in data["profiles"]:
            if p["name"] == "bravo":
                p["env_vars"] = {"BAD": "it's"}
        (tmp / "profiles.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    def test_bad_env_value_does_not_break_read_paths(self):
        tmp = _make_temp_project()
        try:
            self._inject_bad_env_value(tmp)

            # ps must still work and still list *every* profile — the bad value
            # used to take down the whole scan with a traceback.
            r = _run_cli(tmp, "ps", "--json")
            self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
            names = {row["profile"] for row in json.loads(r.stdout)}
            self.assertIn("alpha", names)
            self.assertIn("bravo", names)
            self.assertIn("lcpp", names)

            # profile list too.
            pl = _run_cli(tmp, "profile", "list", "--json")
            self.assertEqual(pl.returncode, 0, pl.stderr or pl.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_non_integer_port_fails_the_whole_list_cleanly(self):
        tmp = _make_temp_project()
        try:
            data = yaml.safe_load((tmp / "profiles.yaml").read_text())
            for p in data["profiles"]:
                if p["name"] == "bravo":
                    p["port"] = "not-a-number"
            (tmp / "profiles.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

            r = _run_cli(tmp, "profile", "list", "--json")
            self.assertNotEqual(r.returncode, 0)
            self.assertNotIn("Traceback", r.stderr)
            self.assertIn("invalid port", r.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_malformed_entries_fail_cleanly(self):
        tmp = _make_temp_project()
        try:
            data = yaml.safe_load((tmp / "profiles.yaml").read_text())
            data["profiles"].append({"backend": "vllm", "port": 8005})  # no name
            data["profiles"].append("i am not a mapping")               # non-dict
            (tmp / "profiles.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

            r = _run_cli(tmp, "profile", "list", "--json")
            self.assertNotEqual(r.returncode, 0)
            self.assertNotIn("Traceback", r.stderr)
            self.assertIn("name", r.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_render_env_reports_the_offending_profile_cleanly(self):
        tmp = _make_temp_project()
        try:
            self._inject_bad_env_value(tmp)

            r = _run_cli(tmp, "render-env", "bravo")
            self.assertNotEqual(r.returncode, 0)
            out = r.stdout + r.stderr
            self.assertNotIn("Traceback", out)
            self.assertIn("single quote", out)

            # A healthy profile still renders fine.
            ok = _run_cli(tmp, "render-env", "alpha")
            self.assertEqual(ok.returncode, 0, ok.stderr or ok.stdout)

            # --unset is the documented recovery path.
            fix = _run_cli(tmp, "profile", "edit", "bravo", "--unset", "BAD")
            self.assertEqual(fix.returncode, 0, fix.stderr or fix.stdout)
            after = _run_cli(tmp, "render-env", "bravo")
            self.assertEqual(after.returncode, 0, after.stderr or after.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


    def test_config_set_gpu_mem_is_range_checked_on_vllm(self):
        bad = _run_cli(
            self.tmp, "config", "new", "gmset", "--backend", "vllm",
            "--model", "test/model", "--set", "gpu-memory-utilization=1.5",
        )
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("gpu-memory-utilization", bad.stderr + bad.stdout)

        # llama.cpp --set is the raw llama-server flag namespace — same spelling
        # there is just a user flag and must not be policed by the vLLM rule.
        lc = _run_cli(
            self.tmp, "config", "new", "gmsetlc", "--backend", "llamacpp",
            "--set", "gpu-memory-utilization=1.5",
        )
        self.assertEqual(lc.returncode, 0, lc.stderr or lc.stdout)
        _run_cli(self.tmp, "config", "delete", "gmsetlc", "-y")

    def test_profile_edit_rejects_unsanitized_dev_image_tag(self):
        r = _run_cli(
            self.tmp, "profile", "edit", "alpha", "--image-tag", "vllm-dev:feat/foo",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("vllm-dev:feat-foo", r.stderr + r.stdout)

        # A sanitized dev tag and a normal ref are both accepted.
        ok = _run_cli(
            self.tmp, "profile", "edit", "alpha", "--image-tag", "vllm-dev:feat-foo",
        )
        self.assertEqual(ok.returncode, 0, ok.stderr or ok.stdout)
        # Clear the pin so the shared alpha fixture is left as we found it.
        _run_cli(self.tmp, "profile", "edit", "alpha", "--image-tag", "")


    def test_config_disable_enable_round_trip(self):
        _run_cli(
            self.tmp, "config", "new", "dp1", "--backend", "vllm",
            "--model", "m/x", "--set", "max-model-len=4096",
            "--set", "enforce-eager=",
        )
        try:
            # disable → moves to the disabled section (comment marker on disk).
            r = _run_cli(self.tmp, "config", "edit", "dp1", "--disable", "enforce-eager")
            self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

            text = (self.tmp / "config" / "vllm" / "dp1.yaml").read_text()
            self.assertIn("# llmux:disabled enforce-eager", text)
            # Server view (plain YAML) no longer contains it.
            self.assertNotIn("enforce-eager", yaml.safe_load(text))

            show = json.loads(
                _run_cli(self.tmp, "config", "show", "dp1", "--json").stdout
            )
            self.assertIn("enforce-eager", show["disabled"])
            self.assertNotIn("enforce-eager", show["params"])

            # enable → back to active.
            _run_cli(self.tmp, "config", "edit", "dp1", "--enable", "enforce-eager")
            show = json.loads(
                _run_cli(self.tmp, "config", "show", "dp1", "--json").stdout
            )
            self.assertIn("enforce-eager", show["params"])
            self.assertEqual(show["disabled"], {})
        finally:
            _run_cli(self.tmp, "config", "delete", "dp1", "-y")

    def test_config_set_reactivates_disabled_key(self):
        _run_cli(
            self.tmp, "config", "new", "dp2", "--backend", "vllm",
            "--model", "m/x", "--set", "max-model-len=4096",
        )
        try:
            _run_cli(self.tmp, "config", "edit", "dp2", "--disable", "max-model-len")
            # --set on a disabled key re-activates it with the new value.
            _run_cli(self.tmp, "config", "edit", "dp2", "--set", "max-model-len=8192")
            show = json.loads(
                _run_cli(self.tmp, "config", "show", "dp2", "--json").stdout
            )
            self.assertEqual(show["params"]["max-model-len"], 8192)
            self.assertEqual(show["disabled"], {})
        finally:
            _run_cli(self.tmp, "config", "delete", "dp2", "-y")

    def test_config_disable_rejects_unknown_key(self):
        _run_cli(self.tmp, "config", "new", "dp3", "--backend", "vllm", "--model", "m/x")
        try:
            r = _run_cli(self.tmp, "config", "edit", "dp3", "--disable", "nope")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("cannot disable", r.stderr + r.stdout)
        finally:
            _run_cli(self.tmp, "config", "delete", "dp3", "-y")

    def test_config_cannot_toggle_vllm_core_fields(self):
        _run_cli(self.tmp, "config", "new", "dpc", "--backend", "vllm", "--model", "m/x")
        try:
            for flag in ("--disable", "--enable", "--unset"):
                for key in ("model", "gpu-memory-utilization"):
                    r = _run_cli(self.tmp, "config", "edit", "dpc", flag, key)
                    self.assertNotEqual(r.returncode, 0, (flag, key))
                    self.assertIn("core vLLM field", r.stderr + r.stdout)
            # The serializer re-emits both with a default, so an accepted
            # --unset would have looked like success while nothing changed.
            show = json.loads(
                _run_cli(self.tmp, "config", "show", "dpc", "--json").stdout
            )
            self.assertEqual(show["params"]["model"], "m/x")
        finally:
            _run_cli(self.tmp, "config", "delete", "dpc", "-y")

    def test_config_enable_of_bad_gpu_mem_is_rejected(self):
        # Hand-craft a config with a disabled, out-of-range gpu-memory-utilization
        # marker — enabling it must not slip past the range check.
        (self.tmp / "config" / "vllm" / "dpe.yaml").write_text(
            "model: m/x\ngpu-memory-utilization: '0.9'\n"
            "# llmux:disabled gpu-memory-utilization: 9.0\n"
        )
        try:
            r = _run_cli(self.tmp, "config", "edit", "dpe", "--enable", "gpu-memory-utilization")
            self.assertNotEqual(r.returncode, 0)
        finally:
            (self.tmp / "config" / "vllm" / "dpe.yaml").unlink(missing_ok=True)

    def test_config_clone_carries_disabled(self):
        _run_cli(
            self.tmp, "config", "new", "dp4", "--backend", "vllm",
            "--model", "m/x", "--set", "quantization=fp8",
        )
        _run_cli(self.tmp, "config", "edit", "dp4", "--disable", "quantization")
        try:
            _run_cli(self.tmp, "config", "clone", "dp4", "dp4copy", "--backend", "vllm")
            show = json.loads(
                _run_cli(self.tmp, "config", "show", "dp4copy", "--json").stdout
            )
            self.assertIn("quantization", show["disabled"])
        finally:
            _run_cli(self.tmp, "config", "delete", "dp4", "-y")
            _run_cli(self.tmp, "config", "delete", "dp4copy", "-y")


    def test_env_check_passes_on_onboarding_default_env_common(self):
        # The onboarding template leaves HF_TOKEN empty (public repos need none)
        # and has no VLLM_VERSION at all (the runtime injects it at start time).
        # env-check used to require all three and exit 1 on a perfectly valid
        # default setup, while the real start-path gate (validate_common_env)
        # passed the same file.
        env_common = self.tmp / ".env.common"
        env_common.write_text(
            "HF_TOKEN=\n"
            f"HF_CACHE_PATH={self.tmp}/hfcache\n"
            "PREPARE_DOWNLOADER_IMAGE=vllm/vllm-openai:v0.27.1\n"
            "TZ=Asia/Seoul\n"
        )
        env_common.chmod(0o600)
        try:
            r = _run_cli(self.tmp, "env-check")
            self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
            self.assertIn("OK", r.stdout)
            self.assertIn("optional", r.stdout)  # HF_TOKEN annotated, not an error

            j = _run_cli(self.tmp, "env-check", "--json")
            self.assertEqual(j.returncode, 0, j.stderr or j.stdout)
            data = json.loads(j.stdout)
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["issues"], [])
        finally:
            env_common.unlink(missing_ok=True)

    def test_env_check_still_fails_on_relative_hf_cache_path(self):
        env_common = self.tmp / ".env.common"
        env_common.write_text("HF_TOKEN=\nHF_CACHE_PATH=relative/path\n")
        env_common.chmod(0o600)
        try:
            r = _run_cli(self.tmp, "env-check")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("absolute", r.stdout + r.stderr)
        finally:
            env_common.unlink(missing_ok=True)

    def test_env_check_accepts_home_expansion_from_shipped_template(self):
        env_common = self.tmp / ".env.common"
        env_common.write_text(
            "HF_TOKEN=\n"
            "HF_CACHE_PATH=$HOME/.cache/huggingface\n"
            "PREPARE_DOWNLOADER_IMAGE=vllm/vllm-openai:v0.27.1\n"
        )
        env_common.chmod(0o600)
        try:
            result = _run_cli(self.tmp, "env-check")
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        finally:
            env_common.unlink(missing_ok=True)

    def test_env_check_rejects_invalid_prepare_worker_count(self):
        env_common = self.tmp / ".env.common"
        env_common.write_text(
            f"HF_CACHE_PATH={self.tmp}/hfcache\nPREPARE_MAX_WORKERS=many\n"
        )
        env_common.chmod(0o600)
        try:
            result = _run_cli(self.tmp, "env-check")
            self.assertEqual(result.returncode, 1)
            self.assertIn("PREPARE_MAX_WORKERS", result.stdout + result.stderr)
        finally:
            env_common.unlink(missing_ok=True)

    def test_env_check_rejects_ambiguous_default_image_references(self):
        for key in ("VLLM_IMAGE", "LLAMACPP_IMAGE"):
            with self.subTest(key=key):
                env_common = self.tmp / ".env.common"
                env_common.write_text(
                    f"HF_CACHE_PATH={self.tmp}/hfcache\n"
                    "PREPARE_DOWNLOADER_IMAGE=vllm/vllm-openai:v0.27.1\n"
                    f"{key}=registry.example/image:latest\n"
                )
                env_common.chmod(0o600)
                try:
                    result = _run_cli(self.tmp, "env-check")
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(key, result.stdout + result.stderr)
                finally:
                    env_common.unlink(missing_ok=True)

    def test_env_check_json_redacts_hf_token(self):
        env_common = self.tmp / ".env.common"
        secret = "hf_secret_value"
        env_common.write_text(
            f"HF_TOKEN={secret}\n"
            f"HF_CACHE_PATH={self.tmp}/hfcache\n"
            "PREPARE_DOWNLOADER_IMAGE=vllm/vllm-openai:v0.27.1\n"
        )
        env_common.chmod(0o600)
        try:
            r = _run_cli(self.tmp, "env-check", "--json")
            self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
            self.assertNotIn(secret, r.stdout + r.stderr)
            data = json.loads(r.stdout)
            self.assertEqual(
                data["HF_TOKEN"], {"set": True, "length": len(secret)}
            )
        finally:
            env_common.unlink(missing_ok=True)


    def test_unknown_backend_is_a_clean_usage_error(self):
        # `ps -b foo` silently fell through to llamacpp and printed rows labelled
        # with the bogus name; the other two died with a raw ValueError.
        for args in (
            ("ps", "--backend", "foo"),
            ("profile", "list", "--backend", "foo"),
            ("render-env", "--backend", "foo"),
        ):
            with self.subTest(args=args):
                r = _run_cli(self.tmp, *args)
                self.assertNotEqual(r.returncode, 0)
                out = r.stderr + r.stdout
                self.assertIn("unknown backend", out)
                self.assertNotIn("Traceback", out)


    def test_config_list_includes_seeded(self):
        r = _run_cli(self.tmp, "config", "list", "--json")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        rows = json.loads(r.stdout)
        names = [row["name"] for row in rows if row["backend"] == "vllm"]
        self.assertIn("alpha", names)

    def test_config_show_yaml(self):
        r = _run_cli(self.tmp, "config", "show", "alpha")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        self.assertIn("model: Qwen/Qwen3-0.6B", r.stdout)
        self.assertIn("max-model-len: 4096", r.stdout)



    def test_profile_new_rejects_privileged_port(self):
        r = _run_cli(
            self.tmp, "profile", "new", "p80", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--port", "80",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("1024", r.stderr + r.stdout)

    def test_profile_new_accepts_multi_digit_gpu_id(self):
        # `0,10` must be valid — the old `[0-9](,[0-9])*` rule capped indices
        # at one digit, making hosts with 10+ GPUs unaddressable.
        r = _run_cli(
            self.tmp, "profile", "new", "gpu10", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--port", "8099", "--gpu-id", "0,10",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        show = _run_cli(self.tmp, "profile", "show", "gpu10", "--json")
        self.assertEqual(json.loads(show.stdout)["gpu_id"], "0,10")

        _run_cli(self.tmp, "profile", "delete", "gpu10", "-y")

    def test_profile_new_rejects_malformed_gpu_id(self):
        r = _run_cli(
            self.tmp, "profile", "new", "gpubad", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--port", "8098", "--gpu-id", "0,,",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("GPU id", r.stderr + r.stdout)

    def test_profile_new_rejects_name_used_by_other_backend(self):
        # Global name uniqueness (finding #1): `lcpp` already exists as a
        # llamacpp profile, so a vLLM profile of the same name must be rejected —
        # container_name defaults to the profile name and would collide.
        r = _run_cli(
            self.tmp, "profile", "new", "lcpp", "--backend", "vllm", "--port", "8055",
        )
        self.assertNotEqual(r.returncode, 0)
        out = r.stderr + r.stdout
        self.assertIn("already exists", out)
        self.assertIn("llamacpp", out)

    def test_profile_edit_rejects_bad_port(self):
        created = _run_cli(
            self.tmp, "profile", "new", "editport", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--port", "8097",
        )
        self.assertEqual(created.returncode, 0, created.stderr or created.stdout)

        r = _run_cli(self.tmp, "profile", "edit", "editport", "--port", "80")
        self.assertNotEqual(r.returncode, 0)

        _run_cli(self.tmp, "profile", "delete", "editport", "-y")


    def test_profile_new_config_must_exist_unless_named_after_profile(self):
        # A typo'd --config used to be stored verbatim; start-up would then
        # auto-create an empty config under the wrong name.
        bad = _run_cli(
            self.tmp, "profile", "new", "cfgtypo", "--backend", "vllm",
            "--port", "8096", "--config", "alfa",   # real one is "alpha"
        )
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("config not found", bad.stderr + bad.stdout)

        # Exception: a config named after the profile is auto-created at start.
        ok = _run_cli(
            self.tmp, "profile", "new", "cfgself", "--backend", "vllm",
            "--port", "8095", "--config", "cfgself",
        )
        self.assertEqual(ok.returncode, 0, ok.stderr or ok.stdout)

        _run_cli(self.tmp, "profile", "delete", "cfgself", "-y")

    def test_profile_rejects_linking_the_tracked_example_config(self):
        # example.yaml is the tracked template — the TUI filters it out of every
        # config picker, but the CLI used to link it silently and write profile
        # params into a git-tracked file.
        explicit = _run_cli(
            self.tmp, "profile", "new", "exlink", "--backend", "vllm",
            "--port", "8090", "--config", "example",
        )
        self.assertNotEqual(explicit.returncode, 0)
        self.assertIn("example", explicit.stderr + explicit.stdout)

        # Also when the profile is merely *named* example and picks it up as the
        # default link.
        by_name = _run_cli(
            self.tmp, "profile", "new", "example", "--backend", "vllm",
            "--port", "8089",
        )
        self.assertNotEqual(by_name.returncode, 0)
        self.assertIn("example", by_name.stderr + by_name.stdout)

        # And on edit.
        created = _run_cli(
            self.tmp, "profile", "new", "exedit", "--backend", "vllm",
            "--port", "8088",
        )
        self.assertEqual(created.returncode, 0, created.stderr or created.stdout)
        r = _run_cli(self.tmp, "profile", "edit", "exedit", "--config", "example")
        self.assertNotEqual(r.returncode, 0)

        _run_cli(self.tmp, "profile", "delete", "exedit", "-y")

    def test_config_new_and_clone_refuse_to_create_example(self):
        # example.yaml is the tracked template — recreating/overwriting it has no
        # legitimate use, so even --overwrite is refused.
        r = _run_cli(
            self.tmp, "config", "new", "example", "--backend", "vllm",
            "--model", "test/model", "--overwrite",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("example", r.stderr + r.stdout)

        c = _run_cli(
            self.tmp, "config", "clone", "alpha", "example",
            "--backend", "vllm", "--overwrite",
        )
        self.assertNotEqual(c.returncode, 0)
        self.assertIn("example", c.stderr + c.stdout)

    def test_config_edit_of_example_warns_but_proceeds(self):
        # edit stays available as an escape hatch, but must not act silently.
        (self.tmp / "config" / "vllm" / "example.yaml").write_text(
            yaml.safe_dump({"model": "tpl/model"}, sort_keys=False)
        )
        try:
            r = _run_cli(
                self.tmp, "config", "edit", "example", "--backend", "vllm",
                "--set", "max-model-len=2048",
            )
            self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
            self.assertIn("warning", r.stdout.lower())
            self.assertIn("example.yaml", r.stdout)
        finally:
            (self.tmp / "config" / "vllm" / "example.yaml").unlink(missing_ok=True)

    def test_profile_set_rejects_value_compose_cannot_parse(self):
        # shlex.quote renders `it's` as `'it'"'"'s'` — valid shell, but docker
        # compose's dotenv parser chokes on it and `up` fails opaquely.
        r = _run_cli(
            self.tmp, "profile", "new", "quoteval", "--backend", "vllm",
            "--port", "8087", "--set", "K=it's",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("single quote", r.stderr + r.stdout)

    def test_profile_set_still_accepts_spaces_and_commas(self):
        r = _run_cli(
            self.tmp, "profile", "new", "okval", "--backend", "vllm",
            "--port", "8086", "--set", "K=a b", "--set", "L=0,1",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        show = _run_cli(self.tmp, "profile", "show", "okval", "--json")
        self.assertEqual(json.loads(show.stdout)["env_vars"], {"K": "a b", "L": "0,1"})

        _run_cli(self.tmp, "profile", "delete", "okval", "-y")


    def test_config_new_rejects_non_numeric_gpu_mem(self):
        r = _run_cli(
            self.tmp, "config", "new", "gmbad1", "--backend", "vllm",
            "--model", "test/model", "--gpu-mem", "abc",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("gpu-memory-utilization", r.stderr + r.stdout)

    def test_config_new_rejects_out_of_range_gpu_mem(self):
        r = _run_cli(
            self.tmp, "config", "new", "gmbad2", "--backend", "vllm",
            "--model", "test/model", "--gpu-mem", "1.5",
        )
        self.assertNotEqual(r.returncode, 0)
        # Rich line-wraps the error box, so match a substring that can't straddle
        # a break.
        self.assertIn("gpu-memory-utilization", r.stderr + r.stdout)

    def test_config_new_accepts_valid_gpu_mem(self):
        r = _run_cli(
            self.tmp, "config", "new", "gmok", "--backend", "vllm",
            "--model", "test/model", "--gpu-mem", "0.85",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        show = _run_cli(self.tmp, "config", "show", "gmok", "--json")
        self.assertEqual(
            json.loads(show.stdout)["params"]["gpu-memory-utilization"], "0.85"
        )

        _run_cli(self.tmp, "config", "delete", "gmok", "-y")

    def test_config_edit_rejects_out_of_range_gpu_mem(self):
        r = _run_cli(self.tmp, "config", "edit", "alpha", "--gpu-mem", "1.5")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("gpu-memory-utilization", r.stderr + r.stdout)

    def test_quick_setup_rejects_out_of_range_gpu_mem(self):
        r = _run_cli(
            self.tmp, "profile", "quick-setup", "test/model", "--backend", "vllm",
            "--name", "qsgm", "--gpu-mem", "1.5",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("gpu-memory-utilization", r.stderr + r.stdout)


    def test_profile_edit_gpu_id_announces_derived_tensor_parallel(self):
        created = _run_cli(
            self.tmp, "profile", "new", "tp1", "--backend", "vllm",
            "--port", "8094", "--gpu-id", "0", "--model", "test/model",
        )
        self.assertEqual(created.returncode, 0, created.stderr or created.stdout)

        r = _run_cli(self.tmp, "profile", "edit", "tp1", "--gpu-id", "0,1")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        # The re-derive used to be silent.
        self.assertIn("tensor_parallel_size adjusted to 2", r.stdout)

        show = _run_cli(self.tmp, "profile", "show", "tp1", "--json")
        self.assertEqual(json.loads(show.stdout)["tensor_parallel_size"], 2)

        _run_cli(self.tmp, "profile", "delete", "tp1", "-y")

    def test_profile_edit_explicit_tensor_parallel_overrides_gpu_id(self):
        created = _run_cli(
            self.tmp, "profile", "new", "tp2", "--backend", "vllm",
            "--port", "8093", "--gpu-id", "0", "--model", "test/model",
        )
        self.assertEqual(created.returncode, 0, created.stderr or created.stdout)

        r = _run_cli(
            self.tmp, "profile", "edit", "tp2",
            "--gpu-id", "0,1", "--tensor-parallel", "1",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        self.assertNotIn("adjusted", r.stdout)

        show = _run_cli(self.tmp, "profile", "show", "tp2", "--json")
        data = json.loads(show.stdout)
        self.assertEqual(data["gpu_id"], "0,1")
        self.assertEqual(data["tensor_parallel_size"], 1)

        _run_cli(self.tmp, "profile", "delete", "tp2", "-y")

    def test_profile_edit_gpu_id_stays_silent_on_llamacpp(self):
        # TP is vLLM-only: llama.cpp never persists the field and rejects
        # --tensor-parallel, so announcing an "adjustment" (and pointing at that
        # flag) was both false and self-contradictory.
        created = _run_cli(
            self.tmp, "profile", "new", "tplc", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--port", "8091", "--gpu-id", "0",
        )
        self.assertEqual(created.returncode, 0, created.stderr or created.stdout)

        r = _run_cli(self.tmp, "profile", "edit", "tplc", "--gpu-id", "0,1")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        self.assertNotIn("tensor_parallel_size", r.stdout)
        self.assertNotIn("--tensor-parallel", r.stdout)

        show = _run_cli(self.tmp, "profile", "show", "tplc", "--json")
        self.assertEqual(json.loads(show.stdout)["gpu_id"], "0,1")

        _run_cli(self.tmp, "profile", "delete", "tplc", "-y")

    def test_profile_new_rejects_tensor_parallel_on_llamacpp(self):
        r = _run_cli(
            self.tmp, "profile", "new", "tp3", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--port", "8092", "--tensor-parallel", "2",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--tensor-parallel", r.stderr + r.stdout)


    def test_profile_new_rejects_vllm_only_option_on_llamacpp(self):
        # These used to be accepted and silently dropped by _profile_to_entry.
        r = _run_cli(
            self.tmp, "profile", "new", "xbk1", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--model", "org/some-model",
        )
        self.assertNotEqual(r.returncode, 0)
        out = r.stderr + r.stdout
        self.assertIn("--model", out)
        self.assertIn("vLLM-only", out)

    def test_profile_new_rejects_llamacpp_only_option_on_vllm(self):
        r = _run_cli(
            self.tmp, "profile", "new", "xbk2", "--backend", "vllm",
            "--model", "org/m", "--hf-repo", "org/x",
        )
        self.assertNotEqual(r.returncode, 0)
        out = r.stderr + r.stdout
        self.assertIn("--hf-repo", out)
        self.assertIn("llama.cpp-only", out)

    def test_profile_edit_rejects_cross_backend_option(self):
        created = _run_cli(
            self.tmp, "profile", "new", "xbk3", "--backend", "llamacpp",
            "--hf-repo", "org/x",
        )
        self.assertEqual(created.returncode, 0, created.stderr or created.stdout)

        r = _run_cli(self.tmp, "profile", "edit", "xbk3", "--extra-pip", "foo")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("vLLM-only", r.stderr + r.stdout)

        _run_cli(self.tmp, "profile", "delete", "xbk3", "-y")

    def test_quick_setup_rejects_llamacpp_only_option_on_vllm(self):
        r = _run_cli(
            self.tmp, "profile", "quick-setup", "org/m", "--backend", "vllm",
            "--ctx-size", "4096",
        )
        self.assertNotEqual(r.returncode, 0)
        out = r.stderr + r.stdout
        self.assertIn("--ctx-size", out)
        self.assertIn("llama.cpp-only", out)

    def test_quick_setup_rejects_vllm_only_option_on_llamacpp(self):
        r = _run_cli(
            self.tmp, "profile", "quick-setup", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--hf-file", "m.gguf", "--gpu-mem", "0.5",
        )
        self.assertNotEqual(r.returncode, 0)
        out = r.stderr + r.stdout
        self.assertIn("--gpu-mem", out)
        self.assertIn("vLLM-only", out)

    def test_profile_new_rejects_invalid_container_name(self):
        r = _run_cli(
            self.tmp, "profile", "new", "ctr1", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--container", "Bad_Name",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--container", r.stderr + r.stdout)

    def test_profile_delete_refuses_running_container(self):
        bin_dir = self.tmp / "test-bin"
        bin_dir.mkdir(exist_ok=True)
        docker = bin_dir / "docker"
        docker.write_text("#!/bin/sh\nprintf 'alpha\\n'\n")
        docker.chmod(0o755)
        with patch.dict(
            os.environ, {"PATH": f"{bin_dir}:{os.environ['PATH']}"}
        ):
            r = _run_cli(self.tmp, "profile", "delete", "alpha", "--yes")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("running", r.stderr + r.stdout)
        show = _run_cli(self.tmp, "profile", "show", "alpha", "--json")
        self.assertEqual(show.returncode, 0, show.stderr or show.stdout)

    def test_profile_clone_replaces_custom_container_name(self):
        created = _run_cli(
            self.tmp,
            "profile",
            "new",
            "clone-source",
            "--backend",
            "vllm",
            "--container",
            "custom-container",
        )
        self.assertEqual(created.returncode, 0, created.stderr or created.stdout)
        cloned = _run_cli(
            self.tmp, "profile", "clone", "clone-source", "clone-target"
        )
        self.assertEqual(cloned.returncode, 0, cloned.stderr or cloned.stdout)
        shown = _run_cli(
            self.tmp, "profile", "show", "clone-target", "--json"
        )
        self.assertEqual(shown.returncode, 0, shown.stderr or shown.stdout)
        self.assertEqual(json.loads(shown.stdout)["container_name"], "clone-target")
        _run_cli(self.tmp, "profile", "delete", "clone-target", "--yes")
        _run_cli(self.tmp, "profile", "delete", "clone-source", "--yes")

    def test_force_does_not_classify_port_conflicts_as_bypassable(self):
        from tui.cli._runtime import partition_conflict_warnings

        hard, soft = partition_conflict_warnings(
            [
                "port conflict (llmux): occupied",
                "GPU conflict: GPU 0 overlap",
            ]
        )
        self.assertEqual(hard, ["port conflict (llmux): occupied"])
        self.assertEqual(soft, ["GPU conflict: GPU 0 overlap"])

    def test_config_new_rejects_uppercase_name(self):
        # Creation follows the TUI's lowercase rule...
        r = _run_cli(self.tmp, "config", "new", "BadName", "--backend", "vllm")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("lowercase", r.stderr + r.stdout)

        # ...but an existing odd-named file stays manageable (permissive rule
        # on the reference paths).
        (self.tmp / "config" / "vllm" / "Legacy.yaml").write_text("model: x\n")
        r = _run_cli(self.tmp, "config", "show", "Legacy", "--backend", "vllm")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        r = _run_cli(self.tmp, "config", "delete", "Legacy", "--backend", "vllm", "-y")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

    def test_quick_setup_rejects_missing_copy_from(self):
        # load_config() falls back to an empty Config for a missing file, so a
        # typo'd --copy-from used to silently produce a config with no params.
        r = _run_cli(
            self.tmp, "profile", "quick-setup", "org/m", "--backend", "vllm",
            "--copy-from", "does-not-exist",
        )
        self.assertNotEqual(r.returncode, 0)
        out = r.stderr + r.stdout
        self.assertIn("--copy-from", out)
        self.assertIn("not found", out)

    def test_config_new_rejects_vllm_only_options_on_llamacpp(self):
        r = _run_cli(
            self.tmp, "config", "new", "lcfg", "--backend", "llamacpp",
            "--model", "org/m",
        )
        self.assertNotEqual(r.returncode, 0)
        out = r.stderr + r.stdout
        self.assertIn("--model", out)
        self.assertIn("vLLM-only", out)
        self.assertIn("--set", out)  # points at the right alternative

    def test_config_edit_rejects_vllm_only_options_on_llamacpp(self):
        created = _run_cli(
            self.tmp, "config", "new", "lcfg2", "--backend", "llamacpp",
            "--set", "ctx-size=4096",
        )
        self.assertEqual(created.returncode, 0, created.stderr or created.stdout)

        r = _run_cli(
            self.tmp, "config", "edit", "lcfg2", "--backend", "llamacpp",
            "--gpu-mem", "0.8",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("vLLM-only", r.stderr + r.stdout)

        # vLLM configs still accept them.
        r = _run_cli(
            self.tmp, "config", "new", "vcfg", "--backend", "vllm",
            "--model", "org/m", "--gpu-mem", "0.8",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

    def test_up_rejects_tag_with_default_image(self):
        r = _run_cli(self.tmp, "up", "nonexistent", "--tag", "x", "--default-image")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("mutually exclusive", r.stderr + r.stdout)

    def test_port_zero_sentinel_honors_user_defaults_override(self):
        # profiles.yaml `defaults:` must beat the hardcoded DEFAULTS when the
        # `--port 0` sentinel is resolved — the loader already honors it.
        tmp = _make_temp_project()
        try:
            data = yaml.safe_load((tmp / "profiles.yaml").read_text())
            data["defaults"]["llamacpp"]["port"] = 9000
            (tmp / "profiles.yaml").write_text(yaml.safe_dump(data))

            r = _run_cli(
                tmp, "profile", "new", "userdef", "--backend", "llamacpp",
                "--hf-repo", "org/x", "--port", "0",
            )
            self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
            show = _run_cli(tmp, "profile", "show", "userdef", "--json")
            self.assertEqual(json.loads(show.stdout)["port"], 9000)

            # ...and the same for `edit --port 0`.
            r = _run_cli(
                tmp, "profile", "edit", "userdef", "--port", "8123",
            )
            self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
            r = _run_cli(tmp, "profile", "edit", "userdef", "--port", "0")
            self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
            show = _run_cli(tmp, "profile", "show", "userdef", "--json")
            self.assertEqual(json.loads(show.stdout)["port"], 9000)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_profile_edit_port_zero_resets_to_backend_default(self):
        # `--port 0` is the same "use the backend default" sentinel `new`
        # accepts; it must be resolved eagerly, never stored as a literal 0
        # (which would render `LLAMA_PORT=0` into the runtime .env).
        created = _run_cli(
            self.tmp, "profile", "new", "portzero", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--port", "8096",
        )
        self.assertEqual(created.returncode, 0, created.stderr or created.stdout)

        r = _run_cli(self.tmp, "profile", "edit", "portzero", "--port", "0")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        show = _run_cli(self.tmp, "profile", "show", "portzero", "--json")
        self.assertEqual(json.loads(show.stdout)["port"], 8080)

        _run_cli(self.tmp, "profile", "delete", "portzero", "-y")

    def test_profile_new_then_edit_then_delete(self):
        r = _run_cli(
            self.tmp, "profile", "new", "tmp1", "--backend", "vllm",
            "--port", "8003", "--gpu-id", "0", "--model", "test/model",
            "--set", "OMP_NUM_THREADS=4",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        r = _run_cli(self.tmp, "profile", "show", "tmp1", "--json")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        data = json.loads(r.stdout)
        self.assertEqual(data["port"], 8003)
        self.assertEqual(data["env_vars"], {"OMP_NUM_THREADS": "4"})

        r = _run_cli(
            self.tmp, "profile", "edit", "tmp1", "--port", "8004",
            "--set", "VLLM_USE_FOO=1", "--unset", "OMP_NUM_THREADS",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        r = _run_cli(self.tmp, "profile", "show", "tmp1", "--json")
        data = json.loads(r.stdout)
        self.assertEqual(data["port"], 8004)
        self.assertEqual(data["env_vars"], {"VLLM_USE_FOO": "1"})

        r = _run_cli(self.tmp, "profile", "delete", "tmp1", "-y")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        r = _run_cli(self.tmp, "profile", "show", "tmp1")
        self.assertNotEqual(r.returncode, 0)

    def test_config_new_set_overwrite_delete(self):
        r = _run_cli(
            self.tmp, "config", "new", "tmpcfg", "--model", "test/m",
            "--set", "max-model-len=8192", "--set", "enable-prefix-caching=",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        r = _run_cli(self.tmp, "config", "show", "tmpcfg", "--json")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        # show --json is now {"params": {...}, "disabled": {...}}.
        params = json.loads(r.stdout)["params"]
        self.assertEqual(params["model"], "test/m")
        self.assertEqual(params["max-model-len"], 8192)
        self.assertIs(params["enable-prefix-caching"], True)

        r = _run_cli(self.tmp, "config", "new", "tmpcfg", "--model", "x/y")
        self.assertNotEqual(r.returncode, 0)

        r = _run_cli(self.tmp, "config", "delete", "tmpcfg", "-y")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)


    def test_render_env_writes_runtime_file(self):
        r = _run_cli(self.tmp, "render-env", "alpha")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        out = self.tmp / ".runtime" / "vllm" / "alpha.env"
        self.assertTrue(out.exists(), out)
        content = out.read_text()
        self.assertIn("VLLM_PORT=8001", content)
        self.assertIn("GPU_ID=1", content)

    def test_render_env_all(self):
        r = _run_cli(self.tmp, "render-env")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        for n in ("alpha", "bravo"):
            self.assertTrue((self.tmp / ".runtime" / "vllm" / f"{n}.env").exists())


    def test_gpu_reports_missing_nvidia_smi(self):
        empty_path = self.tmp / "empty-path"
        empty_path.mkdir()
        r = _run_cli_with_env(
            self.tmp,
            ["gpu"],
            {"PATH": str(empty_path)},
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("nvidia-smi", (r.stdout + r.stderr).lower())

    def test_env_check_missing_env_common(self):
        r = _run_cli(self.tmp, "env-check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("MISSING", (r.stdout + r.stderr).upper())


    def test_config_show_rejects_path_traversal(self):
        # `..%2Fetc%2Fpasswd` style: dots-and-slashes name should be rejected
        # by _validate_name *before* any filesystem access happens.
        r = _run_cli(self.tmp, "config", "show", "../../etc/passwd")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("invalid", (r.stdout + r.stderr).lower())

    def test_config_delete_rejects_path_traversal(self):
        r = _run_cli(self.tmp, "config", "delete", "../alpha", "-y")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("invalid", (r.stdout + r.stderr).lower())

    def test_profile_set_rejects_invalid_env_key(self):
        # Lowercase-and-special chars violate POSIX env var naming rules.
        # Should fail at parse time with BadParameter, not deep in profile_store.
        r = _run_cli(
            self.tmp, "profile", "new", "tmpenv", "--backend", "vllm",
            "--port", "8005", "--gpu-id", "0",
            "--set", "bad-key=1",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--set", (r.stdout + r.stderr))

    def test_profile_decline_confirm_returns_zero(self):
        # User explicitly answering "n" to confirmation is not an error.
        # Pre-create a throwaway profile.
        _run_cli(
            self.tmp, "profile", "new", "to-decline", "--backend", "vllm",
            "--port", "8006", "--gpu-id", "0", "--model", "x/y",
        )
        # Pipe "n" to stdin to decline.
        env = {**os.environ, "LLMUX_ROOT": str(self.tmp), "NO_COLOR": "1"}
        r = subprocess.run(
            [sys.executable, "-m", "tui", "profile", "delete", "to-decline"],
            input="n\n",
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        # Profile should still exist.
        r2 = _run_cli(self.tmp, "profile", "show", "to-decline", "--json")
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        # Cleanup
        _run_cli(self.tmp, "profile", "delete", "to-decline", "-y")

    def test_config_show_unknown_backend_explicit(self):
        # --backend llamacpp + name only existing in vllm → clear error
        r = _run_cli(self.tmp, "config", "show", "alpha", "--backend", "llamacpp")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not found", (r.stdout + r.stderr).lower())

    def test_profile_show_llamacpp(self):
        # Symmetric to vllm — make sure llamacpp profiles also dispatch
        # correctly via auto-detect.
        r = _run_cli(self.tmp, "profile", "show", "lcpp", "--json")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual(data["backend"], "llamacpp")
        self.assertEqual(data["port"], 8080)


    def test_quick_setup_llamacpp_writes_profile_and_config(self):
        """Regression: `_quick_setup_llamacpp` used `run_async` without
        importing it from `_runtime`, causing NameError on first call. This
        exercises the full end-to-end path with `list_hf_repo_files` stubbed
        so the test stays hermetic.
        """
        r = _run_cli_stubbed_hf(
            self.tmp,
            ["model-Q4_K_M.gguf", "model-Q8_0.gguf"],
            "profile", "quick-setup",
            "--backend", "llamacpp",
            "--hf-repo", "fake/Model-GGUF",
            "--hf-file", "model-Q4_K_M.gguf",
            "--name", "lc-smoke",
            "--port", "8087",
            "--gpu-id", "0",
            "--ctx-size", "1024",
            "--n-gpu-layers", "0",
            "--cache-type-k", "f16",
            "--cache-type-v", "f16",
            "--no-flash-attn",
            "--no-jinja",
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Created profile + config: lc-smoke", r.stdout)

        # Profile written through profile_store.save_profile
        r2 = _run_cli(self.tmp, "profile", "show", "lc-smoke", "--json")
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        data = json.loads(r2.stdout)
        self.assertEqual(data["backend"], "llamacpp")
        self.assertEqual(data["port"], 8087)
        self.assertEqual(data["hf_repo"], "fake/Model-GGUF")
        self.assertEqual(data["hf_file"], "model-Q4_K_M.gguf")
        self.assertEqual(data["model_file"], "model-Q4_K_M.gguf")

        # Config written through llamacpp backend save_config — verify the
        # key set + ints are typed (not str).
        cfg_path = self.tmp / "config" / "llamacpp" / "lc-smoke.yaml"
        self.assertTrue(cfg_path.exists(), cfg_path)
        cfg = yaml.safe_load(cfg_path.read_text())
        self.assertEqual(cfg.get("model-file"), "model-Q4_K_M.gguf")
        self.assertEqual(cfg.get("alias"), "lc-smoke")
        self.assertEqual(cfg.get("ctx-size"), 1024)
        self.assertEqual(cfg.get("n-gpu-layers"), 0)
        self.assertEqual(cfg.get("cache-type-k"), "f16")
        self.assertEqual(cfg.get("cache-type-v"), "f16")
        # --no-flash-attn / --no-jinja → keys must NOT be present.
        self.assertNotIn("flash-attn", cfg)
        self.assertNotIn("jinja", cfg)
        # Cleanup.
        _run_cli(self.tmp, "profile", "delete", "lc-smoke", "-y")

    def test_quick_setup_vllm_accepts_recipe_inputs(self):
        result = _run_cli_stubbed_recipe(
            self.tmp,
            "profile",
            "quick-setup",
            "custom/Quantized",
            "--backend",
            "vllm",
            "--name",
            "recipe-profile",
            "--recipe-from",
            "Qwen/Base",
            "--variant",
            "fp8",
            "--feature",
            "reasoning",
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        config = json.loads(
            _run_cli(
                self.tmp,
                "config",
                "show",
                "recipe-profile",
                "--json",
            ).stdout
        )["params"]
        self.assertEqual(config["model"], "custom/Quantized")
        self.assertEqual(config["max-model-len"], 8192)
        self.assertEqual(config["quantization"], "fp8")
        self.assertEqual(config["reasoning-parser"], "qwen3")
        _run_cli(
            self.tmp,
            "profile",
            "delete",
            "recipe-profile",
            "--yes",
            "--with-config",
        )

    def test_quick_setup_llamacpp_rejects_non_integer_ctx_size(self):
        # --ctx-size / --n-gpu-layers must validate as ints up front
        # (like --port / --gpu-id), not silently store a string that only breaks
        # at llama-server start.
        r = _run_cli_stubbed_hf(
            self.tmp,
            ["model-Q4_K_M.gguf"],
            "profile", "quick-setup",
            "--backend", "llamacpp",
            "--hf-repo", "fake/Model-GGUF",
            "--hf-file", "model-Q4_K_M.gguf",
            "--name", "lc-badctx",
            "--port", "8089",
            "--ctx-size", "notanumber",
        )
        self.assertNotEqual(r.returncode, 0)
        out = r.stderr + r.stdout
        self.assertIn("--ctx-size", out)
        self.assertIn("integer", out)
        # Nothing should have been persisted.
        show = _run_cli(self.tmp, "profile", "show", "lc-badctx", "--json")
        self.assertNotEqual(show.returncode, 0)

    def test_quick_setup_llamacpp_warns_on_positional_model(self):
        """Positional MODEL has no meaning for --backend llamacpp (GGUF profiles
        derive their identity from --hf-repo / --hf-file). Silently dropping
        the argument hides scripted typos — verify we warn on stderr but still
        proceed with the auto-derived name (no hard fail, no behavior change).
        """
        r = _run_cli_stubbed_hf(
            self.tmp,
            ["model-Q4_K_M.gguf"],
            "profile", "quick-setup",
            "smalltest",  # positional MODEL — meaningless for llamacpp
            "--backend", "llamacpp",
            "--hf-repo", "fake/Model-GGUF",
            "--hf-file", "model-Q4_K_M.gguf",
            "--port", "8089",
            "--gpu-id", "0",
        )
        # Should still succeed — auto-derive from repo tail.
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Warning", r.stderr)
        self.assertIn("smalltest", r.stderr)
        self.assertIn("ignored", r.stderr.lower())
        self.assertIn("--name", r.stderr)
        # Auto-derived name (lowercased, stripped, -GGUF removed).
        self.assertIn("Created profile + config: model", r.stdout)
        # Cleanup any auto-derived name that actually got created.
        r2 = _run_cli(self.tmp, "profile", "list", "--json", "--backend", "llamacpp")
        if r2.returncode == 0:
            for row in json.loads(r2.stdout):
                if row["name"].startswith("model"):
                    _run_cli(
                        self.tmp, "profile", "delete", row["name"],
                        "-y", "--with-config",
                    )

    def test_quick_setup_llamacpp_warns_when_positional_with_explicit_name(self):
        """When both positional MODEL and --name are given for --backend
        llamacpp, the positional is still ignored — but the warning should
        reference the explicit --name so the user can see which one won.
        """
        r = _run_cli_stubbed_hf(
            self.tmp,
            ["model-Q4_K_M.gguf"],
            "profile", "quick-setup",
            "smalltest",
            "--backend", "llamacpp",
            "--hf-repo", "fake/Model-GGUF",
            "--hf-file", "model-Q4_K_M.gguf",
            "--name", "lc-explicit",
            "--port", "8090",
            "--gpu-id", "0",
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Warning", r.stderr)
        self.assertIn("smalltest", r.stderr)
        self.assertIn("lc-explicit", r.stderr)
        self.assertIn("Created profile + config: lc-explicit", r.stdout)
        _run_cli(self.tmp, "profile", "delete", "lc-explicit", "-y", "--with-config")

    def test_quick_setup_llamacpp_rejects_file_not_in_repo(self):
        """If the HF API returns a listing and --hf-file isn't in it, error
        out with a clear message (parity with the TUI's Fetch+select gate).
        """
        r = _run_cli_stubbed_hf(
            self.tmp,
            ["only-Q4_K_M.gguf"],
            "profile", "quick-setup",
            "--backend", "llamacpp",
            "--hf-repo", "fake/Model-GGUF",
            "--hf-file", "missing-Q8_0.gguf",
            "--name", "lc-bad",
            "--port", "8088",
            "--gpu-id", "0",
        )
        self.assertNotEqual(r.returncode, 0)
        combined = (r.stdout or "") + (r.stderr or "")
        self.assertIn("not found", combined.lower())
        self.assertIn("only-Q4_K_M.gguf", combined)

    def test_quick_setup_llamacpp_rejects_repo_with_no_gguf_files(self):
        r = _run_cli_stubbed_hf(
            self.tmp,
            [],
            "profile",
            "quick-setup",
            "--backend",
            "llamacpp",
            "--hf-repo",
            "fake/Empty",
            "--hf-file",
            "invented.gguf",
            "--name",
            "empty-repo",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no GGUF", r.stderr + r.stdout)

    def test_quick_setup_llamacpp_saves_batch_size(self):
        r = _run_cli_stubbed_hf(
            self.tmp,
            ["model.gguf"],
            "profile",
            "quick-setup",
            "--backend",
            "llamacpp",
            "--hf-repo",
            "fake/Model-GGUF",
            "--hf-file",
            "model.gguf",
            "--name",
            "batch-profile",
            "--batch-size",
            "512",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        shown = _run_cli(
            self.tmp,
            "config",
            "show",
            "batch-profile",
            "--backend",
            "llamacpp",
            "--json",
        )
        self.assertEqual(shown.returncode, 0, shown.stderr or shown.stdout)
        self.assertEqual(json.loads(shown.stdout)["params"]["batch-size"], 512)
        _run_cli(
            self.tmp,
            "profile",
            "delete",
            "batch-profile",
            "--yes",
            "--with-config",
        )


if __name__ == "__main__":
    unittest.main()
