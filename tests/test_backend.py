import json
import os
import subprocess
import sys
import tempfile
import importlib.util
import unittest
from contextlib import contextmanager

import pytest
import yaml
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from tui.backends.llamacpp import backend as lbackend
from tui.backends.llamacpp import backend_runtime as lbackend_rt
from tui.backends.vllm import backend
from tui.common import ssl_ctx
from tui.backends.vllm.backend_inspect import (
    _pick_preferred_tag,
    get_dockerhub_nightly_date,
    get_dockerhub_release_version,
)
from tui.backends.vllm.backend_runtime import (
    _build_lora_options,
    _detect_gpu_arch,
    _ensure_common_env,
    _force_local_arch_for_deepep,
    _gpu_conflict_messages,
    _post_start_validation,
    _verify_vllm_version,
)
from tui.common import profile_store


@pytest.fixture
def isolated_vllm_storage(tmp_path, monkeypatch):
    from tui.backends.vllm import backend_common, backend_storage

    profiles_yaml = tmp_path / "profiles.yaml"
    profiles_yaml.write_text("version: 1\ndefaults: {}\nprofiles: []\n")
    runtime_root = tmp_path / ".runtime"
    runtime_dir = runtime_root / "vllm"
    config_dir = tmp_path / "config" / "vllm"
    config_dir.mkdir(parents=True)
    monkeypatch.setattr(profile_store, "PROFILES_YAML", profiles_yaml)
    monkeypatch.setattr(profile_store, "RUNTIME_DIR", runtime_root)
    monkeypatch.setattr(backend_common, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(backend_common, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(backend_storage, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(backend, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(backend, "CONFIG_DIR", config_dir)


@pytest.mark.usefixtures("isolated_vllm_storage")
class LoadConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._created: list[Path] = []

    def tearDown(self) -> None:
        for path in self._created:
            path.unlink(missing_ok=True)

    def _write_config(self, name: str, content: str) -> None:
        path = backend.CONFIG_DIR / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self._created.append(path)

    def test_load_config_reads_mapping_yaml(self) -> None:
        self._write_config(
            "__test_valid_config__",
            "model: org/model\ngpu-memory-utilization: 0.8\nmax-model-len: 4096\n",
        )

        config = backend.load_config("__test_valid_config__")

        self.assertEqual(config.name, "__test_valid_config__")
        self.assertEqual(config.model, "org/model")
        self.assertEqual(config.gpu_memory_utilization, "0.8")
        self.assertEqual(config.extra_params, {"max-model-len": 4096})

    def test_load_config_rejects_non_mapping_yaml(self) -> None:
        self._write_config("__test_invalid_config__", "- not\n- a\n- mapping\n")

        with self.assertRaisesRegex(ValueError, "mapping"):
            backend.load_config("__test_invalid_config__")

    def test_load_config_missing_file_returns_defaults(self) -> None:
        config = backend.load_config("__test_missing_config_xyz__")

        self.assertEqual(config.name, "__test_missing_config_xyz__")
        self.assertEqual(config.model, "")
        self.assertEqual(config.gpu_memory_utilization, "0.9")
        self.assertEqual(config.extra_params, {})


class VllmComposeSecurityTests(unittest.TestCase):
    def test_trust_remote_code_is_not_forced(self) -> None:
        compose_path = Path(__file__).resolve().parents[1] / "compose" / "vllm" / "docker-compose.yaml"
        service = yaml.safe_load(compose_path.read_text())["services"]["vllm"]
        self.assertNotIn("--trust-remote-code", service["command"])

    def test_config_mount_is_read_only(self) -> None:
        compose_path = Path(__file__).resolve().parents[1] / "compose" / "vllm" / "docker-compose.yaml"
        service = yaml.safe_load(compose_path.read_text())["services"]["vllm"]
        self.assertTrue(any(volume.endswith(":/config/:ro") for volume in service["volumes"]))

    def test_backend_ports_bind_only_to_loopback(self) -> None:
        root = Path(__file__).resolve().parents[1] / "compose"
        cases = (
            (root / "vllm" / "docker-compose.yaml", "vllm"),
            (root / "llamacpp" / "docker-compose.yaml", "llama-server"),
        )
        for compose_path, service_name in cases:
            with self.subTest(service=service_name):
                service = yaml.safe_load(compose_path.read_text())["services"][service_name]
                self.assertTrue(all(port.startswith("127.0.0.1:") for port in service["ports"]))


@pytest.mark.usefixtures("isolated_vllm_storage")
class SaveLoadConfigRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self._name = "__test_roundtrip_config__"
        self._path = backend.CONFIG_DIR / f"{self._name}.yaml"

    def tearDown(self) -> None:
        self._path.unlink(missing_ok=True)

    def test_save_then_load_preserves_fields(self) -> None:
        cfg = backend.Config(
            name=self._name,
            model="org/model",
            gpu_memory_utilization="0.75",
            extra_params={"max-model-len": 4096, "trust-remote-code": True},
        )
        backend.save_config(cfg)

        loaded = backend.load_config(self._name)
        self.assertEqual(loaded.model, "org/model")
        self.assertEqual(loaded.gpu_memory_utilization, "0.75")
        self.assertEqual(loaded.extra_params["max-model-len"], 4096)
        self.assertIs(loaded.extra_params["trust-remote-code"], True)

    def test_save_empty_string_value_becomes_true_flag(self) -> None:
        cfg = backend.Config(
            name=self._name,
            model="org/model",
            extra_params={"enforce-eager": ""},
        )
        backend.save_config(cfg)

        loaded = backend.load_config(self._name)
        self.assertIs(loaded.extra_params["enforce-eager"], True)


class ConfigParamValueTests(unittest.TestCase):
    def test_parse_blank_returns_true(self) -> None:
        self.assertIs(backend.parse_config_param_value(""), True)

    def test_parse_int_and_float(self) -> None:
        self.assertEqual(backend.parse_config_param_value("4096"), 4096)
        self.assertEqual(backend.parse_config_param_value("0.9"), 0.9)

    def test_parse_bool_and_null(self) -> None:
        self.assertIs(backend.parse_config_param_value("false"), False)
        self.assertIsNone(backend.parse_config_param_value("null"))

    def test_parse_list(self) -> None:
        self.assertEqual(backend.parse_config_param_value("[a, b, c]"), ["a", "b", "c"])

    def test_parse_invalid_yaml_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid YAML"):
            backend.parse_config_param_value("{unbalanced")

    def test_format_true_becomes_empty(self) -> None:
        self.assertEqual(backend.format_config_param_value(True), "")

    def test_format_false_preserved_as_string(self) -> None:
        self.assertEqual(backend.format_config_param_value(False), "false")

    def test_format_none_is_null(self) -> None:
        self.assertEqual(backend.format_config_param_value(None), "null")

    def test_format_list_uses_flow_style(self) -> None:
        formatted = backend.format_config_param_value([1, 2, 3])
        self.assertIn("[", formatted)
        self.assertIn("1", formatted)


class ValidateNameTests(unittest.TestCase):
    def test_accepts_plain_names(self) -> None:
        self.assertTrue(backend.validate_name("abc"))
        self.assertTrue(backend.validate_name("my-profile_01"))
        self.assertTrue(backend.validate_name("qwen3-0-8b"))

    def test_rejects_uppercase(self) -> None:
        # docker compose project names are lowercase-only — both backends
        # now share that rule so a profile cannot validate on one backend
        # and fail on the other.
        self.assertFalse(backend.validate_name("Qwen3-0-8b"))
        self.assertFalse(backend.validate_name("UPPER"))

    def test_rejects_leading_dash(self) -> None:
        self.assertFalse(backend.validate_name("-injection"))

    def test_rejects_special_chars(self) -> None:
        self.assertFalse(backend.validate_name("name with space"))
        self.assertFalse(backend.validate_name("path/traversal"))
        self.assertFalse(backend.validate_name("dot.in.name"))
        self.assertFalse(backend.validate_name(""))


class ParseEnvFileTests(unittest.TestCase):
    def _write(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
        tmp.write(content)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        return Path(tmp.name)

    def test_missing_file_returns_empty_dict(self) -> None:
        missing = Path(tempfile.gettempdir()) / "__definitely_missing_env_file__"
        self.assertEqual(backend._parse_env_file(missing), {})

    def test_parses_basic_key_value(self) -> None:
        path = self._write("KEY=value\nOTHER=123\n")
        self.assertEqual(backend._parse_env_file(path), {"KEY": "value", "OTHER": "123"})

    def test_ignores_comments_and_blank_lines(self) -> None:
        path = self._write("# comment\n\nKEY=value\n# another\n")
        self.assertEqual(backend._parse_env_file(path), {"KEY": "value"})

    def test_strips_inline_comments_when_unquoted(self) -> None:
        path = self._write("KEY=value # trailing\n")
        self.assertEqual(backend._parse_env_file(path), {"KEY": "value"})

    def test_preserves_quoted_values_including_hash(self) -> None:
        path = self._write('KEY="value # keep"\n')
        self.assertEqual(backend._parse_env_file(path), {"KEY": "value # keep"})

    def test_strips_matching_quotes(self) -> None:
        path = self._write("KEY='quoted'\nOTHER=\"double\"\n")
        self.assertEqual(
            backend._parse_env_file(path),
            {"KEY": "quoted", "OTHER": "double"},
        )

    def test_double_quoted_value_drops_trailing_comment(self) -> None:
        # compose reads `a b`; the old shlex parser kept the quotes + comment.
        path = self._write('KEY="a b" # comment\n')
        self.assertEqual(backend._parse_env_file(path), {"KEY": "a b"})

    def test_unquoted_backslash_is_literal(self) -> None:
        # compose keeps `a\b`; the old shlex(posix) parser swallowed the `\`.
        path = self._write("KEY=a\\b\n")
        self.assertEqual(backend._parse_env_file(path), {"KEY": "a\\b"})

    def test_single_quoted_is_literal(self) -> None:
        path = self._write("KEY='a\\b'\n")
        self.assertEqual(backend._parse_env_file(path), {"KEY": "a\\b"})

    def test_double_quote_escapes_interpreted(self) -> None:
        path = self._write('KEY="a\\nb"\n')
        self.assertEqual(backend._parse_env_file(path), {"KEY": "a\nb"})

    def test_our_renderer_single_quote_output_round_trips(self) -> None:
        # `_env_line` emits `FOO='a b'` via shlex.quote — must still read back.
        path = self._write("FOO='a b'\nBAR=0,1\n")
        self.assertEqual(
            backend._parse_env_file(path), {"FOO": "a b", "BAR": "0,1"}
        )

    def test_export_keyword_is_stripped(self) -> None:
        # dotenv/godotenv drop a leading `export`; without it the key became
        # `export HF_TOKEN` and the real var was silently lost.
        path = self._write("export HF_TOKEN=secret\nexport\tK2=v2\nK3=v3\n")
        self.assertEqual(
            backend._parse_env_file(path),
            {"HF_TOKEN": "secret", "K2": "v2", "K3": "v3"},
        )


@pytest.mark.usefixtures("isolated_vllm_storage")
class SaveLoadProfileRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self._name = "test-roundtrip-profile"
        self._path = backend.RUNTIME_DIR / f"{self._name}.env"

    def tearDown(self) -> None:
        backend.delete_profile(self._name)

    def test_save_then_load_preserves_fields(self) -> None:
        profile = backend.Profile(
            name=self._name,
            container_name="my-container",
            port="8123",
            gpu_id="0,1",
            tensor_parallel="2",
            config_name="my-config",
            model_id="org/model",
            enable_lora="true",
            max_loras="4",
            max_lora_rank="32",
            lora_modules="alpha=/path/a,beta=/path/b",
            extra_pip_packages="pkg-a pkg-b",
            env_vars={"CUSTOM": "x"},
        )
        backend.save_profile(profile)

        loaded = backend.load_profile(self._name)
        self.assertEqual(loaded.container_name, "my-container")
        self.assertEqual(loaded.port, "8123")
        self.assertEqual(loaded.gpu_id, "0,1")
        self.assertEqual(loaded.tensor_parallel, "2")
        self.assertEqual(loaded.config_name, "my-config")
        self.assertEqual(loaded.model_id, "org/model")
        self.assertEqual(loaded.enable_lora, "true")
        self.assertEqual(loaded.max_loras, "4")
        self.assertEqual(loaded.max_lora_rank, "32")
        self.assertEqual(loaded.lora_modules, "alpha=/path/a,beta=/path/b")
        # `extra_pip_packages` is now a first-class field on Profile (no
        # longer stuffed through env_vars["EXTRA_PIP_PACKAGES"]).
        self.assertEqual(loaded.extra_pip_packages, "pkg-a pkg-b")
        self.assertEqual(loaded.env_vars["CUSTOM"], "x")

    def test_save_rejects_reserved_env_var(self) -> None:
        # env_vars must not be able to shadow GPU_ID / VLLM_PORT / etc — that
        # would let conflict checks see one value and the rendered .env emit
        # another. Saving with such an override raises at render time and
        # leaves profiles.yaml untouched (atomic save_profile).
        profile = backend.Profile(
            name=self._name,
            env_vars={"GPU_ID": "9"},
        )
        with self.assertRaises(ValueError):
            backend.save_profile(profile)


class ProfileStoreYamlTests(unittest.TestCase):
    def test_empty_non_mapping_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles_yaml = Path(tmp) / "profiles.yaml"
            profiles_yaml.write_text("[]\n")
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml):
                with self.assertRaisesRegex(ValueError, "must be a mapping"):
                    profile_store.list_profiles("vllm")

    def test_non_mapping_defaults_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles_yaml = Path(tmp) / "profiles.yaml"
            profiles_yaml.write_text("version: 1\ndefaults: []\nprofiles: []\n")
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml):
                with self.assertRaisesRegex(ValueError, "defaults must be a mapping"):
                    profile_store.list_profiles("vllm")

    def test_non_mapping_backend_defaults_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles_yaml = Path(tmp) / "profiles.yaml"
            profiles_yaml.write_text(
                "version: 1\ndefaults:\n  vllm: invalid\nprofiles: []\n"
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml):
                with self.assertRaisesRegex(ValueError, "defaults.vllm must be a mapping"):
                    profile_store.list_profiles("vllm")

    def test_user_defaults_are_applied_when_profile_omits_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = root / "profiles.yaml"
            profiles_yaml.write_text(
                "version: 1\n"
                "defaults:\n"
                "  vllm:\n"
                "    port: 9100\n"
                "    gpu_id: '2'\n"
                "    tensor_parallel_size: 2\n"
                "    enable_lora: false\n"
                "profiles:\n"
                "  - name: p\n"
                "    backend: vllm\n"
                "    model_id: org/model\n"
            )

            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                stored = profile_store.load_profile("p", "vllm")

            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(stored.port, 9100)
            self.assertEqual(stored.gpu_id, "2")
            self.assertEqual(stored.tensor_parallel_size, 2)

    def test_string_false_is_parsed_as_false(self) -> None:
        stored = profile_store._to_profile(
            {
                "name": "p",
                "backend": "vllm",
                "enable_lora": "false",
            }
        )

        self.assertFalse(stored.enable_lora)

    def test_render_env_quotes_shell_sensitive_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = profile_store.StoredProfile(
                name="p",
                backend="llamacpp",
                model_file="$(touch /tmp/llmux-pwned) model.gguf",
            )

            with patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"):
                path = profile_store.render_env(profile)

            rendered = path.read_text()
            self.assertIn("MODEL_FILE='$(touch /tmp/llmux-pwned) model.gguf'", rendered)

    def test_render_env_rejects_invalid_env_var_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = profile_store.StoredProfile(
                name="p",
                backend="vllm",
                env_vars={"BAD-NAME": "value"},
            )

            with patch("tui.common.profile_store.RUNTIME_DIR", Path(tmp) / ".runtime"):
                with self.assertRaises(ValueError):
                    profile_store.render_env(profile)

    def test_effective_defaults_applies_user_overrides(self) -> None:
        # The `--port 0` sentinel resolves through effective_defaults(); reading
        # the hardcoded DEFAULTS instead would return 8080 and silently ignore
        # the user's profiles.yaml `defaults:` block (which the loader honors).
        with tempfile.TemporaryDirectory() as tmp:
            profiles_yaml = Path(tmp) / "profiles.yaml"
            profiles_yaml.write_text(
                "version: 1\n"
                "defaults:\n"
                "  llamacpp:\n"
                "    port: 9000\n"
                "profiles: []\n"
            )

            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml):
                defaults = profile_store.effective_defaults("llamacpp")
                vllm_defaults = profile_store.effective_defaults("vllm")

            self.assertEqual(defaults["port"], 9000)
            self.assertEqual(defaults["gpu_id"], "0")  # untouched key still present
            # A backend with no user override falls back to the built-in value.
            self.assertEqual(vllm_defaults["port"], 8000)

    def test_sanitize_docker_tag_maps_branch_names_to_valid_tags(self) -> None:
        from tui.common.dev_build import sanitize_docker_tag

        self.assertEqual(sanitize_docker_tag("feat/foo"), "feat-foo")
        self.assertEqual(sanitize_docker_tag("releases/v0.21.0"), "releases-v0.21.0")
        self.assertEqual(sanitize_docker_tag("main"), "main")
        # Tags may not start with `.` or `-`.
        self.assertEqual(sanitize_docker_tag("/leading"), "leading")
        self.assertEqual(sanitize_docker_tag("///"), "branch")

    def test_llamacpp_env_vars_round_trip_through_yaml(self) -> None:
        # env_vars used to be persisted only on the vllm branch, so a llamacpp
        # profile silently dropped them between save and load.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = root / "profiles.yaml"
            profiles_yaml.write_text("version: 1\ndefaults: {}\nprofiles: []\n")
            profile = profile_store.StoredProfile(
                name="p",
                backend="llamacpp",
                hf_repo="org/Model-GGUF",
                env_vars={"MY_VAR": "hello"},
            )

            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                profile_store.save_profile(profile)
                loaded = profile_store.load_profile("p", "llamacpp")

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.env_vars, {"MY_VAR": "hello"})

    def test_llamacpp_render_env_includes_env_vars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = profile_store.StoredProfile(
                name="p",
                backend="llamacpp",
                env_vars={"MY_VAR": "hello"},
            )

            with patch("tui.common.profile_store.RUNTIME_DIR", Path(tmp) / ".runtime"):
                path = profile_store.render_env(profile)

            self.assertIn("MY_VAR=hello", path.read_text())

    def test_llamacpp_render_env_rejects_reserved_env_var(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = profile_store.StoredProfile(
                name="p",
                backend="llamacpp",
                env_vars={"GPU_ID": "9"},
            )

            with patch("tui.common.profile_store.RUNTIME_DIR", Path(tmp) / ".runtime"):
                with self.assertRaises(ValueError):
                    profile_store.render_env(profile)

    def test_save_profile_does_not_write_yaml_when_env_render_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = root / "profiles.yaml"
            profiles_yaml.write_text("version: 1\ndefaults: {}\nprofiles: []\n")
            profile = profile_store.StoredProfile(
                name="p",
                backend="vllm",
                env_vars={"BAD-NAME": "value"},
            )

            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                with self.assertRaises(ValueError):
                    profile_store.save_profile(profile)

            self.assertEqual(
                profiles_yaml.read_text(), "version: 1\ndefaults: {}\nprofiles: []\n"
            )

    def test_save_profile_rolls_back_env_when_yaml_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = root / "profiles.yaml"
            original_yaml = "version: 1\ndefaults: {}\nprofiles: []\n"
            profiles_yaml.write_text(original_yaml)
            runtime_dir = root / ".runtime"
            env_path = runtime_dir / "vllm" / "p.env"
            env_path.parent.mkdir(parents=True)
            env_path.write_text("OLD=1\n")
            real_replace = os.replace
            failed = False

            def fail_yaml_once(src, dst):
                nonlocal failed
                if Path(dst) == profiles_yaml and not failed:
                    failed = True
                    raise OSError("yaml replace failed")
                return real_replace(src, dst)

            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", runtime_dir
            ), patch("tui.common.profile_store.os.replace", side_effect=fail_yaml_once):
                with self.assertRaisesRegex(OSError, "yaml replace failed"):
                    profile_store.save_profile(
                        profile_store.StoredProfile(name="p", backend="vllm")
                    )

            self.assertEqual(profiles_yaml.read_text(), original_yaml)
            self.assertEqual(env_path.read_text(), "OLD=1\n")

    def test_render_env_rejects_single_quote_values(self) -> None:
        # This used to assert the value round-tripped through _parse_env_file —
        # it does, because that parser is shlex-based. docker compose's dotenv
        # parser is not: shlex.quote emits `'O'"'"'Reilly model'` and compose
        # fails to parse the whole file, so `up` broke long after the save
        # "succeeded". render_env now refuses the value instead.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = profile_store.StoredProfile(
                name="p",
                backend="vllm",
                model_id="O'Reilly model",
            )

            with patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"):
                with self.assertRaises(ValueError) as ctx:
                    profile_store.render_env(profile)

            self.assertIn("single quote", str(ctx.exception))

    def test_invalid_backend_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            profile_store.list_profiles("bogus")  # type: ignore[arg-type]

    def test_profile_store_cli_invalid_backend_returns_usage_error(self) -> None:
        with patch("sys.argv", ["profile_store", "list", "bogus"]):
            self.assertEqual(profile_store._cli(), 2)


class AuditFindingsProfileStoreTests(unittest.TestCase):
    """Regressions for the profiles.yaml write path: dedup and global name
    uniqueness."""

    @staticmethod
    def _write_yaml(root: Path, body: str) -> Path:
        p = root / "profiles.yaml"
        p.write_text(body)
        return p

    def test_save_profile_rejects_non_mapping_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._write_yaml(
                root,
                "version: 1\nprofiles:\n  - foo\n  - name: keep\n    backend: vllm\n",
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                with self.assertRaisesRegex(ValueError, "mapping"):
                    profile_store.save_profile(
                        profile_store.StoredProfile(name="added", backend="vllm")
                    )
            self.assertIn("- foo", profiles_yaml.read_text())

    def test_delete_profile_rejects_non_mapping_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._write_yaml(
                root,
                "version: 1\nprofiles:\n  - foo\n  - name: gone\n    backend: vllm\n",
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                with self.assertRaisesRegex(ValueError, "mapping"):
                    profile_store.delete_profile("gone", "vllm")
            self.assertIn("name: gone", profiles_yaml.read_text())

    def test_list_profiles_rejects_duplicate_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._write_yaml(
                root,
                "version: 1\nprofiles:\n"
                "  - name: dup\n    backend: vllm\n    port: 8001\n"
                "  - name: dup\n    backend: vllm\n    port: 8002\n",
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml):
                with self.assertRaisesRegex(ValueError, "duplicate profile name"):
                    profile_store.list_profiles("vllm")

    def test_find_name_owner_spans_both_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._write_yaml(
                root,
                "version: 1\nprofiles:\n  - name: shared\n    backend: llamacpp\n",
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml):
                self.assertEqual(profile_store.find_name_owner("shared"), "llamacpp")
                self.assertIsNone(profile_store.find_name_owner("free"))
                # The profile being edited excludes itself.
                self.assertIsNone(
                    profile_store.find_name_owner(
                        "shared", exclude=("llamacpp", "shared")
                    )
                )

    def test_raw_profile_name_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._write_yaml(
                root,
                "version: 1\nprofiles:\n  - name: ../outside\n    backend: vllm\n",
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml):
                with self.assertRaisesRegex(ValueError, "profile name"):
                    profile_store.list_profiles("vllm")

    def test_raw_config_name_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._write_yaml(
                root,
                "version: 1\nprofiles:\n"
                "  - name: safe\n    backend: vllm\n    config_name: ../../outside\n",
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml):
                with self.assertRaisesRegex(ValueError, "config name"):
                    profile_store.list_profiles("vllm")

    def test_duplicate_effective_container_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._write_yaml(
                root,
                "version: 1\nprofiles:\n"
                "  - name: first\n    backend: vllm\n    container_name: shared\n"
                "  - name: second\n    backend: llamacpp\n    container_name: shared\n",
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml):
                with self.assertRaisesRegex(ValueError, "container name"):
                    profile_store.list_profiles("vllm")

    def test_non_mapping_profile_entry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._write_yaml(root, "version: 1\nprofiles:\n  - bad\n")
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml):
                with self.assertRaisesRegex(ValueError, "mapping"):
                    profile_store.list_profiles("vllm")

    def test_save_profile_rejects_duplicate_container_name_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._write_yaml(
                root,
                "version: 1\nprofiles:\n"
                "  - name: first\n    backend: vllm\n    container_name: shared\n",
            )
            original = profiles_yaml.read_text()
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                with self.assertRaisesRegex(ValueError, "container name"):
                    profile_store.save_profile(
                        profile_store.StoredProfile(
                            name="second",
                            backend="llamacpp",
                            container_name="shared",
                        )
                    )
            self.assertEqual(profiles_yaml.read_text(), original)


class DeriveConfigNameTests(unittest.TestCase):
    def test_dotted_model_id_drops_dots_to_match_new_name_rule(self) -> None:
        # `config from-recipe <dotted id>` with no --name derived a dotted name
        # that _NEW_NAME_RE (no dots) then rejected — now dots collapse to `-`.
        from tui.cli.config import _NEW_NAME_RE, _derive_config_name

        name = _derive_config_name("Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(name, "qwen2-5-7b-instruct")
        self.assertTrue(_NEW_NAME_RE.match(name))


class InvalidConfigYamlTests(unittest.TestCase):
    def test_vllm_load_rejects_non_mapping_yaml(self) -> None:
        from tui.backends.vllm import backend_storage

        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "bad.yaml").write_text("- model\n- value\n")
            with patch.object(backend_storage, "CONFIG_DIR", config_dir):
                with self.assertRaisesRegex(ValueError, "mapping"):
                    backend_storage.load_config("bad")

    def test_llamacpp_load_rejects_non_mapping_yaml(self) -> None:
        from tui.backends.llamacpp import backend

        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "bad.yaml").write_text("plain scalar\n")
            with patch.object(backend, "CONFIG_DIR", config_dir):
                with self.assertRaisesRegex(ValueError, "mapping"):
                    backend.load_config("bad")


class RenameProfileTests(unittest.TestCase):
    """profile_store.rename_profile — name/env/config-link bookkeeping."""

    def _seed(self, root: Path, entry: str) -> Path:
        profiles_yaml = root / "profiles.yaml"
        profiles_yaml.write_text(
            "version: 1\ndefaults: {}\nprofiles:\n" + entry
        )
        return profiles_yaml

    def test_rename_moves_entry_and_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._seed(
                root, "- name: old\n  backend: vllm\n  port: 8001\n"
            )
            runtime = root / ".runtime"
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", runtime
            ):
                old_env = profile_store.render_env(
                    profile_store.load_profile("old", "vllm")
                )
                self.assertTrue(old_env.exists())
                profile_store.rename_profile("old", "new", "vllm")

                self.assertIsNone(profile_store.load_profile("old", "vllm"))
                renamed = profile_store.load_profile("new", "vllm")
                self.assertIsNotNone(renamed)
                assert renamed is not None
                self.assertEqual(renamed.port, 8001)
                self.assertFalse(old_env.exists())
                self.assertTrue((runtime / "vllm" / "new.env").exists())

    def test_rename_pins_implicit_config_to_old_name(self) -> None:
        # No config_name key means "resolve to the profile name"; renaming
        # without pinning would silently repoint the profile at a config that
        # does not exist.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._seed(root, "- name: old\n  backend: vllm\n")
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                renamed = profile_store.rename_profile("old", "new", "vllm")
                self.assertEqual(renamed.config_name, "old")
                self.assertEqual(
                    profile_store.load_profile("new", "vllm").config_name, "old"
                )

    def test_rename_keeps_explicit_config_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._seed(
                root, "- name: old\n  backend: vllm\n  config_name: shared\n"
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                renamed = profile_store.rename_profile("old", "new", "vllm")
                self.assertEqual(renamed.config_name, "shared")

    def test_rename_lets_implicit_container_follow_new_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._seed(root, "- name: old\n  backend: vllm\n")
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                renamed = profile_store.rename_profile("old", "new", "vllm")
                self.assertEqual(renamed.container_name, "new")

    def test_rename_keeps_explicit_container_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._seed(
                root, "- name: old\n  backend: vllm\n  container_name: pinned\n"
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                renamed = profile_store.rename_profile("old", "new", "vllm")
                self.assertEqual(renamed.container_name, "pinned")

    def test_rename_rejects_name_taken_in_other_backend(self) -> None:
        # Names are global: a vllm/x and llamacpp/x would share one container.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._seed(
                root,
                "- name: old\n  backend: vllm\n- name: taken\n  backend: llamacpp\n",
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                with self.assertRaises(ValueError):
                    profile_store.rename_profile("old", "taken", "vllm")
                self.assertIsNotNone(profile_store.load_profile("old", "vllm"))

    def test_rename_rejects_unknown_and_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = self._seed(root, "- name: old\n  backend: vllm\n")
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                with self.assertRaises(ValueError):
                    profile_store.rename_profile("old", "old", "vllm")
                with self.assertRaises(ValueError):
                    profile_store.rename_profile("ghost", "new", "vllm")


class RenameConfigTests(unittest.TestCase):
    """config_store.rename_config — file move + profile reference repair."""

    def _setup(self, root: Path, profiles: str) -> tuple[Path, Path]:
        profiles_yaml = root / "profiles.yaml"
        profiles_yaml.write_text("version: 1\ndefaults: {}\nprofiles:\n" + profiles)
        config_dir = root / "config" / "vllm"
        config_dir.mkdir(parents=True)
        (config_dir / "cfg.yaml").write_text("model: org/M\n")
        return profiles_yaml, config_dir

    def test_rename_moves_file_and_repoints_explicit_reference(self) -> None:
        from tui.common import config_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml, config_dir = self._setup(
                root, "- name: p\n  backend: vllm\n  config_name: cfg\n"
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ), patch("tui.common.config_store.config_dir", lambda b: config_dir):
                updated = config_store.rename_config("vllm", "cfg", "cfg2")

                self.assertEqual(updated, ["p"])
                self.assertFalse((config_dir / "cfg.yaml").exists())
                self.assertTrue((config_dir / "cfg2.yaml").exists())
                self.assertEqual(
                    profile_store.load_profile("p", "vllm").config_name, "cfg2"
                )

    def test_rename_repoints_profile_that_referenced_config_implicitly(self) -> None:
        # A profile named `cfg` with no config_name resolves to config `cfg`;
        # renaming the file must carry that profile along too.
        from tui.common import config_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml, config_dir = self._setup(
                root, "- name: cfg\n  backend: vllm\n"
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ), patch("tui.common.config_store.config_dir", lambda b: config_dir):
                updated = config_store.rename_config("vllm", "cfg", "cfg2")

                self.assertEqual(updated, ["cfg"])
                self.assertEqual(
                    profile_store.load_profile("cfg", "vllm").config_name, "cfg2"
                )

    def test_rename_rejects_collision_and_leaves_files_intact(self) -> None:
        from tui.common import config_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml, config_dir = self._setup(root, "- name: p\n  backend: vllm\n")
            (config_dir / "taken.yaml").write_text("model: org/Other\n")
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.config_store.config_dir", lambda b: config_dir
            ):
                with self.assertRaises(ValueError):
                    config_store.rename_config("vllm", "cfg", "taken")
                self.assertTrue((config_dir / "cfg.yaml").exists())
                self.assertEqual(
                    (config_dir / "taken.yaml").read_text(), "model: org/Other\n"
                )

    def test_rename_rejects_example_template_and_bad_names(self) -> None:
        from tui.common import config_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml, config_dir = self._setup(root, "- name: p\n  backend: vllm\n")
            (config_dir / "example.yaml").write_text("model: org/E\n")
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.config_store.config_dir", lambda b: config_dir
            ):
                with self.assertRaises(ValueError):
                    config_store.rename_config("vllm", "example", "other")
                with self.assertRaises(ValueError):
                    config_store.rename_config("vllm", "cfg", "example")
                with self.assertRaises(ValueError):
                    config_store.rename_config("vllm", "cfg", "../escape")
                with self.assertRaises(ValueError):
                    config_store.rename_config("vllm", "cfg", "UPPER")
                self.assertTrue((config_dir / "cfg.yaml").exists())

    def test_rename_rolls_back_file_when_batch_write_fails(self) -> None:
        from tui.common import config_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml, config_dir = self._setup(
                root, "- name: p\n  backend: vllm\n  config_name: cfg\n"
            )
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ), patch("tui.common.config_store.config_dir", lambda b: config_dir), patch(
                "tui.common.profile_store._replace_files",
                side_effect=OSError("profile write failed"),
            ):
                with self.assertRaisesRegex(OSError, "profile write failed"):
                    config_store.rename_config("vllm", "cfg", "cfg2")

            self.assertTrue((config_dir / "cfg.yaml").exists())
            self.assertFalse((config_dir / "cfg2.yaml").exists())
            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml):
                self.assertEqual(
                    profile_store.load_profile("p", "vllm").config_name, "cfg"
                )


@pytest.mark.usefixtures("isolated_vllm_storage")
class DeleteProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._profile_name = "test-del-profile"
        self._config_name = "test-del-shared-config"
        self._profile_path = backend.RUNTIME_DIR / f"{self._profile_name}.env"
        self._config_path = backend.CONFIG_DIR / f"{self._config_name}.yaml"
        self._other_profile_path = backend.RUNTIME_DIR / "test-del-other.env"

    def tearDown(self) -> None:
        backend.delete_profile(self._profile_name)
        backend.delete_profile("test-del-other")
        self._config_path.unlink(missing_ok=True)

    def test_delete_removes_profile_file(self) -> None:
        backend.save_profile(
            backend.Profile(name=self._profile_name, container_name="x", port="9000")
        )
        self.assertTrue(self._profile_path.exists())

        backend.delete_profile(self._profile_name)
        self.assertFalse(self._profile_path.exists())

    def test_delete_with_delete_config_removes_orphan_config(self) -> None:
        backend.save_config(backend.Config(name=self._config_name, model="org/m"))
        backend.save_profile(
            backend.Profile(
                name=self._profile_name,
                container_name="x",
                port="9000",
                config_name=self._config_name,
            )
        )

        backend.delete_profile(self._profile_name, delete_config=True)
        self.assertFalse(self._profile_path.exists())
        self.assertFalse(self._config_path.exists())

    def test_delete_with_delete_config_keeps_shared_config(self) -> None:
        backend.save_config(backend.Config(name=self._config_name, model="org/m"))
        backend.save_profile(
            backend.Profile(
                name=self._profile_name,
                container_name="x",
                port="9000",
                config_name=self._config_name,
            )
        )
        backend.save_profile(
            backend.Profile(
                name="test-del-other",
                container_name="y",
                port="9001",
                config_name=self._config_name,
            )
        )

        backend.delete_profile(self._profile_name, delete_config=True)
        self.assertFalse(self._profile_path.exists())
        self.assertTrue(
            self._config_path.exists(),
            "config must remain because another profile still references it",
        )


class BuildLoraOptionsTests(unittest.TestCase):
    def test_returns_empty_when_lora_disabled(self) -> None:
        profile = backend.Profile(name="p", enable_lora="false", max_loras="4")
        self.assertEqual(_build_lora_options(profile), "")

    def test_builds_enable_only_without_optional_fields(self) -> None:
        profile = backend.Profile(name="p", enable_lora="true")
        self.assertEqual(_build_lora_options(profile), "--enable-lora")

    def test_includes_loras_and_rank(self) -> None:
        profile = backend.Profile(
            name="p",
            enable_lora="true",
            max_loras="4",
            max_lora_rank="32",
        )
        result = _build_lora_options(profile)
        self.assertIn("--enable-lora", result)
        self.assertIn("--max-loras 4", result)
        self.assertIn("--max-lora-rank 32", result)

    def test_converts_lora_modules_comma_to_space(self) -> None:
        profile = backend.Profile(
            name="p",
            enable_lora="true",
            lora_modules="alpha=/path/a,beta=/path/b",
        )
        result = _build_lora_options(profile)
        self.assertIn("--lora-modules alpha=/path/a beta=/path/b", result)


class EnsureCommonEnvTests(unittest.TestCase):
    def test_missing_common_env_returns_error(self) -> None:
        profile = backend.Profile(name="p")
        with patch("tui.backends.vllm.backend_runtime.COMMON_ENV", Path("/nonexistent/.env.common")):
            ok, messages = _ensure_common_env(profile)
        self.assertFalse(ok)
        self.assertTrue(any(".env.common" in m for m in messages))

    def test_missing_hf_cache_path_returns_error(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as tmp:
            tmp.write("HF_CACHE_PATH=\n")
            tmp_path = Path(tmp.name)
        self.addCleanup(tmp_path.unlink, missing_ok=True)

        profile = backend.Profile(name="p")
        with patch("tui.backends.vllm.backend_runtime.COMMON_ENV", tmp_path):
            ok, messages = _ensure_common_env(profile)
        self.assertFalse(ok)
        self.assertTrue(any("HF_CACHE_PATH" in m for m in messages))

    def test_relative_hf_cache_path_rejected(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as tmp:
            tmp.write("HF_CACHE_PATH=relative/path\n")
            tmp_path = Path(tmp.name)
        self.addCleanup(tmp_path.unlink, missing_ok=True)

        profile = backend.Profile(name="p")
        with patch("tui.backends.vllm.backend_runtime.COMMON_ENV", tmp_path):
            ok, messages = _ensure_common_env(profile)
        self.assertFalse(ok)
        self.assertTrue(any("absolute" in m for m in messages))

    def test_valid_absolute_path_succeeds(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as tmp:
            tmp.write(
                "HF_CACHE_PATH=/abs/cache\n"
                "PREPARE_DOWNLOADER_IMAGE=vllm/vllm-openai:v0.27.1\n"
                "VLLM_USE_V2_MODEL_RUNNER=1\n"
            )
            tmp_path = Path(tmp.name)
        self.addCleanup(tmp_path.unlink, missing_ok=True)

        profile = backend.Profile(name="p")
        with patch("tui.backends.vllm.backend_runtime.COMMON_ENV", tmp_path):
            ok, messages = _ensure_common_env(profile)
        self.assertTrue(ok)
        self.assertEqual(messages, [])

    def test_unset_v2_model_runner_warns_but_succeeds(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as tmp:
            tmp.write(
                "HF_CACHE_PATH=/abs/cache\n"
                "PREPARE_DOWNLOADER_IMAGE=vllm/vllm-openai:v0.27.1\n"
            )
            tmp_path = Path(tmp.name)
        self.addCleanup(tmp_path.unlink, missing_ok=True)

        profile = backend.Profile(name="p")
        with patch("tui.backends.vllm.backend_runtime.COMMON_ENV", tmp_path):
            ok, messages = _ensure_common_env(profile)
        self.assertTrue(ok)
        self.assertTrue(any("VLLM_USE_V2_MODEL_RUNNER" in m for m in messages))

    def test_lora_requires_lora_base_path(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as tmp:
            tmp.write(
                "HF_CACHE_PATH=/abs/cache\n"
                "PREPARE_DOWNLOADER_IMAGE=vllm/vllm-openai:v0.27.1\n"
            )
            tmp_path = Path(tmp.name)
        self.addCleanup(tmp_path.unlink, missing_ok=True)

        profile = backend.Profile(name="p", enable_lora="true")
        with patch("tui.backends.vllm.backend_runtime.COMMON_ENV", tmp_path):
            ok, messages = _ensure_common_env(profile)
        self.assertFalse(ok)
        self.assertTrue(any("LORA_BASE_PATH" in m for m in messages))


class DetectGpuArchTests(unittest.IsolatedAsyncioTestCase):
    # `_detect_gpu_arch` delegates to dev_build.detect_local_gpu_caps(), which
    # shells out via dev_build._run() — patch that, not the runtime helper.
    async def test_single_gpu_keeps_dot_form(self) -> None:
        async def fake_run(*args, **kwargs):
            return 0, "8.9\n"

        with patch("tui.common.dev_build._run", fake_run):
            result = await _detect_gpu_arch()
        self.assertEqual(result, "8.9")

    async def test_multi_gpu_mixed_capabilities_deduped(self) -> None:
        async def fake_run(*args, **kwargs):
            return 0, "8.9\n8.6\n8.9\n"

        with patch("tui.common.dev_build._run", fake_run):
            result = await _detect_gpu_arch()
        self.assertEqual(result, "8.6 8.9")

    async def test_failure_returns_empty(self) -> None:
        async def fake_run(*args, **kwargs):
            return 1, ""

        with patch("tui.common.dev_build._run", fake_run):
            result = await _detect_gpu_arch()
        self.assertEqual(result, "")


class ForceLocalArchForDeepEPTests(unittest.TestCase):
    def _write_tmp(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".Dockerfile", delete=False)
        tmp.write(content)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        return Path(tmp.name)

    def test_rewrites_hardcoded_deepep_arch_line(self) -> None:
        dockerfile = self._write_tmp(
            "RUN --mount=type=cache,target=/root/.cache/uv \\\n"
            "    mkdir -p /tmp/ep_kernels_workspace/dist && \\\n"
            "    export TORCH_CUDA_ARCH_LIST='9.0a 10.0a' && \\\n"
            "    /tmp/install_python_libraries.sh \\\n"
            "        --workspace /tmp/ep_kernels_workspace\n"
        )

        ok, message = _force_local_arch_for_deepep(dockerfile)

        self.assertTrue(ok)
        self.assertIn("Patched DeepEP stage", message)
        patched = dockerfile.read_text()
        self.assertIn('export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" && \\', patched)
        self.assertNotIn("9.0a 10.0a", patched)

    def test_accepts_already_dynamic_arch_line(self) -> None:
        dockerfile = self._write_tmp(
            "RUN --mount=type=cache,target=/root/.cache/uv \\\n"
            "    export TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST}\" && \\\n"
            "    /tmp/install_python_libraries.sh --workspace /tmp/ep_kernels_workspace\n"
        )

        ok, message = _force_local_arch_for_deepep(dockerfile)

        self.assertTrue(ok)
        self.assertIn("already respects local TORCH_CUDA_ARCH_LIST", message)

    def test_fails_when_deepep_export_line_missing(self) -> None:
        dockerfile = self._write_tmp(
            "RUN --mount=type=cache,target=/root/.cache/uv \\\n"
            "    mkdir -p /tmp/ep_kernels_workspace/dist && \\\n"
            "    /tmp/install_python_libraries.sh --workspace /tmp/ep_kernels_workspace\n"
        )

        ok, message = _force_local_arch_for_deepep(dockerfile)

        self.assertFalse(ok)
        self.assertIn("could not locate DeepEP arch export line", message)

    def test_ignores_copy_line_and_patches_run_step(self) -> None:
        dockerfile = self._write_tmp(
            "COPY tools/ep_kernels/install_python_libraries.sh /tmp/install_python_libraries.sh\n"
            "RUN --mount=type=cache,target=/root/.cache/uv \\\n"
            "    export TORCH_CUDA_ARCH_LIST='9.0a 10.0a' && \\\n"
            "    /tmp/install_python_libraries.sh --workspace /tmp/ep_kernels_workspace\n"
        )

        ok, message = _force_local_arch_for_deepep(dockerfile)

        self.assertTrue(ok)
        self.assertIn("Patched DeepEP stage", message)


class PickPreferredTagTests(unittest.TestCase):
    def test_prefers_highest_stable_version(self) -> None:
        self.assertEqual(_pick_preferred_tag(["v0.6.0", "v0.8.2", "v0.7.3"]), "v0.8.2")

    def test_returns_versioned_over_latest(self) -> None:
        self.assertEqual(_pick_preferred_tag(["v0.8.2", "latest"]), "v0.8.2")

    def test_ignores_latest_alone(self) -> None:
        self.assertIsNone(_pick_preferred_tag(["latest", "random-tag"]))

    def test_ignores_nightly_alone(self) -> None:
        self.assertIsNone(_pick_preferred_tag(["nightly", "random-tag"]))

    def test_returns_none_when_no_versioned_tag(self) -> None:
        self.assertIsNone(_pick_preferred_tag(["zeta", "alpha", "mu"]))

    def test_returns_none_for_empty(self) -> None:
        self.assertIsNone(_pick_preferred_tag([]))


class DockerHubTagLookupTests(unittest.IsolatedAsyncioTestCase):
    async def test_release_version_scans_next_page_when_first_page_has_no_stable(self) -> None:
        page1 = {
            "results": [{"name": "nightly"}, {"name": "latest"}],
            "next": "https://hub.docker.com/page2",
        }
        page2 = {
            "results": [{"name": "v0.19.0"}, {"name": "v0.19.1"}],
            "next": None,
        }
        fetch = AsyncMock(side_effect=[page1, page2])

        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            version = await get_dockerhub_release_version()

        self.assertEqual(version, "v0.19.1")
        self.assertEqual(fetch.await_count, 2)

    async def test_release_version_returns_unknown_when_fetch_fails(self) -> None:
        fetch = AsyncMock(return_value=None)
        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            version = await get_dockerhub_release_version()
        self.assertEqual(version, "unknown")

    async def test_release_version_falls_back_to_registry_domain(self) -> None:
        fetch = AsyncMock(
            side_effect=[
                None,
                {
                    "results": [{"name": "v0.20.0"}],
                    "next": None,
                },
            ]
        )
        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            version = await get_dockerhub_release_version()
        self.assertEqual(version, "v0.20.0")
        self.assertEqual(fetch.await_count, 2)

    async def test_release_version_falls_back_to_docker_registry_tags(self) -> None:
        fetch = AsyncMock(
            side_effect=[
                None,
                None,
                None,
                None,
                None,
                None,
                {"token": "token"},
                {"tags": ["latest", "nightly", "v0.19.0", "v0.20.1"]},
            ]
        )
        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            version = await get_dockerhub_release_version()
        self.assertEqual(version, "v0.20.1")

    async def test_nightly_date_parses_last_updated(self) -> None:
        fetch = AsyncMock(return_value={"last_updated": "2026-04-23T12:34:56.000000Z"})
        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            nightly_date = await get_dockerhub_nightly_date()
        self.assertEqual(nightly_date, "2026-04-23")

    async def test_nightly_date_falls_back_to_registry_domain(self) -> None:
        fetch = AsyncMock(
            side_effect=[
                None,
                {"last_updated": "2026-04-24T01:02:03.000000Z"},
            ]
        )
        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            nightly_date = await get_dockerhub_nightly_date()
        self.assertEqual(nightly_date, "2026-04-24")
        self.assertEqual(fetch.await_count, 2)

    async def test_nightly_date_returns_available_from_docker_registry_tags(self) -> None:
        fetch = AsyncMock(
            side_effect=[
                None,
                None,
                None,
                None,
                None,
                None,
                {"token": "token"},
                {"tags": ["nightly", "v0.20.1"]},
            ]
        )
        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            nightly_date = await get_dockerhub_nightly_date()
        self.assertEqual(nightly_date, "available")


class SslContextBuilderTests(unittest.TestCase):
    def test_picks_existing_cafile_from_candidates(self) -> None:
        import certifi

        cafile = Path(certifi.where())

        with patch.object(ssl_ctx, "_cached", None), patch.object(
            ssl_ctx, "_CA_CANDIDATES", (str(cafile),)
        ), patch.dict("sys.modules", {"certifi": None}):
            ctx = ssl_ctx.get_ssl_context()

            self.assertIsNotNone(ctx)
            self.assertIs(ssl_ctx.get_ssl_context(), ctx)

    def test_falls_back_to_default_when_no_candidate_exists(self) -> None:
        default_context = Mock()
        default_context.get_ca_certs.return_value = [{"subject": "system"}]
        with patch.object(ssl_ctx, "_cached", None), patch.object(
            ssl_ctx, "_CA_CANDIDATES", ("/nonexistent/ca.pem",)
        ), patch.dict("sys.modules", {"certifi": None}), patch.object(
            ssl_ctx.ssl, "create_default_context", return_value=default_context
        ):
            ctx = ssl_ctx.get_ssl_context()

        self.assertIs(ctx, default_context)
        default_context.load_verify_locations.assert_not_called()


class CheckPortConflictTests(unittest.IsolatedAsyncioTestCase):
    async def test_check_port_conflict_returns_readable_profile_reference(self) -> None:
        profile = backend.Profile(name="current", container_name="current", port="8000")
        other = backend.Profile(name="other", container_name="other-container", port="8000")

        globals_dict = backend.check_port_conflict.__globals__

        async def fake_run_command(*args, **kwargs):
            return 0, "other-container\t127.0.0.1:8000->8000/tcp\n"

        def fake_load_profile(name: str):
            return {"current": profile, "other": other}[name]

        with patch.dict(
            globals_dict,
            {
                "run_command": fake_run_command,
                "list_profile_names": lambda: ["current", "other"],
                "load_profile": fake_load_profile,
            },
        ):
            conflict = await backend.check_port_conflict(profile)

        self.assertEqual(conflict, "profile 'other'")

    async def test_check_port_conflict_returns_none_when_profiles_stopped(self) -> None:
        """Static profile-to-profile port overlap with no running container must not conflict."""
        profile = backend.Profile(name="current", container_name="current", port="18999")
        other = backend.Profile(name="other", container_name="other", port="18999")

        async def fake_run_command(*args, **kwargs):
            return 0, ""

        with patch.dict(
            backend.check_port_conflict.__globals__,
            {
                "run_command": fake_run_command,
                "list_profile_names": lambda: ["current", "other"],
                "load_profile": lambda n: {"current": profile, "other": other}[n],
            },
        ):
            conflict = await backend.check_port_conflict(profile)

        self.assertIsNone(conflict)

    async def test_own_running_container_on_the_port_is_not_a_conflict(self) -> None:
        # A re-`up` of an already-running container is a compose no-op — and is
        # exactly what _post_start_validation tells the user to do. The bind
        # probe cannot distinguish our own docker-proxy from a foreign listener,
        # so it used to report "another local process" and block the start.
        #
        # Bind a REAL listener on the port: if the guard is missing, the probe
        # is reached and fails, so this test genuinely exercises the fix.
        import socket as socket_mod

        listener = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
        listener.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = str(listener.getsockname()[1])

        try:
            profile = backend.Profile(name="current", container_name="current", port=port)

            async def fake_run_command(*args, **kwargs):
                return 0, f"current\t127.0.0.1:{port}->8000/tcp\n"

            with patch.dict(
                backend.check_port_conflict.__globals__,
                {
                    "run_command": fake_run_command,
                    "list_profile_names": lambda: ["current"],
                    "load_profile": lambda n: profile,
                },
            ):
                conflict = await backend.check_port_conflict(profile)

            self.assertIsNone(conflict)
        finally:
            listener.close()

    async def test_own_container_running_on_a_different_port_still_probes(self) -> None:
        # The guard must stay narrow: our container being up on some *other*
        # port says nothing about whether this port is free.
        import socket as socket_mod

        listener = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
        listener.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = str(listener.getsockname()[1])

        try:
            profile = backend.Profile(name="current", container_name="current", port=port)

            async def fake_run_command(*args, **kwargs):
                # Same container, but published on an unrelated host port.
                return 0, "current\t127.0.0.1:19999->8000/tcp\n"

            with patch.dict(
                backend.check_port_conflict.__globals__,
                {
                    "run_command": fake_run_command,
                    "list_profile_names": lambda: ["current"],
                    "load_profile": lambda n: profile,
                },
            ):
                conflict = await backend.check_port_conflict(profile)

            self.assertIsNotNone(conflict)
            self.assertIn("another local process", conflict)
        finally:
            listener.close()

    async def test_llamacpp_own_running_container_on_the_port_is_not_a_conflict(self) -> None:
        # Parity: the llama.cpp runtime carries the identical guard.
        import socket as socket_mod

        listener = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
        listener.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        try:
            profile = lbackend.Profile(name="current", container_name="current", port=port)

            async def fake_docker_run(*args, **kwargs):
                return 0, f"current\t127.0.0.1:{port}->8080/tcp\n"

            with patch.dict(
                lbackend_rt.check_port_conflict.__globals__,
                {
                    "_docker_run": fake_docker_run,
                    "list_profile_names": lambda: ["current"],
                    "load_profile": lambda n: profile,
                },
            ):
                conflict = await lbackend_rt.check_port_conflict(profile)

            self.assertIsNone(conflict)
        finally:
            listener.close()

    async def test_check_port_conflict_sets_so_reuseaddr(self) -> None:
        """Regression: the fallback bind() check must set SO_REUSEADDR.

        Our own _post_start_validation hits /v1/models on this port after
        every successful `up`, which leaves a TIME-WAIT entry on
        127.0.0.1:<port> for ~60s. Without SO_REUSEADDR a plain bind()
        refuses TIME-WAIT ports and every up→down→up cycle within that
        window would falsely report 'another local process'."""
        import socket as socket_mod
        from unittest import mock

        profile = backend.Profile(name="current", container_name="current", port="18999")

        async def fake_run_command(*args, **kwargs):
            return 0, ""

        mock_sock = mock.MagicMock()

        with mock.patch("socket.socket", return_value=mock_sock), patch.dict(
            backend.check_port_conflict.__globals__,
            {
                "run_command": fake_run_command,
                "list_profile_names": lambda: ["current"],
                "load_profile": lambda n: profile,
            },
        ):
            await backend.check_port_conflict(profile)

        mock_sock.setsockopt.assert_any_call(
            socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1
        )

    async def test_check_port_conflict_detects_external_process(self) -> None:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        bound_port = sock.getsockname()[1]
        self.addCleanup(sock.close)

        profile = backend.Profile(name="current", container_name="current", port=str(bound_port))

        async def fake_run_command(*args, **kwargs):
            return 0, ""

        with patch.dict(
            backend.check_port_conflict.__globals__,
            {
                "run_command": fake_run_command,
                "list_profile_names": lambda: ["current"],
                "load_profile": lambda n: profile,
            },
        ):
            conflict = await backend.check_port_conflict(profile)

        self.assertIsNotNone(conflict)
        self.assertIn(str(bound_port), conflict)


class _ExistingStore:
    """profile_store stand-in for stream_container_up tests: the profile exists
    (so the "profile not found" guard passes) and rendering is a no-op."""

    @staticmethod
    def load_profile(name, backend_name):
        return profile_store.StoredProfile(name=name, backend=backend_name)

    @staticmethod
    def render_env_for_profile(_name, _backend):
        return None

    @staticmethod
    @contextmanager
    def storage_transaction():
        yield

    @staticmethod
    def render_env(_stored):
        return None


class SystemCommandFailureTests(unittest.TestCase):
    def test_mem_estimate_failure_without_gpus_exits_nonzero(self) -> None:
        import typer

        from tui.cli import system
        from tui.common import docker, mem

        with patch.object(mem, "estimate_model_memory", AsyncMock(return_value="failed")), \
             patch.object(docker, "get_gpu_info", AsyncMock(return_value=[])):
            with self.assertRaises(typer.Exit) as ctx:
                system.mem_estimate("org/model", json_out=False)

        self.assertEqual(ctx.exception.exit_code, 1)


class UpdateCommandRegressionTests(unittest.TestCase):
    def test_root_skips_startup_check_for_explicit_update(self) -> None:
        from tui.cli import _root

        ctx = SimpleNamespace(invoked_subcommand="update")
        with patch("tui.cli._configure_logging"), \
             patch("tui.cli._maybe_run_onboarding") as onboarding, \
             patch("tui.cli._maybe_check_for_update") as startup_check:
            _root(ctx, version=False, log_level="WARNING")

        onboarding.assert_not_called()
        startup_check.assert_not_called()

    def test_json_update_emits_one_valid_document(self) -> None:
        import contextlib
        import io

        from tui.cli import update_cmd
        from tui.common import version_check as vc

        status = vc.UpdateStatus(vc.BEHIND, tag="v9.9.9", local_version="1.0.0")
        output = io.StringIO()
        with patch.object(vc, "resolve_status", return_value=status), \
             patch.object(vc, "update_blocked_reason", return_value=""), \
             patch.object(vc, "apply_update", return_value=(True, "updated")), \
             contextlib.redirect_stdout(output):
            update_cmd(check=False, yes=True, json_out=True)

        payload = json.loads(output.getvalue())
        self.assertTrue(payload["update_ok"])
        self.assertEqual(payload["message"], "updated")


class ProbeFailureAndGuardTests(unittest.IsolatedAsyncioTestCase):
    """Failed probes and empty inputs must not read as definite answers."""

    async def test_container_exists_returns_none_when_probe_fails(self) -> None:
        from tui.backends.vllm import backend_runtime as rt

        async def failing_ps(*_a, **_k):
            return 1, ""  # docker ps failed / timed out

        with patch.object(rt, "run_command", failing_ps):
            self.assertIsNone(await rt._container_exists("c"))

    async def test_container_down_does_not_report_success_on_probe_failure(self) -> None:
        from tui.backends.vllm import backend_runtime as rt

        profile = backend.Profile(name="p", container_name="p", port="8000")

        async def failing_ps(*_a, **_k):
            return 1, ""

        with patch.object(rt.profile_store, "load_profile", return_value=SimpleNamespace()), \
             patch.object(rt, "load_profile", lambda _n: profile), \
             patch.object(rt, "run_command", failing_ps):
            rc, msg = await rt.container_down("p")

        self.assertNotEqual(rc, 0)
        self.assertIn("could not determine", msg)

    async def test_vllm_container_down_rejects_unknown_profile(self) -> None:
        from tui.backends.vllm import backend_runtime as rt

        with patch.object(rt.profile_store, "load_profile", return_value=None):
            rc, msg = await rt.container_down("missing")

        self.assertNotEqual(rc, 0)
        self.assertIn("not found", msg)

    async def test_cli_logs_missing_docker_returns_command_not_found(self) -> None:
        from tui.cli import _runtime

        with patch.object(
            _runtime.asyncio,
            "create_subprocess_exec",
            side_effect=FileNotFoundError("docker"),
        ):
            rc = await _runtime.docker_logs_once("p", tail=10)

        self.assertEqual(rc, 127)

    async def test_cli_follow_logs_missing_docker_returns_command_not_found(self) -> None:
        from tui.cli import _runtime

        with patch.object(
            _runtime.asyncio,
            "create_subprocess_exec",
            side_effect=FileNotFoundError("docker"),
        ):
            rc = await _runtime.docker_logs_follow("p", tail=10)

        self.assertEqual(rc, 127)

    async def test_vllm_network_cleanup_failure_is_reported(self) -> None:
        from tui.backends.vllm import backend_runtime as rt

        async def failed_remove(*_args, **_kwargs):
            return 1, "network has active endpoints"

        with patch.object(rt, "run_command", failed_remove):
            rc, msg = await rt._remove_compose_network("p")

        self.assertNotEqual(rc, 0)
        self.assertIn("active endpoints", msg)

    async def test_llamacpp_network_cleanup_failure_is_reported(self) -> None:
        from tui.backends.llamacpp import backend_runtime as rt

        async def failed_remove(*_args, **_kwargs):
            return 1, "network has active endpoints"

        with patch.object(rt, "_docker_run", failed_remove):
            rc, msg = await rt._remove_compose_network("p")

        self.assertNotEqual(rc, 0)
        self.assertIn("active endpoints", msg)

    async def test_disk_usage_command_failure_raises(self) -> None:
        from tui.common import docker

        with patch.object(docker, "run_command", AsyncMock(return_value=(1, "df failed"))):
            with self.assertRaisesRegex(RuntimeError, "df failed"):
                await docker.get_disk_usage("/missing")

    async def test_disk_usage_malformed_output_raises(self) -> None:
        from tui.common import docker

        with patch.object(docker, "run_command", AsyncMock(return_value=(0, "bad"))):
            with self.assertRaisesRegex(RuntimeError, "unexpected df output"):
                await docker.get_disk_usage("/tmp")

    async def test_llamacpp_container_exists_returns_none_when_probe_fails(self) -> None:
        from tui.backends.llamacpp import backend_runtime as rt

        async def failing_ps(*_a, **_k):
            return 1, ""

        with patch.object(rt, "_docker_run", failing_ps):
            self.assertIsNone(await rt._container_exists("c"))

    async def test_llamacpp_container_down_does_not_report_success_on_probe_failure(
        self,
    ) -> None:
        from tui.backends.llamacpp import backend as lb
        from tui.backends.llamacpp import backend_runtime as rt

        profile = lb.Profile(name="p", container_name="p", port=8080)
        stored = SimpleNamespace(name="p", backend="llamacpp")

        async def failing_ps(*_a, **_k):
            return 1, ""

        with patch.object(rt, "load_profile", lambda _n: profile), \
             patch.object(rt.profile_store, "load_profile", lambda _n, _b: stored), \
             patch.object(rt.profile_store, "render_env", lambda _s: None), \
             patch.object(rt, "_override_path", lambda _n: Path("/nonexistent.yaml")), \
             patch.object(rt, "_docker_run", failing_ps):
            rc, msg = await rt.container_down("p")

        self.assertNotEqual(rc, 0)
        self.assertIn("확인할 수 없음", msg)

    async def test_async_gpu_conflict_empty_gpu_set_is_silent(self) -> None:
        from tui.common import conflicts

        # An empty GPU set against a wildcard-GPU running container used to
        # produce a false "all GPUs" warning.
        running = backend.Profile(  # not used directly; we stub list_profiles
            name="other", container_name="other", port="8001",
        )

        async def fake_ps(*_a, **_k):
            return 0, "other\n"

        with patch.object(conflicts, "run_command", fake_ps), \
             patch("tui.common.profile_store.list_profiles",
                   lambda bk: [running] if bk == "vllm" else []):
            msgs = await conflicts.gpu_conflict_messages(
                profile_name="me", container_name="me",
                profile_gpu_id="", backend="vllm",
            )
        self.assertEqual(msgs, [])

    async def test_vllm_stream_up_rejects_unknown_profile(self) -> None:
        from tui.backends.vllm import backend_runtime as rt

        class _EmptyStore:
            @staticmethod
            def load_profile(_n, _b):
                return None

        with patch.object(rt, "profile_store", _EmptyStore):
            events = [e async for e in rt.stream_container_up("ghost")]

        self.assertIn(("rc", 1), events)
        self.assertTrue(any("프로필 없음" in d for k, d in events if k == "log"), events)

    async def test_llamacpp_stream_up_rejects_unknown_profile_before_env(self) -> None:
        from tui.backends.llamacpp import backend_runtime as rt

        with patch.object(rt.profile_store, "load_profile", return_value=None), \
            patch.object(rt, "validate_common_env", side_effect=AssertionError("env called")):
            events = [event async for event in rt.stream_container_up("ghost")]

        self.assertEqual(events[-1], ("rc", 1))
        self.assertTrue(any("프로필 없음" in data for kind, data in events if kind == "log"))

    async def test_llamacpp_prepare_rejects_unknown_profile_before_env(self) -> None:
        from tui.backends.llamacpp import backend_runtime as rt

        with patch.object(rt.profile_store, "load_profile", return_value=None), \
            patch.object(rt, "validate_common_env", side_effect=AssertionError("env called")):
            events = [event async for event in rt.stream_container_prepare("ghost")]

        self.assertEqual(events[-1], ("rc", 1))
        self.assertTrue(any("프로필 없음" in data for kind, data in events if kind == "log"))

    async def test_llamacpp_invalid_image_rejects_before_preflight(self) -> None:
        from tui.backends.llamacpp import backend_runtime as rt

        stored = SimpleNamespace(name="p")
        with patch.object(rt.profile_store, "load_profile", return_value=stored), \
            patch.object(rt, "validate_common_env", side_effect=AssertionError("env called")), \
            patch.object(rt, "load_profile", side_effect=AssertionError("profile loaded")):
            events = [
                event
                async for event in rt.stream_container_up(
                    "p", tag="ghcr.io/example/llama:latest"
                )
            ]

        self.assertEqual(events[-1], ("rc", 1))
        self.assertIn("reference", events[0][1])


class LlamacppDownloadedProbeTests(unittest.TestCase):
    def test_model_file_only_without_hf_repo_is_not_downloaded(self) -> None:
        # A model_file-only profile can't start (no hf_repo → render-override
        # fails, MODEL_DIR isn't mounted), so it must not read as ready even if
        # a matching file sits in ./models.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "m.gguf").write_bytes(b"x" * 10)
            stored = profile_store.StoredProfile(
                name="lp", backend="llamacpp", port=8080, model_file="m.gguf",
            )  # no hf_repo
            with patch.object(lbackend, "_get_model_dir", lambda: model_dir), \
                 patch.object(lbackend, "list_profile_names", lambda: ["lp"]), \
                 patch.object(lbackend, "load_profile", lambda _n: lbackend._to_profile(stored)), \
                 patch.object(lbackend, "find_cached_gguf", lambda *_a: None):
                profiles = lbackend.list_profiles(running=set())
        self.assertEqual(len(profiles), 1)
        self.assertFalse(profiles[0].downloaded)


class StreamContainerUpPortConflictTests(unittest.IsolatedAsyncioTestCase):
    async def test_vllm_invalid_one_off_image_stops_before_preflight(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")

        with patch.dict(
            backend.stream_container_up.__globals__,
            {
                "load_profile": lambda _: profile,
                "profile_store": _ExistingStore,
            },
        ):
            events = [
                event
                async for event in backend.stream_container_up(
                    "p", tag="custom/vllm"
                )
            ]

        self.assertEqual(events[-1], ("rc", 1))
        self.assertIn("image", events[0][1].lower())

    async def test_port_conflict_stops_before_preflight(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")

        async def fake_check_port_conflict(_profile):
            return "another process"

        globals_dict = backend.stream_container_up.__globals__
        with patch.dict(
            globals_dict,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": fake_check_port_conflict,
                "profile_store": _ExistingStore,
            },
        ):
            events = [event async for event in backend.stream_container_up("p")]

        self.assertEqual(events[-1], ("rc", 1))
        self.assertIn("Port 8000 is already in use", events[0][1])

    async def test_no_port_conflict_reaches_common_env_preflight(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")

        async def fake_check_port_conflict(_profile):
            return None

        globals_dict = backend.stream_container_up.__globals__
        with patch.dict(
            globals_dict,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": fake_check_port_conflict,
                "_ensure_common_env": lambda _profile: (False, ["common env missing"]),
                "profile_store": _ExistingStore,
            },
        ):
            events = [event async for event in backend.stream_container_up("p")]

        self.assertEqual(events, [("log", "common env missing"), ("rc", 1)])

    async def test_missing_local_image_resolves_and_pulls_stable_release(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")
        commands: list[list[str]] = []

        async def no_conflict(_profile):
            return None

        async def no_gpu_conflicts(_profile):
            return []

        async def fake_stream_command(cmd, **_kwargs):
            commands.append(cmd)
            yield ("rc", 1)

        globals_dict = backend.stream_container_up.__globals__
        with patch.dict(
            globals_dict,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": no_conflict,
                "_ensure_common_env": lambda _profile: (True, []),
                "_ensure_profile_config": lambda _profile: (True, []),
                "_gpu_conflict_messages": no_gpu_conflicts,
                "_compose_env": lambda *_args, **_kwargs: {},
                "get_local_latest_tag": AsyncMock(return_value="none"),
                "get_dockerhub_release_version": AsyncMock(return_value="v0.21.0"),
                "profile_store": _ExistingStore,
                "stream_command": fake_stream_command,
            },
        ):
            events = [event async for event in backend.stream_container_up("p")]

        logs = [data for kind, data in events if kind == "log"]
        self.assertIn("Using image: vllm/vllm-openai:v0.21.0", logs)
        self.assertEqual(len(commands), 1)
        self.assertIn("--pull", commands[0])
        self.assertEqual(commands[0][commands[0].index("--pull") + 1], "missing")

    async def test_dev_tag_is_sanitized_to_match_the_built_image(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")
        inspected: list[str] = []

        async def no_conflict(_p):
            return None

        async def fake_matches(image_tag, _repo, _branch):
            inspected.append(f"matches:{image_tag}")
            return True

        async def fake_gpu_conflicts(_p):
            return []

        async def fake_stream_command(cmd, **_kw):
            inspected.append("compose:" + " ".join(cmd))
            yield ("rc", 1)

        globals_dict = backend.stream_container_up.__globals__
        with patch.dict(
            globals_dict,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": no_conflict,
                "_dev_image_matches": fake_matches,
                "get_dev_build_defaults": lambda: ("https://x/y.git", "feat/foo"),
                "_ensure_common_env": lambda _p: (True, []),
                "_ensure_profile_config": lambda _p: (True, []),
                "_gpu_conflict_messages": fake_gpu_conflicts,
                "_compose_env": lambda *_a, **_k: {},
                "stream_command": fake_stream_command,
                "profile_store": _ExistingStore,
            },
        ), patch.object(
            globals_dict["dev_build"],
            "image_exists_locally",
            AsyncMock(return_value=True),
        ):
            events = [
                event
                async for event in backend.stream_container_up("p", use_dev=True)
            ]

        logs = [d for k, d in events if k == "log"]
        self.assertTrue(
            any("vllm-dev:feat-foo" in line for line in inspected + logs),
            (inspected, logs),
        )
        self.assertFalse(
            any("vllm-dev:feat/foo" in line for line in inspected + logs),
            (inspected, logs),
        )

    async def test_unrenderable_env_value_fails_this_profile_loudly(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")
        bad = profile_store.StoredProfile(
            name="p", backend="vllm", env_vars={"BAD": "it's"}
        )

        async def fake_check_port_conflict(_profile):
            return None

        class _FakeStore:
            StoredProfile = profile_store.StoredProfile

            @staticmethod
            def load_profile(_name, _bk):
                return bad

            @staticmethod
            def render_env_for_profile(_name, _backend):
                return profile_store.render_env(bad)  # raises ValueError

            @staticmethod
            @contextmanager
            def storage_transaction():
                yield

            @staticmethod
            def render_env(_stored):
                return profile_store.render_env(bad)

        globals_dict = backend.stream_container_up.__globals__
        with patch.dict(
            globals_dict,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": fake_check_port_conflict,
                "_ensure_common_env": lambda _p: (True, []),
                "_ensure_profile_config": lambda _p: (True, []),
                "profile_store": _FakeStore,
            },
        ):
            events = [
                event async for event in backend.stream_container_up("p")
            ]

        logs = [d for k, d in events if k == "log"]
        self.assertTrue(any("single quote" in line for line in logs), logs)
        self.assertIn(("rc", 1), events)

    async def test_use_default_image_drops_pinned_image_tag(self) -> None:
        profile = backend.Profile(
            name="p", container_name="p", port="8000", image_tag="vllm-dev:pinned"
        )

        async def fake_check_port_conflict(_profile):
            return None

        async def fake_gpu_conflicts(_p):
            return []

        async def fake_stream_command(cmd, **_kw):
            yield ("rc", 1)  # halt at compose; the image decision is already made

        globals_dict = backend.stream_container_up.__globals__
        with patch.dict(
            globals_dict,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": fake_check_port_conflict,
                "_ensure_common_env": lambda _p: (True, []),
                "_ensure_profile_config": lambda _p: (True, []),
                "_gpu_conflict_messages": fake_gpu_conflicts,
                "_compose_env": lambda *_a, **_k: {},
                "get_local_latest_tag": AsyncMock(return_value="v0.11.0"),
                "profile_store": _ExistingStore,
                "stream_command": fake_stream_command,
            },
        ):
            events = [
                event
                async for event in backend.stream_container_up(
                    "p", use_default_image=True
                )
            ]

        logs = [data for kind, data in events if kind == "log"]
        self.assertTrue(any("Default Image" in line for line in logs), logs)
        # The pinned-image branch must not have been taken.
        self.assertFalse(
            any("Using image: vllm-dev:pinned" in line for line in logs), logs
        )
        self.assertEqual(profile.image_tag, "")

    async def test_use_default_image_does_not_persist_the_cleared_pin(self) -> None:
        # _ensure_profile_config() calls save_profile() when it auto-links a
        # config. Clearing image_tag before that ran wrote image_tag="" into
        # profiles.yaml — a one-off override permanently destroying the pin.
        profile = backend.Profile(
            name="p",
            container_name="p",
            port="8000",
            config_name="",  # forces the auto-link save path
            image_tag="vllm-dev:pinned",
        )
        seen_tags: list[str] = []

        def fake_ensure_profile_config(p):
            # Stand in for the real helper's save_profile() call.
            seen_tags.append(p.image_tag)
            return (True, [])

        async def fake_check_port_conflict(_profile):
            return None

        async def fake_gpu_conflicts(_p):
            return []

        async def fake_stream_command(cmd, **_kw):
            yield ("rc", 1)

        globals_dict = backend.stream_container_up.__globals__
        with patch.dict(
            globals_dict,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": fake_check_port_conflict,
                "_ensure_common_env": lambda _p: (True, []),
                "_ensure_profile_config": fake_ensure_profile_config,
                "_gpu_conflict_messages": fake_gpu_conflicts,
                "_compose_env": lambda *_a, **_k: {},
                "get_local_latest_tag": AsyncMock(return_value="v0.11.0"),
                "profile_store": _ExistingStore,
                "stream_command": fake_stream_command,
            },
        ):
            _ = [
                event
                async for event in backend.stream_container_up(
                    "p", use_default_image=True
                )
            ]

        # Whatever gets persisted must still carry the user's pin.
        self.assertEqual(seen_tags, ["vllm-dev:pinned"])
        # ...and the in-memory clear still happened for the image decision.
        self.assertEqual(profile.image_tag, "")

    async def test_pinned_image_tag_is_honored_without_default_image(self) -> None:
        # Guard the other direction: absent --default-image the pin still wins.
        profile = backend.Profile(
            name="p", container_name="p", port="8000", image_tag="vllm-dev:pinned"
        )

        async def fake_check_port_conflict(_profile):
            return None

        globals_dict = backend.stream_container_up.__globals__
        with patch.dict(
            globals_dict,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": fake_check_port_conflict,
                "_ensure_common_env": lambda _profile: (False, ["stop here"]),
                "profile_store": _ExistingStore,
            },
        ):
            events = [event async for event in backend.stream_container_up("p")]

        logs = [data for kind, data in events if kind == "log"]
        self.assertFalse(any("Default Image" in line for line in logs), logs)
        self.assertEqual(profile.image_tag, "vllm-dev:pinned")

    async def test_full_vllm_image_reference_is_used_verbatim(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")
        compose_env_calls: list[dict] = []

        async def no_conflict(_profile):
            return None

        async def no_gpu_conflicts(_profile):
            return []

        async def stop_at_compose(_cmd, **_kwargs):
            yield ("rc", 1)

        def capture_env(*_args, **kwargs):
            compose_env_calls.append(kwargs)
            return {}

        with patch.dict(
            backend.stream_container_up.__globals__,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": no_conflict,
                "_ensure_common_env": lambda _profile: (True, []),
                "_ensure_profile_config": lambda _profile: (True, []),
                "_gpu_conflict_messages": no_gpu_conflicts,
                "_compose_env": capture_env,
                "profile_store": _ExistingStore,
                "stream_command": stop_at_compose,
            },
        ):
            events = [
                event
                async for event in backend.stream_container_up(
                    "p", tag="vllm/vllm-openai:v0.20.1"
                )
            ]

        logs = [data for kind, data in events if kind == "log"]
        self.assertIn("Using image: vllm/vllm-openai:v0.20.1", logs)
        self.assertEqual(
            compose_env_calls[-1].get("vllm_image"),
            "vllm/vllm-openai:v0.20.1",
        )

    async def test_registryless_tagged_vllm_reference_is_used_verbatim(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")
        compose_env_calls: list[dict] = []

        async def no_conflict(_profile):
            return None

        async def no_gpu_conflicts(_profile):
            return []

        async def stop_at_compose(_cmd, **_kwargs):
            yield "rc", 1

        with patch.dict(
            backend.stream_container_up.__globals__,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": no_conflict,
                "_ensure_common_env": lambda _profile: (True, []),
                "_ensure_profile_config": lambda _profile: (True, []),
                "_gpu_conflict_messages": no_gpu_conflicts,
                "_compose_env": lambda *_args, **kwargs: compose_env_calls.append(kwargs) or {},
                "profile_store": _ExistingStore,
                "stream_command": stop_at_compose,
            },
        ):
            events = [
                event
                async for event in backend.stream_container_up(
                    "p", tag="my-vllm:v1"
                )
            ]

        self.assertIn("Using image: my-vllm:v1", [data for kind, data in events if kind == "log"])
        self.assertEqual(compose_env_calls[-1].get("vllm_image"), "my-vllm:v1")

    async def test_pinned_vllm_version_is_verified_after_start(self) -> None:
        profile = backend.Profile(
            name="p",
            container_name="p",
            port="8000",
            image_tag="vllm/vllm-openai:v0.20.1",
        )
        verified: list[tuple[str, str]] = []
        commands: list[list[str]] = []

        async def no_conflict(_profile):
            return None

        async def no_gpu_conflicts(_profile):
            return []

        async def compose_ok(cmd, **_kwargs):
            commands.append(cmd)
            yield ("rc", 0)

        async def validation_ok(*_args, **_kwargs):
            yield ("result", True, [])

        async def capture_verify(container_name, expected_tag):
            verified.append((container_name, expected_tag))
            if False:
                yield ("log", "")

        with patch.dict(
            backend.stream_container_up.__globals__,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": no_conflict,
                "_ensure_common_env": lambda _profile: (True, []),
                "_ensure_profile_config": lambda _profile: (True, []),
                "_gpu_conflict_messages": no_gpu_conflicts,
                "_compose_env": lambda *_args, **_kwargs: {},
                "_post_start_validation": validation_ok,
                "_verify_vllm_version": capture_verify,
                "profile_store": _ExistingStore,
                "stream_command": compose_ok,
            },
        ):
            events = [
                event async for event in backend.stream_container_up("p", pull=True)
            ]

        self.assertEqual(events[-1], ("rc", 0))
        self.assertEqual(verified, [("p", "v0.20.1")])
        self.assertEqual(commands[0][-2:], ["--pull", "always"])


class LlamacppRuntimeImageTests(unittest.IsolatedAsyncioTestCase):
    def test_common_image_is_used_when_profile_has_no_pin(self) -> None:
        from tui.backends.llamacpp import backend_runtime as rt

        profile = lbackend.Profile(name="p", container_name="p", port=8080)
        with tempfile.TemporaryDirectory() as tmp:
            common = Path(tmp) / ".env.common"
            common.write_text("LLAMACPP_IMAGE=registry.example/llama:v1\n")
            with patch.object(rt, "COMMON_ENV", common):
                resolved = rt._resolve_runtime_image(profile)
        self.assertEqual(resolved, "registry.example/llama:v1")

    def test_default_selection_uses_common_image_instead_of_hardcoded_official(self) -> None:
        from tui.backends.llamacpp import backend_runtime as rt

        profile = lbackend.Profile(
            name="p",
            container_name="p",
            port=8080,
            image_tag="llamacpp-dev:pinned",
        )
        with tempfile.TemporaryDirectory() as tmp:
            common = Path(tmp) / ".env.common"
            common.write_text("LLAMACPP_IMAGE=registry.example/llama:v2\n")
            with patch.object(rt, "COMMON_ENV", common):
                resolved = rt._resolve_runtime_image(profile, use_default_image=True)
        self.assertEqual(resolved, "registry.example/llama:v2")

    async def test_fresh_host_up_allows_pull_for_non_dev_image(self) -> None:
        from tui.backends.llamacpp import backend_runtime as rt

        stored = profile_store.StoredProfile(
            name="p", backend="llamacpp", container_name="p", port=8080
        )
        profile = lbackend.Profile(name="p", container_name="p", port=8080)
        commands: list[list[str]] = []

        class Store:
            @staticmethod
            def load_profile(_name, _backend):
                return stored

            @staticmethod
            def render_env_for_profile(_name, _backend):
                return Path("p.env")

            @staticmethod
            @contextmanager
            def storage_transaction():
                yield

            @staticmethod
            def render_env(_stored):
                return Path("p.env")

        async def no_conflict(_profile):
            return None

        async def no_gpu_conflicts(_profile):
            return []

        async def render_ok(_name):
            return 0, ""

        async def stop_at_compose(cmd, **_kwargs):
            commands.append(cmd)
            yield ("rc", 1)

        with patch.dict(
            rt.stream_container_up.__globals__,
            {
                "validate_common_env": lambda _path: (True, []),
                "profile_store": Store,
                "load_profile": lambda _name: profile,
                "check_port_conflict": no_conflict,
                "_gpu_conflict_messages": no_gpu_conflicts,
                "_ensure_profile_config": lambda *_args: (True, []),
                "_resolve_runtime_image": lambda *_args, **_kwargs: rt.LLAMACPP_OFFICIAL_IMAGE,
                "_render_override": render_ok,
                "_compose_env": lambda _profile: {},
                "_compose_base_args": lambda _profile: ["docker", "compose"],
                "_stream": stop_at_compose,
            },
        ):
            events = [event async for event in rt.stream_container_up("p")]
            forced_events = [
                event async for event in rt.stream_container_up("p", pull=True)
            ]

        self.assertEqual(events[-1], ("rc", 1))
        self.assertEqual(forced_events[-1], ("rc", 1))
        self.assertEqual(commands[0][-2:], ["--pull", "missing"])
        self.assertEqual(commands[1][-2:], ["--pull", "always"])


class DynamicFlagImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_vllm_registryless_tagged_image_is_discovered_verbatim(self) -> None:
        from tui.backends.vllm import backend_inspect as inspect

        calls: list[tuple[str, ...]] = []

        async def fake_run(*args, **_kwargs):
            calls.append(args)
            return 0, '["max-model-len"]'

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(inspect, "_VLLM_PARAMS_CACHE_DIR", Path(tmp)), \
                patch.object(inspect, "run_command", fake_run), \
                patch("tui.common.docker.image_identity", AsyncMock(return_value="sha256:v1")):
                params = await inspect.extract_vllm_params("my-vllm:v1")

        self.assertEqual(params, {"max-model-len"})
        self.assertIn("my-vllm:v1", calls[0])

    async def test_vllm_configured_image_is_used_without_local_official_tag(self) -> None:
        from tui.backends.vllm import backend_inspect as inspect

        calls: list[tuple[str, ...]] = []

        async def fake_run(*args, **_kwargs):
            calls.append(args)
            return 0, '["max-model-len"]\n'

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(inspect, "_VLLM_PARAMS_CACHE_DIR", Path(tmp)), \
                patch.object(inspect, "_configured_vllm_image", return_value="custom/vllm:v1"), \
                patch.object(inspect, "get_local_latest_tag", AsyncMock(return_value="none")), \
                patch.object(inspect, "run_command", fake_run), \
                patch("tui.common.docker.image_identity", AsyncMock(return_value="sha256:v1")):
                params = await inspect.extract_vllm_params()

        self.assertEqual(params, {"max-model-len"})
        self.assertIn("custom/vllm:v1", calls[0])

    async def test_vllm_custom_images_use_separate_cache_entries(self) -> None:
        from tui.backends.vllm import backend_inspect as inspect

        calls: list[str] = []

        async def fake_run(*args, **_kwargs):
            image = args[5]
            calls.append(image)
            value = "flag-a" if image.endswith(":a") else "flag-b"
            return 0, json.dumps([value])

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(inspect, "_VLLM_PARAMS_CACHE_DIR", Path(tmp)), \
                patch.object(inspect, "run_command", fake_run), \
                patch(
                    "tui.common.docker.image_identity",
                    AsyncMock(side_effect=lambda ref: f"sha256:{ref.rsplit(':', 1)[-1]}"),
                ):
                first = await inspect.extract_vllm_params("custom/vllm:a")
                second = await inspect.extract_vllm_params("custom/vllm:b")
                cached = await inspect.extract_vllm_params("custom/vllm:a")

        self.assertEqual(first, {"flag-a"})
        self.assertEqual(second, {"flag-b"})
        self.assertEqual(cached, {"flag-a"})
        self.assertEqual(calls, ["custom/vllm:a", "custom/vllm:b"])

    async def test_llamacpp_flag_discovery_uses_explicit_profile_image(self) -> None:
        calls: list[tuple[str, ...]] = []

        class Process:
            returncode = 0

            async def communicate(self):
                return b"--ctx-size N\n--n-gpu-layers N\n", b""

        async def fake_create(*args, **_kwargs):
            calls.append(args)
            return Process()

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(lbackend, "CONFIG_DIR", Path(tmp)), \
                patch("asyncio.create_subprocess_exec", fake_create), \
                patch("tui.common.docker.image_identity", AsyncMock(return_value="sha256:v1")):
                flags = await lbackend.extract_llama_server_flags(
                    "custom/llama:v1"
                )

        self.assertEqual(flags, {"ctx-size", "n-gpu-layers"})
        self.assertIn("custom/llama:v1", calls[0])

    async def test_vllm_moving_tag_cache_changes_with_image_identity(self) -> None:
        from tui.backends.vllm import backend_inspect as inspect

        identity = "sha256:a"
        extracted: list[str] = []

        async def fake_identity(_ref):
            return identity

        async def fake_run(*_args, **_kwargs):
            value = "flag-a" if identity.endswith("a") else "flag-b"
            extracted.append(value)
            return 0, json.dumps([value])

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(inspect, "_VLLM_PARAMS_CACHE_DIR", Path(tmp)), \
                patch.object(inspect, "run_command", fake_run), \
                patch("tui.common.docker.image_identity", fake_identity):
                first = await inspect.extract_vllm_params("custom/vllm:nightly")
                identity = "sha256:b"
                second = await inspect.extract_vllm_params("custom/vllm:nightly")

        self.assertEqual(first, {"flag-a"})
        self.assertEqual(second, {"flag-b"})
        self.assertEqual(extracted, ["flag-a", "flag-b"])


async def _drain_validation(gen):
    """Run the _post_start_validation async generator to completion.

    Returns (ok, result_messages, log_lines) — the final ("result", ...)
    payload plus any ("log", ...) heartbeat lines emitted along the way.
    Raises AssertionError if the generator finishes without ever yielding a
    final ("result", ...) event, so every caller implicitly verifies that
    contract on both the success and failure paths.
    """
    ok: bool | None = None
    messages: list[str] = []
    logs: list[str] = []
    async for ev in gen:
        if ev[0] == "result":
            ok, messages = ev[1], ev[2]
        else:
            logs.append(ev[1])
    if ok is None:
        raise AssertionError(
            "_post_start_validation finished without yielding a ('result', ...) event"
        )
    return ok, messages, logs


class PostStartValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fails_when_container_cannot_be_inspected(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")

        async def fake_run_command(*args, **kwargs):
            return 1, "No such object: p"

        with patch.dict(
            _post_start_validation.__globals__,
            {"run_command": fake_run_command},
        ):
            ok, messages, _ = await _drain_validation(
                _post_start_validation(profile, timeout=0.1, poll_interval=0.05)
            )

        self.assertFalse(ok)
        self.assertTrue(any("could not inspect container" in m for m in messages))

    async def test_fails_when_container_exits_during_startup(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")

        async def fake_run_command(*args, **kwargs):
            if args[:2] == ("docker", "inspect"):
                return 0, "exited\tunhealthy"
            if args[:2] == ("docker", "logs"):
                return 0, "line-a\nline-b\n"
            return 0, ""

        with patch.dict(
            _post_start_validation.__globals__,
            {
                "run_command": fake_run_command,
                "_models_endpoint_ready": AsyncMock(return_value=False),
            },
        ):
            ok, messages, _ = await _drain_validation(
                _post_start_validation(profile, timeout=0.1, poll_interval=0.05)
            )

        self.assertFalse(ok)
        self.assertTrue(any("exited during startup" in m for m in messages))
        self.assertTrue(any("line-b" in m for m in messages))

    async def test_fails_when_models_endpoint_not_ready_before_timeout(self) -> None:
        # Timeout no longer reports success: a container that started but has
        # not finished loading the model is NOT ready, and chaining a
        # benchmark on it used to mask real failures. Validation now returns
        # (False, [...]) so callers stop the success cascade.
        profile = backend.Profile(name="p", container_name="p", port="8000")

        async def fake_run_command(*args, **kwargs):
            if args[:2] == ("docker", "inspect"):
                return 0, "running\tstarting"
            return 0, ""

        with patch.dict(
            _post_start_validation.__globals__,
            {
                "run_command": fake_run_command,
                "_models_endpoint_ready": AsyncMock(return_value=False),
            },
        ):
            ok, messages, _ = await _drain_validation(
                _post_start_validation(profile, timeout=0.1, poll_interval=0.05)
            )

        self.assertFalse(ok)
        self.assertTrue(any("/v1/models is not ready within timeout" in m for m in messages))

    async def test_download_keeps_waiting_past_flat_timeout(self) -> None:
        # A first-run multi-GB download grows the HF cache for longer than the
        # flat readiness budget. The stall deadline must reset on cache growth
        # so the container is not falsely reported "not ready" while the model
        # is still downloading inside the container.
        profile = backend.Profile(name="p", container_name="p", port="8000")

        state = {"du_calls": 0}

        async def fake_run_command(*args, **kwargs):
            if args[0] == "du":
                state["du_calls"] += 1
                # Cache grows for the first 10 probes, then plateaus.
                gb = min(state["du_calls"], 10)
                return 0, f"{gb * 1_000_000_000}\t/fake/cache\n"
            if args[:2] == ("docker", "inspect"):
                return 0, "running\tstarting"
            if args[:2] == ("docker", "logs"):
                # Log stays frozen — llama.cpp's downloader is silent in
                # `docker logs`, so cache growth is the only progress signal.
                return 0, "frozen-download-log\n"
            return 0, ""

        async def fake_ready(_port):
            # Endpoint comes up only once the cache has stopped growing.
            return state["du_calls"] > 10

        with patch.dict(
            _post_start_validation.__globals__,
            {"run_command": fake_run_command, "_models_endpoint_ready": fake_ready},
        ):
            ok, messages, _ = await _drain_validation(
                _post_start_validation(
                    profile,
                    timeout=0.1,
                    poll_interval=0.02,
                    hf_cache_path="/fake/cache",
                )
            )

        # Download wall-time (~9 polls x 0.02s ~= 0.18s) far exceeds the 0.1s
        # budget; a fixed deadline would have failed long before the endpoint
        # came up. The stall deadline resets on each cache-growth poll instead.
        self.assertTrue(ok, msg=messages)
        self.assertEqual(messages, [])

    async def test_capped_when_log_churns_without_readiness(self) -> None:
        # A container that stays "running" and keeps emitting fresh log lines
        # (e.g. an error-retry loop) would reset the stall deadline forever.
        # The absolute max_wait backstop must still end the wait.
        profile = backend.Profile(name="p", container_name="p", port="8000")

        state = {"n": 0}

        async def fake_run_command(*args, **kwargs):
            if args[:2] == ("docker", "inspect"):
                return 0, "running\tstarting"
            if args[:2] == ("docker", "logs"):
                state["n"] += 1
                return 0, f"ever-changing-log-line-{state['n']}\n"
            return 0, ""

        with patch.dict(
            _post_start_validation.__globals__,
            {
                "run_command": fake_run_command,
                "_models_endpoint_ready": AsyncMock(return_value=False),
            },
        ):
            ok, messages, _ = await _drain_validation(
                _post_start_validation(
                    profile, timeout=10.0, poll_interval=0.02, max_wait=0.1
                )
            )

        # The stall deadline (10s) never fires — the log churns every poll —
        # so only the max_wait backstop can end this.
        self.assertFalse(ok)
        self.assertTrue(any("still not ready after" in m for m in messages))


class LlamacppPostStartValidationTests(unittest.IsolatedAsyncioTestCase):
    """Parity guard: the llama.cpp _post_start_validation must behave like the
    vLLM mirror. It wraps docker calls in `_docker_run` (not `run_command`),
    so a copy-paste divergence would slip past the vLLM-only tests above.
    """

    async def test_download_keeps_waiting_past_flat_timeout(self) -> None:
        profile = lbackend.Profile(name="p", container_name="p", port=8000)

        state = {"du_calls": 0}

        async def fake_docker_run(*args, **kwargs):
            if args[0] == "du":
                state["du_calls"] += 1
                gb = min(state["du_calls"], 10)
                return 0, f"{gb * 1_000_000_000}\t/fake/cache\n"
            if args[:2] == ("docker", "inspect"):
                return 0, "running\tstarting"
            if args[:2] == ("docker", "logs"):
                return 0, "frozen-download-log\n"
            return 0, ""

        async def fake_ready(_port):
            return state["du_calls"] > 10

        with patch.dict(
            lbackend_rt._post_start_validation.__globals__,
            {"_docker_run": fake_docker_run, "_models_endpoint_ready": fake_ready},
        ):
            ok, messages, _ = await _drain_validation(
                lbackend_rt._post_start_validation(
                    profile,
                    timeout=0.1,
                    poll_interval=0.02,
                    hf_cache_path="/fake/cache",
                )
            )

        self.assertTrue(ok, msg=messages)
        self.assertEqual(messages, [])

    async def test_fails_when_models_endpoint_not_ready_before_timeout(self) -> None:
        profile = lbackend.Profile(name="p", container_name="p", port=8000)

        async def fake_docker_run(*args, **kwargs):
            if args[:2] == ("docker", "inspect"):
                return 0, "running\tstarting"
            return 0, ""

        with patch.dict(
            lbackend_rt._post_start_validation.__globals__,
            {
                "_docker_run": fake_docker_run,
                "_models_endpoint_ready": AsyncMock(return_value=False),
            },
        ):
            ok, messages, _ = await _drain_validation(
                lbackend_rt._post_start_validation(
                    profile, timeout=0.1, poll_interval=0.05
                )
            )

        self.assertFalse(ok)
        self.assertTrue(any("/v1/models is not ready within timeout" in m for m in messages))


class VllmRuntimeWarningTests(unittest.IsolatedAsyncioTestCase):
    async def test_version_verification_warns_when_exec_fails(self) -> None:
        async def fake_run_command(*args, **kwargs):
            return 1, "exec failed"

        with patch.dict(
            _verify_vllm_version.__globals__,
            {"run_command": fake_run_command},
        ):
            events = [event async for event in _verify_vllm_version("p", "v0.19.1")]

        self.assertEqual(events[0][0], "log")
        self.assertIn("could not verify vLLM version", events[0][1])


class BenchmarkValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_run_counts_are_rejected(self) -> None:
        from tui.common.http import run_bench

        with self.assertRaisesRegex(ValueError, "runs"):
            await run_bench(8000, "model", runs=0)
        with self.assertRaisesRegex(ValueError, "warmup"):
            await run_bench(8000, "model", warmup=-1)

    async def test_missing_completion_usage_is_rejected(self) -> None:
        from tui.common import http

        with patch.object(
            http,
            "chat_completion_bench",
            AsyncMock(return_value={"elapsed": 1.0, "usage": {}}),
        ):
            with self.assertRaisesRegex(RuntimeError, "completion_tokens"):
                await http.run_bench(8000, "model", runs=1, warmup=0)


class VllmGpuConflictTests(unittest.IsolatedAsyncioTestCase):
    async def test_gpu_conflicts_include_llamacpp_profiles(self) -> None:
        # vLLM and llama.cpp now both call the shared
        # tui.common.conflicts.gpu_conflict_messages, so this scenario is
        # patched at the common module instead of per-backend globals.
        from tui.common import conflicts as _conflicts

        vllm_profile = profile_store.StoredProfile(
            name="v",
            backend="vllm",
            container_name="v",
            gpu_id="0",
        )
        other = profile_store.StoredProfile(
            name="l",
            backend="llamacpp",
            container_name="l",
            gpu_id="0",
        )

        async def fake_run_command(*args, **kwargs):
            # The running container is "l" (the llama.cpp profile), and the
            # caller is the vLLM profile "v" — exactly the cross-backend
            # clash we care about reporting.
            return 0, "l\n"

        def fake_list_profiles(backend_name):
            return {"vllm": [vllm_profile], "llamacpp": [other]}[backend_name]

        with patch.dict(
            _conflicts.__dict__,
            {"run_command": fake_run_command},
        ), patch.object(profile_store, "list_profiles", side_effect=fake_list_profiles):
            messages = await _gpu_conflict_messages(
                backend.Profile(name="v", container_name="v", gpu_id="0")
            )

        self.assertTrue(any("llama.cpp" in m and "GPU 0" in m for m in messages))


class LlamacppValidationTests(unittest.TestCase):
    def test_validate_name_is_compose_safe(self) -> None:
        self.assertTrue(lbackend.validate_name("qwen3_0-6b"))
        self.assertFalse(lbackend.validate_name("Qwen3"))
        self.assertFalse(lbackend.validate_name("dot.name"))


class LlamacppCheckPortConflictTests(unittest.IsolatedAsyncioTestCase):
    async def test_check_port_conflict_sets_so_reuseaddr(self) -> None:
        """Regression: the fallback bind() check must set SO_REUSEADDR.

        Mirrors the vLLM regression at CheckPortConflictTests.
        Our own _post_start_validation hits /v1/models on this port after
        every successful `up`, which leaves a TIME-WAIT entry on
        127.0.0.1:<port> for ~60s. Without SO_REUSEADDR a plain bind()
        refuses TIME-WAIT ports and every up→down→up cycle within that
        window would falsely report 'another local process'."""
        import socket as socket_mod
        from unittest import mock

        profile = lbackend.Profile(name="current", container_name="current", port=18999)

        async def fake_docker_run(*args, **kwargs):
            return 0, ""

        mock_sock = mock.MagicMock()

        with mock.patch("socket.socket", return_value=mock_sock), patch.dict(
            lbackend_rt.check_port_conflict.__globals__,
            {
                "_docker_run": fake_docker_run,
                "list_profile_names": lambda: ["current"],
                "load_profile": lambda n: profile,
            },
        ):
            await lbackend_rt.check_port_conflict(profile)

        mock_sock.setsockopt.assert_any_call(
            socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1
        )

    async def test_check_port_conflict_detects_external_listener(self) -> None:
        """An actively LISTENING peer on the port must still be reported as
        a conflict even with SO_REUSEADDR on the probe — SO_REUSEADDR only
        skips TIME-WAIT, not active LISTEN."""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        bound_port = sock.getsockname()[1]
        self.addCleanup(sock.close)

        profile = lbackend.Profile(name="current", container_name="current", port=bound_port)

        async def fake_docker_run(*args, **kwargs):
            return 0, ""

        with patch.dict(
            lbackend_rt.check_port_conflict.__globals__,
            {
                "_docker_run": fake_docker_run,
                "list_profile_names": lambda: ["current"],
                "load_profile": lambda n: profile,
            },
        ):
            conflict = await lbackend_rt.check_port_conflict(profile)

        self.assertIsNotNone(conflict)
        self.assertIn(str(bound_port), conflict)


class LlamacppStreamContainerUpTests(unittest.IsolatedAsyncioTestCase):
    """Dev-tag rebuild policy and config auto-generation."""

    def _fakes(self, tmp: Path, *, config_name: str, exists: bool, matches: bool):
        from tui.common import profile_store as ps

        stored = ps.StoredProfile(
            name="p", backend="llamacpp", container_name="p", port=8080,
            config_name=config_name, hf_repo="o/r", hf_file="m.gguf",
        )
        profile = lbackend.Profile(
            name="p", container_name="p", port=8080,
            config_name=config_name, hf_repo="o/r", hf_file="m.gguf",
        )
        state = {
            "builds": [],
            "saved": [],
            "stored": stored,
            "profile": profile,
            "exists": exists,
            "matches": matches,
        }

        class FakePS:
            @staticmethod
            def load_profile(_name, _backend):
                return stored

            @staticmethod
            def render_env_for_profile(_name, _backend):
                return None

            @staticmethod
            @contextmanager
            def storage_transaction():
                yield

            @staticmethod
            def render_env(_stored):
                return None

            @staticmethod
            def save_profile(s):
                # Snapshot image_tag at save time — a transient override must
                # never be persisted.
                state["saved"].append(s.image_tag)

        async def fake_build(*a, **kw):
            state["builds"].append((a, kw))
            yield ("rc", 0)

        async def fake_render_override(_name):
            return (1, "<halt before compose>")

        async def no_conflict(_p):
            return None

        async def no_gpu(_p):
            return []

        state["patch"] = {
            "validate_common_env": lambda _p: (True, []),
            "load_profile": lambda _: profile,
            "check_port_conflict": no_conflict,
            "_gpu_conflict_messages": no_gpu,
            "profile_store": FakePS,
            "_stream_build_dev_image": fake_build,
            "_render_override": fake_render_override,
            "CONFIG_DIR": tmp,
        }
        return state

    async def test_explicit_dev_tag_with_mismatched_source_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "c.yaml").write_text("alias: c\n")
            state = self._fakes(tmp, config_name="c", exists=True, matches=False)

            g = lbackend_rt.stream_container_up.__globals__
            with patch.dict(g, state["patch"]), patch.object(
                lbackend, "CONFIG_DIR", tmp
            ), patch.object(
                lbackend_rt.dev_build,
                "image_exists_locally",
                AsyncMock(return_value=state["exists"]),
            ), patch.object(
                lbackend_rt.dev_build,
                "image_matches",
                AsyncMock(return_value=state["matches"]),
            ):
                _ = [
                    e
                    async for e in lbackend_rt.stream_container_up(
                        "p", use_dev=True, tag="mytag"
                    )
                ]

            self.assertEqual(len(state["builds"]), 1)
            args, kwargs = state["builds"][0]
            self.assertEqual(args, ("master",))
            self.assertEqual(kwargs["custom_tag"], "mytag")

    async def test_branch_derived_tag_still_rebuilds_on_label_mismatch(self) -> None:
        # The other direction: no explicit tag → the label check still guards
        # against reusing a same-named image built from a different source.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "c.yaml").write_text("alias: c\n")
            state = self._fakes(tmp, config_name="c", exists=True, matches=False)

            g = lbackend_rt.stream_container_up.__globals__
            with patch.dict(g, state["patch"]), patch.object(
                lbackend, "CONFIG_DIR", tmp
            ), patch.object(
                lbackend_rt.dev_build,
                "image_exists_locally",
                AsyncMock(return_value=state["exists"]),
            ), patch.object(
                lbackend_rt.dev_build,
                "image_matches",
                AsyncMock(return_value=state["matches"]),
            ):
                _ = [
                    e
                    async for e in lbackend_rt.stream_container_up(
                        "p", use_dev=True, branch="main"
                    )
                ]

            self.assertEqual(len(state["builds"]), 1)

    async def test_missing_config_is_auto_linked_and_created(self) -> None:
        # The Start screen promises "a default config will be generated on
        # start" — on llama.cpp that was a lie; render-override just failed.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = self._fakes(tmp, config_name="", exists=True, matches=True)

            g = lbackend_rt.stream_container_up.__globals__
            with patch.dict(g, state["patch"]), patch.object(
                lbackend, "CONFIG_DIR", tmp
            ):
                events = [
                    e async for e in lbackend_rt.stream_container_up("p")
                ]

            logs = [d for k, d in events if k == "log"]
            self.assertTrue(any("자동 링크" in line for line in logs), logs)
            self.assertTrue(any("기본 config 생성" in line for line in logs), logs)
            self.assertTrue((tmp / "p.yaml").exists(), list(tmp.iterdir()))
            self.assertEqual(state["stored"].config_name, "p")
            # The auto-link save must not carry a transient image override.
            self.assertEqual(state["saved"], [""])


class LlamacppRenderOverrideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "llamacpp" / "render-override.py"
        spec = importlib.util.spec_from_file_location("render_override_for_tests", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.module = module

    def test_hf_repo_and_file_emit_hf_download_flags(self) -> None:
        # render_command now emits container-side `-hf/-hff` download args,
        # not the legacy host-side `--model` path.
        command = self.module.render_command(
            {"ctx-size": 2048},
            hf_repo="org/repo",
            hf_file="model.gguf",
        )
        self.assertEqual(command[command.index("-hf") + 1], "org/repo")
        self.assertEqual(command[command.index("-hff") + 1], "model.gguf")
        self.assertNotIn("--model", command)

    def test_profile_model_file_is_used_when_config_omits_model_file(self) -> None:
        # When the config has no `model-file` and the profile carries no
        # `hf_file`, the profile's `model_file` is the resolved -hff filename.
        command = self.module.render_command(
            {"ctx-size": 2048},
            model_file="model.gguf",
            hf_repo="org/repo",
        )
        self.assertEqual(command[command.index("-hff") + 1], "model.gguf")

    def test_config_model_file_used_as_fallback(self) -> None:
        command = self.module.render_command(
            {"ctx-size": 2048, "model-file": "from-config.gguf"},
            hf_repo="org/repo",
        )
        self.assertEqual(command[command.index("-hff") + 1], "from-config.gguf")

    def test_missing_model_file_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            self.module.render_command({"ctx-size": 2048}, hf_repo="org/repo")

    def test_missing_hf_repo_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            self.module.render_command({"ctx-size": 2048}, hf_file="model.gguf")

    def test_scalar_override_tensors_is_one_flag_not_per_char(self) -> None:
        # A string value used to be iterated char by char → `-ot .`, `-ot *`, …
        command = self.module.render_command(
            {"override-tensors": ".*=CPU"},
            hf_repo="org/repo", hf_file="m.gguf",
        )
        idx = [i for i, a in enumerate(command) if a == "-ot"]
        self.assertEqual(len(idx), 1)
        self.assertEqual(command[idx[0] + 1], ".*=CPU")

    def test_scalar_extra_args_is_shlex_split(self) -> None:
        command = self.module.render_command(
            {"extra-args": "--foo bar --baz"},
            hf_repo="org/repo", hf_file="m.gguf",
        )
        # Split into shell words appended in order, not char-exploded.
        self.assertEqual(command[-3:], ["--foo", "bar", "--baz"])

    def test_list_override_tensors_still_expands_per_item(self) -> None:
        command = self.module.render_command(
            {"override-tensors": ["a=CPU", "b=GPU"]},
            hf_repo="org/repo", hf_file="m.gguf",
        )
        self.assertEqual(command.count("-ot"), 2)

    def test_flash_attn_true_renders_with_on_value(self) -> None:
        """Regression: modern llama-server requires --flash-attn on/off/auto,
        not a bare --flash-attn (it would consume the next arg as its value)."""
        command = self.module.render_command(
            {"flash-attn": True},
            hf_repo="org/repo",
            hf_file="model.gguf",
        )
        idx = command.index("--flash-attn")
        self.assertEqual(command[idx + 1], "on")

    def test_flash_attn_false_renders_with_off_value(self) -> None:
        command = self.module.render_command(
            {"flash-attn": False},
            hf_repo="org/repo",
            hf_file="model.gguf",
        )
        idx = command.index("--flash-attn")
        self.assertEqual(command[idx + 1], "off")

    def test_flash_attn_string_value_passes_through(self) -> None:
        """A user can also write `flash-attn: auto` (or any string) explicitly."""
        command = self.module.render_command(
            {"flash-attn": "auto"},
            hf_repo="org/repo",
            hf_file="model.gguf",
        )
        idx = command.index("--flash-attn")
        self.assertEqual(command[idx + 1], "auto")

    def test_dirs_derive_from_profile_store_root(self) -> None:
        # The script reads config/ and writes .runtime/ while profile_store
        # supplies the profile — both must resolve against the same root, or a
        # LLMUX_ROOT run renders an override from the wrong checkout's config.
        root = profile_store.PROJECT_ROOT
        self.assertEqual(self.module.ROOT, root)
        self.assertEqual(self.module.CONFIG_DIR, root / "config" / "llamacpp")
        self.assertEqual(self.module.RUNTIME_DIR, root / ".runtime" / "llamacpp")

    def test_llmux_root_env_moves_dirs_but_not_import_path(self) -> None:
        # Fresh interpreter: PROJECT_ROOT is resolved at profile_store import,
        # so LLMUX_ROOT has to be set before the script loads.
        script = Path(__file__).resolve().parents[1] / "scripts" / "llamacpp" / "render-override.py"
        with tempfile.TemporaryDirectory() as tmp:
            # LLMUX_ROOT must point at a checkout — compose/ lives there and the
            # resolver rejects a root without it.
            (Path(tmp) / "compose").mkdir()
            probe = (
                "import importlib.util\n"
                f"spec = importlib.util.spec_from_file_location('ro', {str(script)!r})\n"
                "m = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(m)\n"
                "print(m.ROOT)\nprint(m.CONFIG_DIR)\n"
            )
            out = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=str(Path(__file__).resolve().parents[1]),
                env={**os.environ, "LLMUX_ROOT": tmp},
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            lines = out.stdout.strip().splitlines()
            # Data dirs follow LLMUX_ROOT ...
            self.assertEqual(lines[0], str(Path(tmp).resolve()))
            self.assertEqual(lines[1], str(Path(tmp).resolve() / "config" / "llamacpp"))
        # ... while `tui` still imported from the real checkout (exec_module
        # would have raised ImportError otherwise, since tmp has no tui/).

    def test_other_bool_keys_remain_bare_flags(self) -> None:
        """The flash-attn special-case must NOT apply to other bool keys —
        --jinja, --cont-batching, --mlock etc. take no value and
        adding one would break startup. Whitelist must stay narrow."""
        command = self.module.render_command(
            {"jinja": True, "cont-batching": True, "mlock": False},
            hf_repo="org/repo",
            hf_file="model.gguf",
        )
        # Both `--jinja` and `--cont-batching` must appear as bare flags.
        self.assertIn("--jinja", command)
        self.assertIn("--cont-batching", command)
        # And neither may be followed by an "on"/"off"/"true"/"false" token.
        for flag in ("--jinja", "--cont-batching"):
            idx = command.index(flag)
            next_token = command[idx + 1] if idx + 1 < len(command) else ""
            self.assertNotIn(next_token, {"on", "off", "true", "false", "True", "False"})
        # False-valued bare bools are simply omitted.
        self.assertNotIn("--mlock", command)

    def test_host_and_port_from_config_are_dropped(self) -> None:
        # --host/--port are force-injected; a config that also sets them would
        # emit duplicate args, and a different container port would desync the
        # compose port mapping and healthcheck.
        command = self.module.render_command(
            {"host": "127.0.0.1", "port": 9999},
            hf_repo="org/repo",
            hf_file="model.gguf",
        )
        self.assertEqual(command.count("--host"), 1)
        self.assertEqual(command.count("--port"), 1)
        self.assertEqual(command[command.index("--host") + 1], "0.0.0.0")
        self.assertEqual(command[command.index("--port") + 1], "8080")
        self.assertNotIn("127.0.0.1", command)
        self.assertNotIn("9999", command)

    def test_metrics_is_forced_on_and_never_duplicated(self) -> None:
        """`--metrics` is force-injected so the dashboard's live tok/s poll has
        a /metrics endpoint; a config that also sets it must not double it."""
        forced = self.module.render_command(
            {}, hf_repo="org/repo", hf_file="model.gguf"
        )
        self.assertEqual(forced.count("--metrics"), 1)

        # config sets it too (either polarity) — still exactly one bare flag.
        for value in (True, False):
            command = self.module.render_command(
                {"metrics": value}, hf_repo="org/repo", hf_file="model.gguf"
            )
            self.assertEqual(command.count("--metrics"), 1)
            idx = command.index("--metrics")
            next_token = command[idx + 1] if idx + 1 < len(command) else ""
            self.assertNotIn(next_token, {"on", "off", "true", "false", "True", "False"})


class QuickSetupSuffixLogicTests(unittest.TestCase):
    """Smoke-test the name collision suffix algorithm used in QuickSetupScreen."""

    @staticmethod
    def _resolve(safe_name: str, existing_profiles: set[str], existing_configs: set[str]) -> str:
        original = safe_name
        suffix = 0
        while safe_name in existing_profiles or safe_name in existing_configs:
            suffix += 1
            safe_name = f"{original}-{suffix}"
        return safe_name

    def test_no_collision_returns_original(self) -> None:
        self.assertEqual(self._resolve("llama", set(), set()), "llama")

    def test_collision_appends_suffix_1(self) -> None:
        self.assertEqual(self._resolve("llama", {"llama"}, set()), "llama-1")

    def test_collision_increments_until_free(self) -> None:
        self.assertEqual(
            self._resolve("llama", {"llama", "llama-1", "llama-2"}, {"llama-3"}),
            "llama-4",
        )

    def test_collision_across_profile_and_config(self) -> None:
        self.assertEqual(self._resolve("llama", {"llama"}, {"llama-1"}), "llama-2")


class OnboardingTests(unittest.TestCase):
    def test_render_env_overrides_prompted_keys(self) -> None:
        from tui.common import onboarding

        rendered = onboarding._render_env(
            {
                "HF_CACHE_PATH": "/abs/cache",
                "MODEL_DIR": "/abs/models",
                "HF_TOKEN": "hf_secret",
            }
        )
        lines = rendered.splitlines()
        self.assertIn("HF_CACHE_PATH=/abs/cache", lines)
        self.assertIn("MODEL_DIR=/abs/models", lines)
        self.assertIn("HF_TOKEN=hf_secret", lines)
        # Unprompted keys are preserved (key kept, value not asserted to avoid
        # coupling to the template); comments survive.
        self.assertTrue(any(line.startswith("LLAMACPP_IMAGE=") for line in lines))
        self.assertTrue(any(line.startswith("#") for line in lines))

    def test_needs_onboarding_reflects_env_file(self) -> None:
        from tui.common import onboarding

        with tempfile.TemporaryDirectory() as d:
            env_path = Path(d) / ".env.common"
            with patch.object(onboarding, "COMMON_ENV", env_path):
                self.assertTrue(onboarding.needs_onboarding())
                env_path.write_text(
                    "HF_CACHE_PATH=/x\n"
                    "PREPARE_DOWNLOADER_IMAGE=vllm/vllm-openai:v0.27.1\n"
                )
                env_path.chmod(0o600)
                self.assertFalse(onboarding.needs_onboarding())

    def test_run_onboarding_writes_valid_env(self) -> None:
        from tui.common import onboarding

        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            ask = Mock(
                side_effect=[str(dpath / "cache"), str(dpath / "models"), ""]
            )
            with (
                patch.object(onboarding, "COMMON_ENV", dpath / ".env.common"),
                patch("rich.prompt.Prompt.ask", ask),
            ):
                ok = onboarding.run_onboarding()

            self.assertTrue(ok)
            written = (dpath / ".env.common").read_text()
            self.assertIn(f"HF_CACHE_PATH={dpath / 'cache'}", written)
            self.assertIn(f"MODEL_DIR={dpath / 'models'}", written)
            self.assertIn("HF_TOKEN=", written)
            self.assertEqual((dpath / ".env.common").stat().st_mode & 0o777, 0o600)
            # The bind-mount targets are pre-created so docker doesn't own them.
            self.assertTrue((dpath / "cache").is_dir())
            self.assertTrue((dpath / "models").is_dir())
            self.assertTrue(ask.call_args_list[2].kwargs["password"])


class VersionCheckTests(unittest.TestCase):
    def test_repo_slug_parses_remote_url_forms(self) -> None:
        from tui.common import version_check as vc

        for url in (
            "https://github.com/Changroro/llmux.git",
            "https://github.com/Changroro/llmux",
            "git@github.com:Changroro/llmux.git",
            "ssh://git@github.com/Changroro/llmux.git",
        ):
            with patch.object(vc, "_git", return_value=(0, url + "\n")):
                self.assertEqual(vc._repo_slug(), "Changroro/llmux", msg=url)

    def test_repo_slug_none_on_non_github_or_failure(self) -> None:
        from tui.common import version_check as vc

        with patch.object(vc, "_git", return_value=(0, "https://gitlab.com/x/y.git\n")):
            self.assertIsNone(vc._repo_slug())
        with patch.object(vc, "_git", return_value=(1, "")):
            self.assertIsNone(vc._repo_slug())

    def test_is_behind_uses_commit_ancestry(self) -> None:
        from tui.common import version_check as vc

        def absent_commit(*args, **kwargs):
            if args[0] == "cat-file":
                return 1, ""
            if args[0] == "rev-parse":
                return 0, "false\n"
            raise AssertionError(args)

        with patch.object(vc, "_git", side_effect=absent_commit):
            self.assertIs(vc._is_behind("deadbeef"), True)

        # Present and an ancestor of HEAD → up to date.
        def have_and_ancestor(*args, **kwargs):
            if args[0] == "merge-base":
                return 0, ""
            return 0, ""  # cat-file -e succeeds

        with patch.object(vc, "_git", side_effect=have_and_ancestor):
            self.assertIs(vc._is_behind("deadbeef"), False)

        # Present but NOT an ancestor → behind.
        def have_not_ancestor(*args, **kwargs):
            if args[0] == "merge-base":
                return 1, ""
            return 0, ""

        with patch.object(vc, "_git", side_effect=have_not_ancestor):
            self.assertIs(vc._is_behind("deadbeef"), True)

    def test_is_behind_undecided_on_shallow_clone(self) -> None:
        from tui.common import version_check as vc

        # The release commit is absent, but the repo is shallow — old commits
        # are simply not fetched, so behind/ahead is genuinely undecidable.
        def shallow(*args, **kwargs):
            if args[0] == "cat-file":
                return 1, ""
            if args[0] == "rev-parse":  # --is-shallow-repository
                return 0, "true\n"
            return 0, ""

        with patch.object(vc, "_git", side_effect=shallow):
            self.assertIsNone(vc._is_behind("deadbeef"))

    def test_only_failures_back_off(self) -> None:
        from tui.common import version_check as vc

        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "version-check.json"
            with patch.object(vc, "_CACHE_FILE", cache):
                self.assertFalse(vc._cooldown_active())  # missing
                vc._record_failure()
                self.assertTrue(vc._cooldown_active())  # just failed
                vc._clear_failure()
                self.assertFalse(vc._cooldown_active())  # a success clears it
                cache.write_text('{"failed_at": 0}')
                self.assertFalse(vc._cooldown_active())  # epoch 0 → expired

    def test_local_version_read_from_pyproject(self) -> None:
        from tui.common import version_check as vc

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "pyproject.toml").write_text(
                '[project]\nname = "llmux"\nversion = "9.9.9"\n'
            )
            with patch.object(vc, "PROJECT_ROOT", root):
                self.assertEqual(vc._local_version(), "9.9.9")

    def test_matching_tag_skips_the_second_api_call(self) -> None:
        from tui.common import version_check as vc

        with (
            patch.object(vc, "_is_git_checkout", return_value=True),
            patch.object(vc, "_cooldown_active", return_value=False),
            patch.object(vc, "_clear_failure"),
            patch.object(vc, "_repo_slug", return_value="o/r"),
            patch.object(vc, "_local_version", return_value="2.6.0"),
            patch.object(vc, "_latest_release", return_value=("v2.6.0", "url")),
            patch.object(vc, "_release_commit") as commit,
        ):
            status = vc.resolve_status()
        self.assertEqual(status.state, vc.CURRENT)
        commit.assert_not_called()

    def test_failed_lookup_is_unknown_and_records_cooldown(self) -> None:
        from tui.common import version_check as vc

        with (
            patch.object(vc, "_is_git_checkout", return_value=True),
            patch.object(vc, "_cooldown_active", return_value=False),
            patch.object(vc, "_repo_slug", return_value="o/r"),
            patch.object(vc, "_local_version", return_value="2.6.0"),
            patch.object(vc, "_latest_release", return_value=None),
            patch.object(vc, "_record_failure") as failed,
        ):
            status = vc.resolve_status()
        self.assertEqual(status.state, vc.UNKNOWN)
        self.assertIn("rate-limited", status.detail)
        failed.assert_called_once()

    def test_explicit_check_ignores_the_cooldown(self) -> None:
        from tui.common import version_check as vc

        with (
            patch.object(vc, "_is_git_checkout", return_value=True),
            patch.object(vc, "_cooldown_active", return_value=True) as cooling,
            patch.object(vc, "_clear_failure"),
            patch.object(vc, "_repo_slug", return_value="o/r"),
            patch.object(vc, "_local_version", return_value="2.6.0"),
            patch.object(vc, "_latest_release", return_value=("v2.6.0", "url")),
        ):
            forced = vc.resolve_status(respect_cooldown=False)
            backed_off = vc.resolve_status()
        self.assertEqual(forced.state, vc.CURRENT)
        self.assertEqual(backed_off.state, vc.UNKNOWN)
        cooling.assert_called_once()  # only the cooldown-respecting call asks

    def test_local_clean_main_gates_auto_update(self) -> None:
        from tui.common import version_check as vc

        def clean_main(*args, **kwargs):
            if args[0] == "rev-parse":
                return 0, "main\n"
            if args[0] == "status":
                return 0, ""
            return 0, ""

        with patch.object(vc, "_git", side_effect=clean_main):
            self.assertTrue(vc._local_clean_main())

        def feature_branch(*args, **kwargs):
            if args[0] == "rev-parse":
                return 0, "feat/x\n"
            return 0, ""

        with patch.object(vc, "_git", side_effect=feature_branch):
            self.assertFalse(vc._local_clean_main())

        def dirty_main(*args, **kwargs):
            if args[0] == "rev-parse":
                return 0, "main\n"
            if args[0] == "status":
                return 0, " M tui/app.py\n"
            return 0, ""

        with patch.object(vc, "_git", side_effect=dirty_main):
            self.assertFalse(vc._local_clean_main())

    def test_check_for_update_up_to_date_is_silent(self) -> None:
        from tui.common import version_check as vc

        with (
            patch.object(vc, "resolve_status",
                         return_value=vc.UpdateStatus(vc.CURRENT, tag="v9.9.9")),
            patch.object(vc, "_prompt_and_update") as prompt,
        ):
            vc.check_for_update()  # up to date → returns, never prompts
        prompt.assert_not_called()

    def test_startup_check_is_silent_when_undecidable(self) -> None:
        from tui.common import version_check as vc

        with (
            patch.object(vc, "resolve_status",
                         return_value=vc.UpdateStatus(vc.UNKNOWN, detail="offline")),
            patch.object(vc, "_prompt_and_update") as prompt,
        ):
            vc.check_for_update()
        prompt.assert_not_called()

    def test_check_for_update_propagates_systemexit_after_update(self) -> None:
        from tui.common import version_check as vc

        with (
            patch.object(vc, "resolve_status",
                         return_value=vc.UpdateStatus(vc.BEHIND, tag="v9.9.9", url="url")),
            patch.object(vc, "_prompt_and_update", side_effect=SystemExit(0)),
        ):
            # SystemExit from a successful self-update must reach the caller
            # so the process exits and the user restarts on fresh code.
            with self.assertRaises(SystemExit):
                vc.check_for_update()

    def test_check_for_update_swallows_errors(self) -> None:
        from tui.common import version_check as vc

        with patch.object(vc, "resolve_status", side_effect=RuntimeError("boom")):
            vc.check_for_update()  # any non-SystemExit error must not escape


class FindCachedGgufTests(unittest.TestCase):
    """llama-server `-hf` downloads land in the HF hub cache, not MODEL_DIR."""

    def _make_hub(self, tmp: Path, repo: str, rev: str, filename: str) -> Path:
        org, _, name = repo.partition("/")
        snap = tmp / "hub" / f"models--{org}--{name}" / "snapshots" / rev
        snap.mkdir(parents=True)
        gguf = snap / filename
        gguf.write_bytes(b"\0" * 2048)
        return gguf

    def test_finds_gguf_in_hub_snapshot_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            expected = self._make_hub(
                tmp, "unsloth/Qwen3-8B-GGUF", "abc123", "Qwen3-8B-Q4_K_M.gguf"
            )

            with patch.object(lbackend, "_get_hf_cache_dir", return_value=tmp):
                found = lbackend.find_cached_gguf(
                    "unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"
                )

            self.assertEqual(found, expected)

    def test_returns_none_when_repo_or_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_hub(tmp, "unsloth/Qwen3-8B-GGUF", "abc123", "a.gguf")

            with patch.object(lbackend, "_get_hf_cache_dir", return_value=tmp):
                # unknown repo
                self.assertIsNone(lbackend.find_cached_gguf("other/Repo", "a.gguf"))
                # known repo, unknown file
                self.assertIsNone(
                    lbackend.find_cached_gguf("unsloth/Qwen3-8B-GGUF", "nope.gguf")
                )
                # empty inputs must not glob the whole cache
                self.assertIsNone(lbackend.find_cached_gguf("", "a.gguf"))
                self.assertIsNone(lbackend.find_cached_gguf("unsloth/x", ""))

    def test_list_cached_gguf_reports_repo_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_hub(tmp, "unsloth/Qwen3-8B-GGUF", "abc123", "q4.gguf")

            with patch.object(lbackend, "_get_hf_cache_dir", return_value=tmp):
                cached = lbackend.list_cached_gguf()

            self.assertEqual(len(cached), 1)
            self.assertEqual(cached[0]["repo"], "unsloth/Qwen3-8B-GGUF")
            self.assertEqual(cached[0]["name"], "q4.gguf")
            self.assertEqual(cached[0]["size_bytes"], 2048)


class SnapshotParseTests(unittest.TestCase):
    def test_parses_full_vllm_metric_families(self) -> None:
        from tui.common.metrics import parse_snapshot

        text = (
            'vllm:prompt_tokens_total{model_name="m"} 1000.0\n'
            'vllm:generation_tokens_total{model_name="m"} 2000.0\n'
            'vllm:num_requests_running{model_name="m"} 3.0\n'
            'vllm:num_requests_waiting{model_name="m"} 1.0\n'
            'vllm:kv_cache_usage_perc{model_name="m"} 0.34\n'
            'vllm:prefix_cache_hits_total{model_name="m"} 80.0\n'
            'vllm:prefix_cache_queries_total{model_name="m"} 100.0\n'
            'vllm:num_preemptions_total{model_name="m"} 2.0\n'
            'vllm:request_success_total{model_name="m"} 42.0\n'
            'vllm:time_to_first_token_seconds_sum{model_name="m"} 12.5\n'
            'vllm:time_to_first_token_seconds_count{model_name="m"} 50.0\n'
            'vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="m"} 5.0\n'
            'vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} 50.0\n'
            'vllm:inter_token_latency_seconds_sum{model_name="m"} 7.0\n'
            'vllm:inter_token_latency_seconds_count{model_name="m"} 1000.0\n'
        )
        m = parse_snapshot(text)
        self.assertEqual(m.backend, "vllm")
        self.assertEqual(m.prompt_tokens, 1000.0)
        self.assertEqual(m.generation_tokens, 2000.0)
        self.assertEqual(m.requests_running, 3.0)
        self.assertEqual(m.requests_waiting, 1.0)
        self.assertEqual(m.kv_cache_usage, 0.34)
        self.assertEqual(m.prefix_hits, 80.0)
        self.assertEqual(m.prefix_queries, 100.0)
        self.assertEqual(m.preemptions, 2.0)
        self.assertEqual(m.requests_finished, 42.0)
        self.assertEqual(m.ttft.sum, 12.5)
        # The _bucket lines must NOT be counted toward _count.
        self.assertEqual(m.ttft.count, 50.0)
        self.assertEqual(m.ttft.buckets, {0.1: 5.0, float("inf"): 50.0})
        self.assertEqual(m.tpot.sum, 7.0)
        self.assertEqual(m.tpot.count, 1000.0)

    def test_kv_cache_first_family_wins_no_double_count(self) -> None:
        from tui.common.metrics import parse_snapshot

        # A server exposing both the new and the legacy name must not sum them.
        m = parse_snapshot(
            "vllm:kv_cache_usage_perc 0.4\n"
            "vllm:gpu_cache_usage_perc 0.4\n"
        )
        self.assertEqual(m.kv_cache_usage, 0.4)

    def test_tpot_falls_back_to_legacy_name(self) -> None:
        from tui.common.metrics import parse_snapshot

        m = parse_snapshot(
            "vllm:time_per_output_token_seconds_sum 7.0\n"
            "vllm:time_per_output_token_seconds_count 100.0\n"
        )
        self.assertEqual(m.tpot.sum, 7.0)
        self.assertEqual(m.tpot.count, 100.0)

    def test_parses_llamacpp_names_and_leaves_absent_histograms_none(self) -> None:
        from tui.common.metrics import parse_snapshot

        text = (
            "llamacpp:prompt_tokens_total 500\n"
            "llamacpp:tokens_predicted_total 800\n"
            "llamacpp:requests_processing 2\n"
            "llamacpp:requests_deferred 0\n"
            "llamacpp:kv_cache_usage_ratio 0.5\n"
            "llamacpp:predicted_tokens_seconds 85.0\n"
        )
        m = parse_snapshot(text)
        self.assertEqual(m.backend, "llamacpp")
        self.assertEqual(m.prompt_tokens, 500.0)
        self.assertEqual(m.generation_tokens, 800.0)
        self.assertEqual(m.requests_running, 2.0)
        self.assertEqual(m.requests_waiting, 0.0)
        self.assertEqual(m.kv_cache_usage, 0.5)
        self.assertEqual(m.gen_tps_gauge, 85.0)
        self.assertIsNone(m.ttft)
        self.assertIsNone(m.tpot)
        self.assertIsNone(m.prefix_hits)
        self.assertEqual(m.token_counters(), (500.0, 800.0))

    def test_empty_body_yields_all_none(self) -> None:
        from tui.common.metrics import parse_snapshot

        m = parse_snapshot("# only comments\nunrelated_metric 1.0\n")
        self.assertEqual(m.backend, "unknown")
        self.assertIsNone(m.prompt_tokens)
        self.assertIsNone(m.requests_running)
        self.assertIsNone(m.ttft)
        self.assertIsNone(m.token_counters())


class HistTests(unittest.TestCase):
    def test_avg_and_quantile_from_buckets(self) -> None:
        from tui.common.metrics import Hist

        # Cumulative buckets: 10 samples, evenly spread across 0–1s.
        h = Hist(sum=5.0, count=10.0, buckets={
            0.2: 2.0, 0.4: 4.0, 0.6: 6.0, 0.8: 8.0, 1.0: 10.0, float("inf"): 10.0,
        })
        self.assertEqual(h.avg(), 0.5)
        # p50 → target 5 sits between le=0.4 (cum 4) and le=0.6 (cum 6).
        self.assertAlmostEqual(h.quantile(0.5), 0.5)
        self.assertAlmostEqual(h.quantile(0.9), 0.9)

    def test_quantile_none_without_data(self) -> None:
        from tui.common.metrics import Hist

        self.assertIsNone(Hist().quantile(0.5))
        self.assertIsNone(Hist(sum=1.0, count=0.0).avg())


class TokenMetricsTests(unittest.TestCase):
    def test_parses_vllm_labelled_counters_and_sums_label_sets(self) -> None:
        from tui.common.metrics import parse_token_counters

        text = (
            "# HELP vllm:prompt_tokens_total Number of prefill tokens.\n"
            "# TYPE vllm:prompt_tokens_total counter\n"
            'vllm:prompt_tokens_total{model_name="a"} 10.0\n'
            'vllm:prompt_tokens_total{model_name="b"} 5.0\n'
            'vllm:generation_tokens_total{model_name="a"} 100.0\n'
            'vllm:generation_tokens_total{model_name="b"} 20.0\n'
        )

        self.assertEqual(parse_token_counters(text), (15.0, 120.0))

    def test_parses_llamacpp_counter_names(self) -> None:
        from tui.common.metrics import parse_token_counters

        text = (
            "# TYPE llamacpp:prompt_tokens_total counter\n"
            "llamacpp:prompt_tokens_total 7\n"
            "llamacpp:tokens_predicted_total 42\n"
        )

        self.assertEqual(parse_token_counters(text), (7.0, 42.0))

    def test_returns_none_when_no_token_counters_present(self) -> None:
        from tui.common.metrics import parse_token_counters

        self.assertIsNone(parse_token_counters(""))
        self.assertIsNone(parse_token_counters("# only comments\nother_metric 1.0\n"))

    def test_tracker_first_sample_has_no_rate_then_deltas(self) -> None:
        from tui.common.metrics import ThroughputTracker

        t = ThroughputTracker()
        self.assertIsNone(t.update("p", (0.0, 0.0), now=0.0))

        rate = t.update("p", (10.0, 200.0), now=2.0)
        self.assertIsNotNone(rate)
        assert rate is not None
        self.assertAlmostEqual(rate[0], 5.0)     # 10 prompt tokens / 2s
        self.assertAlmostEqual(rate[1], 100.0)   # 200 gen tokens / 2s

    def test_tracker_resets_on_counter_decrease(self) -> None:
        from tui.common.metrics import ThroughputTracker

        t = ThroughputTracker()
        t.update("p", (100.0, 500.0), now=0.0)
        # Server restarted → counters reset to 0. Diffing across that would
        # emit a large negative rate; the tracker must re-baseline instead.
        self.assertIsNone(t.update("p", (0.0, 0.0), now=2.0))
        # Next sample diffs against the fresh baseline.
        rate = t.update("p", (4.0, 20.0), now=4.0)
        assert rate is not None
        self.assertAlmostEqual(rate[1], 10.0)


def _docker_available() -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "ls", "-q"], capture_output=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def _any_local_image() -> str:
    r = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True, timeout=15,
    )
    for line in r.stdout.splitlines():
        if line.strip() and "<none>" not in line:
            return line.strip()
    return ""


class DevImageProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_dev_image_is_false(self) -> None:
        from tui.common import dev_build

        spec = dev_build.DevBuildSpec(
            backend="test",
            image_prefix="image",
            src_dir=Path("."),
            default_repo_url="repo",
        )
        with patch.object(
            dev_build,
            "_run",
            AsyncMock(return_value=(1, "Error response from daemon: No such image: image:tag")),
        ):
            self.assertFalse(await dev_build.image_exists_locally(spec, "tag"))

    async def test_dev_image_probe_failure_raises(self) -> None:
        from tui.common import dev_build

        spec = dev_build.DevBuildSpec(
            backend="test",
            image_prefix="image",
            src_dir=Path("."),
            default_repo_url="repo",
        )
        with patch.object(
            dev_build,
            "_run",
            AsyncMock(return_value=(1, "Cannot connect to the Docker daemon")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot connect"):
                await dev_build.image_exists_locally(spec, "tag")
            with self.assertRaisesRegex(RuntimeError, "Cannot connect"):
                await dev_build.get_image_label("image:tag", "label")


@unittest.skipUnless(_docker_available(), "docker not available")
class GetImageLabelDockerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Unmocked: the previous `--format={{index .Config.Labels 'k'}}` used single
    quotes, which Go templates read as a rune literal — docker exited rc=64 for
    every lookup, so image_matches() was always False and `--dev` rebuilt every
    time. Mocked tests could not catch it; this one shells out for real."""

    async def test_missing_label_returns_empty_without_a_parse_error(self) -> None:
        from tui.common import dev_build

        image = _any_local_image()
        if not image:
            self.skipTest("no local docker images to inspect")

        # rc must be 0 (template parsed) even though the label is absent — the
        # buggy format returned rc=64 here, indistinguishable from "no label".
        rc, _ = await dev_build._run(
            "docker", "inspect", image,
            '--format={{index .Config.Labels "llmux.test.absent"}}',
            timeout=20,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(
            await dev_build.get_image_label(image, "llmux.test.absent"), ""
        )

    async def test_present_label_round_trips(self) -> None:
        from tui.common import dev_build

        image = _any_local_image()
        if not image:
            self.skipTest("no local docker images to inspect")

        keys = subprocess.run(
            ["docker", "image", "inspect", image,
             "--format", '{{range $k, $v := .Config.Labels}}{{$k}}\n{{end}}'],
            capture_output=True, text=True, timeout=20,
        ).stdout.split()
        if not keys:
            self.skipTest(f"{image} carries no labels")

        key = keys[0]
        expected = subprocess.run(
            ["docker", "image", "inspect", image,
             "--format", '{{index .Config.Labels "' + key + '"}}'],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()

        self.assertEqual(await dev_build.get_image_label(image, key), expected)


class ImageTagValidationTests(unittest.TestCase):
    def test_empty_is_allowed(self) -> None:
        from tui.common.dev_build import image_tag_error
        self.assertEqual(image_tag_error(""), "")
        self.assertEqual(image_tag_error("   "), "")

    def test_dev_tag_must_already_be_sanitized(self) -> None:
        from tui.common.dev_build import image_tag_error
        # A slash can't survive as a docker tag — must be rejected with a hint.
        err = image_tag_error("vllm-dev:feat/foo")
        self.assertTrue(err)
        self.assertIn("vllm-dev:feat-foo", err)
        # Already-sanitized dev tag passes.
        self.assertEqual(image_tag_error("llamacpp-dev:feat-foo"), "")

    def test_generic_reference_tag_is_validated(self) -> None:
        from tui.common.dev_build import image_tag_error
        self.assertEqual(image_tag_error("ghcr.io/foo/bar:v1"), "")
        # A host:port registry ref with a real tag passes (the first colon is
        # the port, the last is the tag).
        self.assertEqual(image_tag_error("localhost:5000/foo:v1"), "")
        # An illegal tag after the last colon is rejected.
        self.assertTrue(image_tag_error("ghcr.io/foo/bar:bad tag"))

    def test_untagged_and_latest_refs_are_rejected(self) -> None:
        # A pinned image_tag must name a specific version — an untagged ref (or a
        # `:latest` alias) resolves to `:latest`, the same ambiguous alias the
        # Custom Tag field hard-rejects.
        from tui.common.dev_build import image_tag_error
        self.assertTrue(image_tag_error("vllm/vllm-openai"))          # untagged
        self.assertTrue(image_tag_error("localhost:5000/foo"))       # host:port, no tag
        self.assertTrue(image_tag_error("vllm/vllm-openai:latest"))  # explicit latest
        self.assertIn("latest", image_tag_error("vllm/vllm-openai:latest"))


class VllmContainerStatusImageTests(unittest.IsolatedAsyncioTestCase):
    """The vLLM/llama.cpp ContainerStatus is a field-for-field mirror; the vLLM
    side was leaving `image` empty."""

    async def test_image_field_is_populated_from_profile(self) -> None:
        from tui.backends.vllm import backend_runtime as rt

        profile = backend.Profile(
            name="p", container_name="p", port="8000", image_tag="vllm-dev:mine",
            config_name="",
        )

        with patch.dict(
            rt.get_container_statuses.__globals__,
            {
                "list_profile_names": lambda: ["p"],
                "load_profile": lambda _n: profile,
            },
        ), patch.object(
            rt.common_docker, "container_snapshots", AsyncMock(return_value={})
        ):
            statuses = await rt.get_container_statuses()

        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0].image, "vllm-dev:mine")

    async def test_status_probe_failure_raises_instead_of_reporting_stopped(self) -> None:
        from tui.backends.llamacpp import backend_runtime as lrt
        from tui.backends.vllm import backend_runtime as rt

        with patch.object(
            rt.common_docker,
            "container_snapshots",
            AsyncMock(side_effect=RuntimeError("docker ps failed")),
        ):
            with self.assertRaises(RuntimeError):
                await rt.get_container_statuses()

        with patch.object(
            lrt.common_docker,
            "container_snapshots",
            AsyncMock(side_effect=RuntimeError("docker ps failed")),
        ):
            with self.assertRaises(RuntimeError):
                await lrt.get_container_statuses()


class DevBuildCustomTagTests(unittest.IsolatedAsyncioTestCase):
    """The runtime sanitizes `--tag feat/foo` to `feat-foo` before looking the
    image up, so the builder has to sanitize too — otherwise it builds under a
    name the start path never resolves (or fails on an invalid reference)."""

    async def test_custom_tag_is_sanitized(self) -> None:
        from tui.common import dev_build

        spec = dev_build.DevBuildSpec(
            backend="vllm",
            image_prefix="vllm-dev",
            src_dir=Path("/tmp/does-not-matter"),
            default_repo_url="https://example.invalid/repo.git",
        )
        tag_line = ""
        # Stop at the Tag line — everything after it clones and shells out.
        async for kind, payload in dev_build.stream_build(
            spec, "main", custom_tag="feat/foo"
        ):
            if kind == "log" and payload.startswith("Tag: "):
                tag_line = payload
                break

        self.assertEqual(tag_line, "Tag: vllm-dev:feat-foo")

    def test_repository_credentials_are_removed_from_metadata(self) -> None:
        from tui.common import dev_build

        self.assertEqual(
            dev_build.sanitize_repo_url(
                "https://user:secret@example.com/org/repo.git"
            ),
            "https://example.com/org/repo.git",
        )


class ComposeEnvExpansionTests(unittest.TestCase):
    """compose expands $VAR/~ when it reads --env-file, but we also merge those
    values into the process env — which *outranks* --env-file. Unexpanded, the
    template's default `HF_CACHE_PATH=/home/$USER/.cache/huggingface` got
    bind-mounted as a literal `/home/$USER` directory."""

    def test_expand_env_values_expands_vars_and_tilde(self) -> None:
        from tui.common.env import expand_env_values

        with patch.dict(os.environ, {"USER": "alice"}, clear=False):
            out = expand_env_values({
                "HF_CACHE_PATH": "/home/$USER/.cache/huggingface",
                "MODEL_DIR": "~/models",
                "PLAIN": "/abs/path",
            })

        self.assertEqual(out["HF_CACHE_PATH"], "/home/alice/.cache/huggingface")
        self.assertEqual(out["MODEL_DIR"], str(Path("~/models").expanduser()))
        self.assertEqual(out["PLAIN"], "/abs/path")

    def test_vllm_compose_env_expands_common_and_filters_profile_env(self) -> None:
        from tui.backends.vllm import backend_runtime as vrt

        profile = backend.Profile(name="p", config_name="p", port=8000)
        with patch.dict(os.environ, {"USER": "alice"}, clear=False), \
             patch.object(vrt, "_common_env",
                          lambda **kwargs: {"HF_CACHE_PATH": "/home/alice/.cache/huggingface"}), \
             patch.object(
                 vrt,
                 "_parse_env_file",
                 lambda _p: {
                     "MY_VAR": "$HOME/x",
                     "EXTRA_PIP_PACKAGES": "$HOME/package.whl",
                 },
             ):
            env = vrt._compose_env(profile, use_dev=False, version_tag="v1")

        self.assertEqual(env["HF_CACHE_PATH"], "/home/alice/.cache/huggingface")
        self.assertNotIn("MY_VAR", env)
        self.assertEqual(env["EXTRA_PIP_PACKAGES"], "$HOME/package.whl")

    def test_llamacpp_compose_env_expands_common_and_filters_profile_env(self) -> None:
        from tui.backends.llamacpp import backend_runtime as lrt

        profile = lbackend.Profile(name="p", config_name="p", port=8080)
        calls = {"n": 0}

        def fake_parse(_p, *, expand=False):
            # First call is COMMON_ENV, second is the profile .env.
            calls["n"] += 1
            if calls["n"] == 1:
                self.assertTrue(expand)
                return {"HF_CACHE_PATH": "/home/alice/.cache/huggingface"}
            return {"MY_VAR": "$HOME/x", "CONFIG_NAME": "$HOME/config"}

        with patch.dict(os.environ, {"USER": "alice"}, clear=False), \
             patch.object(type(lrt.COMMON_ENV), "exists", lambda _s: True), \
             patch.object(type(profile.path), "exists", lambda _s: True), \
             patch.object(lrt, "_parse_env_file", fake_parse):
            env = lrt._compose_env(profile)

        self.assertEqual(env["HF_CACHE_PATH"], "/home/alice/.cache/huggingface")
        self.assertNotIn("MY_VAR", env)
        self.assertEqual(env["CONFIG_NAME"], "$HOME/config")


class LlamacppEnvVarsRoundTripTests(unittest.TestCase):
    """The llama.cpp Profile had no env_vars field, so the TUI's load→save cycle
    silently dropped anything the CLI had put there with --set."""

    def test_env_vars_survive_load_save_round_trip(self) -> None:
        stored = profile_store.StoredProfile(
            name="lcpp_env",
            backend="llamacpp",
            port=8080,
            gpu_id="0",
            hf_repo="org/x",
            hf_file="m.gguf",
            env_vars={"LLAMA_ARG_THREADS": "8"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config" / "llamacpp").mkdir(parents=True)
            with patch("tui.common.profile_store.PROFILES_YAML", root / "profiles.yaml"), \
                 patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"):
                profile_store.save_profile(stored)

                # The TUI edit path: backend.load_profile -> backend.save_profile.
                loaded = lbackend.load_profile("lcpp_env")
                self.assertEqual(loaded.env_vars, {"LLAMA_ARG_THREADS": "8"})

                lbackend.save_profile(loaded)

                again = profile_store.load_profile("lcpp_env", "llamacpp")
                assert again is not None
                self.assertEqual(again.env_vars, {"LLAMA_ARG_THREADS": "8"})


class DisabledParamsTests(unittest.TestCase):
    """Disabled config params round-trip via comment markers, and — critically —
    stay invisible to the YAML/flag parser the server itself uses."""

    def _tmp_config_dir(self, backend_mod):
        # Patch the module's CONFIG_DIR to an isolated temp dir.
        return tempfile.TemporaryDirectory()

    def test_vllm_round_trip_and_server_safety(self) -> None:
        from tui.backends.vllm import backend_storage as vs
        from tui.backends.vllm import backend_common as vc
        from tui.backends.vllm.backend_common import Config as VC

        with tempfile.TemporaryDirectory() as tmp:
            # save_config's mkdir uses backend_storage.CONFIG_DIR; Config.path
            # uses backend_common.CONFIG_DIR — patch both.
            with patch.object(vs, "CONFIG_DIR", Path(tmp)), \
                 patch.object(vc, "CONFIG_DIR", Path(tmp)):
                cfg = VC(
                    name="c", model="m/x", gpu_memory_utilization="0.85",
                    extra_params={"max-model-len": 4096},
                    disabled_params={"enforce-eager": True, "quantization": "fp8"},
                )
                vs.save_config(cfg)

                text = (Path(tmp) / "c.yaml").read_text()
                # The server reads YAML — disabled params must not be visible.
                server_view = yaml.safe_load(text)
                self.assertNotIn("enforce-eager", server_view)
                self.assertNotIn("quantization", server_view)
                self.assertIn("max-model-len", server_view)

                loaded = vs.load_config("c")
                self.assertEqual(loaded.extra_params, {"max-model-len": 4096})
                self.assertEqual(
                    loaded.disabled_params,
                    {"enforce-eager": True, "quantization": "fp8"},
                )

    def test_llamacpp_round_trip_and_server_safety(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(lbackend, "CONFIG_DIR", Path(tmp)):
                cfg = lbackend.Config(
                    name="c", params={"ctx-size": 2048, "alias": "a"},
                    disabled_params={"override-tensors": ".*=CPU"},
                )
                lbackend.save_config(cfg)

                text = (Path(tmp) / "c.yaml").read_text()
                server_view = yaml.safe_load(text)
                self.assertNotIn("override-tensors", server_view)
                self.assertEqual(server_view, {"ctx-size": 2048, "alias": "a"})

                loaded = lbackend.load_config("c")
                self.assertEqual(loaded.disabled_params, {"override-tensors": ".*=CPU"})

    def test_long_and_structured_values_survive_disable_enable(self) -> None:
        # C1: yaml's default width=80 used to wrap long markers; taking the
        # first physical line then truncated strings and corrupted list/dict
        # types on re-enable.
        from tui.common.config_markers import (
            render_disabled_markers,
            parse_disabled_markers,
        )

        cases = {
            "longstr": "x" * 85,
            "biglist": list(range(40)),
            "bigdict": {f"k{i}": i for i in range(20)},
            "pattern": ".*=CPU",
            "scalar": 0.85,
        }
        text = render_disabled_markers(cases)
        # Every marker is exactly one line.
        marker_lines = [ln for ln in text.splitlines() if ln.strip()]
        self.assertEqual(len(marker_lines), len(cases))
        for ln in marker_lines:
            self.assertTrue(ln.startswith("# llmux:disabled "))

        back = parse_disabled_markers(text)
        for k, v in cases.items():
            self.assertEqual(back[k], v)
            self.assertIs(type(back[k]), type(v))

    def test_active_key_wins_over_a_stale_marker(self) -> None:
        from tui.common.config_markers import parse_disabled_markers
        from tui.backends.llamacpp import backend as lb

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(lb, "CONFIG_DIR", Path(tmp)):
                # Hand-craft a file where the same key is both active and marked.
                (Path(tmp) / "c.yaml").write_text(
                    "ctx-size: 2048\n# llmux:disabled ctx-size: 999\n"
                )
                # Sanity: the marker parser does see it...
                self.assertIn("ctx-size", parse_disabled_markers(
                    (Path(tmp) / "c.yaml").read_text()
                ))
                # ...but load_config drops it because the active key wins.
                loaded = lb.load_config("c")
                self.assertEqual(loaded.params["ctx-size"], 2048)
                self.assertNotIn("ctx-size", loaded.disabled_params)

    def test_vllm_save_preserves_user_comments(self) -> None:
        # Editing a config used to erase every hand-written `#` note (PyYAML
        # can't round-trip comments). A header, an inline comment, and a
        # trailing block must all survive an edit that changes one value.
        from tui.backends.vllm import backend_storage as vs
        from tui.backends.vllm import backend_common as vc

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(vs, "CONFIG_DIR", Path(tmp)), \
                 patch.object(vc, "CONFIG_DIR", Path(tmp)):
                (Path(tmp) / "c.yaml").write_text(
                    "# header note\n"
                    "model: org/m  # inline note\n"
                    "max-model-len: 8192\n"
                    "# OFF-only trailing explanation.\n"
                )
                cfg = vs.load_config("c")
                cfg.extra_params["max-model-len"] = 4096  # a TUI/CLI edit
                vs.save_config(cfg)

                text = (Path(tmp) / "c.yaml").read_text()
                self.assertIn("# header note", text)
                self.assertIn("# inline note", text)
                self.assertIn("# OFF-only trailing explanation.", text)
                self.assertIn("max-model-len: 4096", text)
                # The server view is still valid YAML with the new value.
                self.assertEqual(yaml.safe_load(text)["max-model-len"], 4096)

    def test_llamacpp_save_preserves_user_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(lbackend, "CONFIG_DIR", Path(tmp)):
                (Path(tmp) / "c.yaml").write_text(
                    "# llama config\nctx-size: 32768  # context length\nn-gpu-layers: 99\n"
                )
                cfg = lbackend.load_config("c")
                cfg.params["ctx-size"] = 16384
                lbackend.save_config(cfg)

                text = (Path(tmp) / "c.yaml").read_text()
                self.assertIn("# llama config", text)
                self.assertIn("# context length", text)
                self.assertIn("ctx-size: 16384", text)

    def test_comment_free_config_stays_byte_identical(self) -> None:
        # The comment-preserving path must not touch comment-less files — their
        # plain PyYAML output has to stay exactly as before.
        from tui.backends.vllm import backend_storage as vs
        from tui.backends.vllm import backend_common as vc

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(vs, "CONFIG_DIR", Path(tmp)), \
                 patch.object(vc, "CONFIG_DIR", Path(tmp)):
                data = {
                    "model": "org/m",
                    "gpu-memory-utilization": "0.9",
                    "max-model-len": 2048,
                }
                plain = yaml.dump(
                    data, default_flow_style=False, allow_unicode=True, sort_keys=False
                )
                (Path(tmp) / "c.yaml").write_text(plain)
                vs.save_config(vs.load_config("c"))  # no-op re-save
                self.assertEqual((Path(tmp) / "c.yaml").read_text(), plain)


class EnvLineQuotingTests(unittest.TestCase):
    """docker compose reads the rendered .env with a dotenv parser, not a shell.

    shlex.quote turns `it's` into `'it'"'"'s'`, which that parser rejects — the
    profile saved fine and `up` then died with an opaque error.
    """

    def test_single_quote_in_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            profile_store._env_line("K", "it's")
        self.assertIn("single quote", str(ctx.exception))

    def test_double_quote_in_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            profile_store._env_line("K", 'say "hi"')

    def test_newline_in_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            profile_store._env_line("K", "a\nb")

    def test_control_character_in_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            profile_store._env_line("K", "a\x07b")

    def test_plain_and_spaced_values_still_render(self) -> None:
        # Values compose *can* read must keep working — including spaces and
        # commas, which shlex quotes but dotenv handles.
        self.assertEqual(profile_store._env_line("K", "v"), "K=v")
        self.assertEqual(profile_store._env_line("K", "a b"), "K='a b'")
        self.assertEqual(profile_store._env_line("K", "0,1"), "K=0,1")
        self.assertEqual(profile_store._env_line("K", 8000), "K=8000")

    def test_env_value_rejection_reports_the_offending_class(self) -> None:
        self.assertEqual(profile_store.env_value_rejection("ok"), "")
        self.assertIn("single quote", profile_store.env_value_rejection("it's"))
        self.assertIn("newline", profile_store.env_value_rejection("a\nb"))


class ListHfRepoFilesTests(unittest.IsolatedAsyncioTestCase):
    """The tree API is non-recursive and paginated by default — a repo that
    keeps its GGUFs in per-quant subfolders (the standard layout for large
    sharded models) would otherwise look empty."""

    class _FakeResponse:
        def __init__(self, payload: list[dict], link: str = "") -> None:
            self._payload = payload
            self.headers = {"Link": link} if link else {}

        def read(self) -> bytes:
            return json.dumps(self._payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> bool:
            return False

    def _patch_urlopen(self, pages: dict[str, "ListHfRepoFilesTests._FakeResponse"]):
        requested: list[str] = []

        def fake_open_url(req, timeout=0):
            requested.append(req.full_url)
            return pages[req.full_url]

        # The listing goes through ssl_ctx.open_url so it carries a working CA
        # bundle; patching urlopen would miss it.
        return requested, patch.object(lbackend, "open_url", fake_open_url)

    async def test_recursive_flag_and_subfolder_paths_preserved(self) -> None:
        base = "https://huggingface.co/api/models/org/repo/tree/main?recursive=true"
        pages = {
            base: self._FakeResponse(
                [
                    {"type": "directory", "path": "Q4_K_M"},
                    {"type": "file", "path": "Q4_K_M/model-00001-of-00002.gguf"},
                ]
            )
        }
        requested, patcher = self._patch_urlopen(pages)
        with patcher:
            files = await lbackend.list_hf_repo_files("org/repo")

        self.assertEqual(requested, [base])
        self.assertEqual(
            [f["path"] for f in files if f["type"] == "file"],
            ["Q4_K_M/model-00001-of-00002.gguf"],
        )

    async def test_follows_link_next_and_merges_pages(self) -> None:
        base = "https://huggingface.co/api/models/org/repo/tree/main?recursive=true"
        page2 = f"{base}&cursor=abc"
        pages = {
            base: self._FakeResponse(
                [{"type": "file", "path": "Q4_K_M/a.gguf"}],
                link=f'<{page2}>; rel="next"',
            ),
            page2: self._FakeResponse([{"type": "file", "path": "Q8_0/b.gguf"}]),
        }
        requested, patcher = self._patch_urlopen(pages)
        with patcher:
            files = await lbackend.list_hf_repo_files("org/repo")

        self.assertEqual(requested, [base, page2])
        self.assertEqual(
            [f["path"] for f in files],
            ["Q4_K_M/a.gguf", "Q8_0/b.gguf"],
        )


if __name__ == "__main__":
    unittest.main()




class PlainMonitorTests(unittest.IsolatedAsyncioTestCase):
    def _row(self, name="m1", backend="vllm", port=8000):
        from tui.common.adapter import DashboardRow
        return DashboardRow(backend=backend, profile_name=name, container_name=name,
                            port=port, running=True, model="Qwen/Q", detail="", gpu_id="0")

    def test_state_derives_rate_over_two_samples(self) -> None:
        from tui.common.monitor_render import MonitorState
        from tui.common.metrics import MetricsSnapshot
        st = MonitorState()
        d1 = st.update(MetricsSnapshot(prompt_tokens=0.0, generation_tokens=0.0), 100.0, 1.0)
        self.assertIsNone(d1.gen_tps)
        d2 = st.update(MetricsSnapshot(prompt_tokens=10.0, generation_tokens=200.0), 101.0, 1.0)
        self.assertAlmostEqual(d2.gen_tps, 200.0)
        self.assertAlmostEqual(d2.prompt_tps, 10.0)

    def _entry(self, snap):
        from tui.common.monitor_render import ModelEntry, MonitorState
        st = MonitorState()
        return ModelEntry(self._row(), snap, st, st.update(snap, 100.0, 1.0))

    def test_render_dashboard_shows_kv_and_gpu(self) -> None:
        import io
        from rich.console import Console
        from tui.common.monitor_render import render_dashboard
        from tui.common.metrics import MetricsSnapshot
        from tui.common.docker import GpuInfo
        m = MetricsSnapshot(backend="vllm", prompt_tokens=1.0, generation_tokens=2.0,
                            requests_running=3.0, requests_waiting=1.0, kv_cache_usage=0.34)
        gpus = [GpuInfo("0", "RTX", "8000", "16000", "78", "71", "210")]
        con = Console(width=110, file=io.StringIO())
        con.print(render_dashboard([self._entry(m)], gpus, {}, 110))
        out = con.file.getvalue()
        self.assertIn("KV", out)
        self.assertIn("34%", out)
        self.assertIn("GPU0", out)
        self.assertIn("210W", out)
        self.assertIn("REQUESTS", out)

    def test_render_dashboard_shows_gpu_with_no_models(self) -> None:
        """The monitor is a system view — GPUs render even with nothing up."""
        import io
        from rich.console import Console
        from tui.common.monitor_render import render_dashboard
        from tui.common.docker import GpuInfo
        gpus = [GpuInfo("0", "RTX", "8000", "16000", "78", "71", "210")]
        con = Console(width=110, file=io.StringIO())
        con.print(render_dashboard([], gpus, {"0": (12.0, 34.0)}, 110))
        out = con.file.getvalue()
        self.assertIn("GPU0", out)
        self.assertIn("210W", out)
        self.assertIn("rx 12", out)
        self.assertIn("MODELS", out)

    def test_render_dashboard_compacts_multiple_models(self) -> None:
        import io
        from rich.console import Console
        from tui.common.monitor_render import render_dashboard, ModelEntry, MonitorState
        from tui.common.metrics import MetricsSnapshot

        def entry(name):
            snap = MetricsSnapshot(backend="vllm", generation_tokens=1.0, kv_cache_usage=0.2)
            st = MonitorState()
            return ModelEntry(self._row(name), snap, st, st.update(snap, 100.0, 1.0))

        con = Console(width=110, file=io.StringIO())
        con.print(render_dashboard([entry("a"), entry("b")], [], {}, 110))
        out = con.file.getvalue()
        self.assertIn("a", out)
        self.assertIn("b", out)

    async def test_resolve_runs_system_view_when_none_running(self) -> None:
        """No profile given and nothing up is not an error — the GPU view still
        opens. Only an explicitly named, non-running profile is rejected."""
        from unittest.mock import patch, AsyncMock
        from tui.common import plain_monitor as mod
        with patch.object(mod, "_running_rows", AsyncMock(return_value=([], []))), \
             patch.object(mod, "run_plain_monitor", AsyncMock(return_value=None)) as run:
            rc = await mod._resolve_and_run(None)
            self.assertEqual(rc, 0)
            run.assert_awaited_once()

    async def test_resolve_rejects_non_running_name(self) -> None:
        from unittest.mock import patch, AsyncMock
        from tui.common import plain_monitor as mod
        with patch.object(mod, "_running_rows", AsyncMock(return_value=([self._row("a")], []))):
            rc = await mod._resolve_and_run("nope")
            self.assertEqual(rc, 1)

    async def test_resolve_does_not_claim_not_running_when_scan_failed(self) -> None:
        from unittest.mock import patch, AsyncMock
        from tui.common import plain_monitor as mod
        with patch.object(
            mod, "_running_rows", AsyncMock(return_value=([], ["docker down"]))
        ):
            rc = await mod._resolve_and_run("a")
        self.assertEqual(rc, 2)
