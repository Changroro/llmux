from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import typer
import yaml

from tui.common import recipes


def _recipe() -> recipes.Recipe:
    return recipes.Recipe(
        model_id="org/model",
        base_args=["--max-model-len", "4096"],
        variants=[
            recipes.RecipeVariant(name="large", vram_minimum_gb=80),
            recipes.RecipeVariant(name="small", vram_minimum_gb=20),
        ],
        features=[recipes.RecipeFeature(name="reasoning", args=["--reasoning-parser", "x"])],
    )


def test_recipe_merge_rejects_name_before_recipe_fetch() -> None:
    from tui.cli import config as cli_config

    fetch = AsyncMock(return_value=_recipe())
    with patch("tui.common.recipes.fetch_recipe", fetch):
        with pytest.raises(typer.BadParameter, match="invalid config name"):
            cli_config.config_from_recipe(
                "org/model",
                recipe_from="",
                variant="small",
                feature=[],
                name="../../victim",
                list_only=False,
                json_out=False,
                overwrite=False,
                merge=True,
            )
    fetch.assert_not_awaited()


def test_unknown_recipe_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown recipe feature.*typo"):
        recipes.build_config(_recipe(), None, ["typo"])


def test_cli_validates_recipe_feature_before_gpu_probe() -> None:
    from tui.cli import config as cli_config

    get_gpu_info = AsyncMock(side_effect=RuntimeError("nvidia-smi failed"))
    with patch("tui.common.recipes.fetch_recipe", AsyncMock(return_value=_recipe())), patch(
        "tui.common.docker.get_gpu_info", get_gpu_info
    ):
        with pytest.raises(typer.BadParameter, match="unknown recipe feature.*typo"):
            cli_config.config_from_recipe(
                "org/model",
                recipe_from="",
                variant="",
                feature=["typo"],
                name="target",
                list_only=False,
                json_out=True,
                overwrite=False,
                merge=False,
            )
    get_gpu_info.assert_not_awaited()


@pytest.mark.parametrize(
    "raw,path",
    [
        ({"model": {"model_id": "org/model", "base_args": "--trust-remote-code"}}, "model.base_args"),
        (
            {
                "model": {"model_id": "org/model", "base_args": []},
                "features": {"reasoning": {"args": "--reasoning-parser"}},
            },
            "features.reasoning.args",
        ),
        (
            {
                "model": {"model_id": "org/model", "base_args": []},
                "variants": {"fp8": {"extra_args": {"quantization": "fp8"}}},
            },
            "variants.fp8.extra_args",
        ),
        ({"model": {"model_id": "org/model", "base_args": []}, "features": None}, "features"),
    ],
)
def test_recipe_schema_rejects_non_list_args(raw: dict, path: str) -> None:
    with pytest.raises(ValueError, match=path):
        recipes._parse(raw, "org/model")


@pytest.mark.asyncio
async def test_remote_recipe_schema_error_is_normalized() -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"model:\n  model_id: org/model\n  base_args: --bad\n"

    with patch.object(recipes, "open_url", return_value=Response()):
        with pytest.raises(recipes.RecipeUnavailable, match="model.base_args"):
            await recipes.fetch_recipe("org/model")


def test_auto_variant_fails_when_no_variant_fits() -> None:
    from tui.cli import config as cli_config

    gpu = SimpleNamespace(memory_total="16384")
    with patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[gpu])):
        with pytest.raises(typer.BadParameter, match="no recipe variant fits"):
            cli_config._pick_recipe_variant(_recipe(), "")


def test_explicit_variant_does_not_require_gpu_fit() -> None:
    from tui.cli import config as cli_config

    with patch("tui.common.docker.get_gpu_info", AsyncMock()) as get_gpu_info:
        chosen = cli_config._pick_recipe_variant(_recipe(), "large")
    assert chosen.name == "large"
    get_gpu_info.assert_not_awaited()


def test_auto_variant_maps_gpu_probe_failure_to_bad_parameter() -> None:
    from tui.cli import config as cli_config

    with patch(
        "tui.common.docker.get_gpu_info",
        AsyncMock(side_effect=RuntimeError("nvidia-smi failed")),
    ):
        with pytest.raises(typer.BadParameter, match="could not read GPU memory"):
            cli_config._pick_recipe_variant(_recipe(), "")


@pytest.mark.asyncio
async def test_recipe_screen_reports_gpu_probe_failure() -> None:
    from textual.widgets import Static

    from tui.app import LlmuxApp
    from tui.backends.vllm.screens.config import ConfigFormScreen

    app = LlmuxApp()
    with patch(
        "tui.backends.vllm.screens.config.extract_vllm_params",
        AsyncMock(return_value={"max-model-len"}),
    ), patch("tui.common.recipes.fetch_recipe", AsyncMock(return_value=_recipe())), patch(
        "tui.common.docker.get_gpu_info",
        AsyncMock(side_effect=RuntimeError("nvidia-smi failed")),
    ):
        async with app.run_test(size=(120, 30)) as pilot:
            screen = ConfigFormScreen()
            await app.push_screen(screen)
            await pilot.pause()
            screen._fetch_recipe("org/model", "org/model")
            await app.workers.wait_for_complete()
            await pilot.pause()
            status = screen.query_one("#recipe-status", Static)

    assert "Could not read GPU memory" in str(status.render())


def test_vllm_config_round_trip_preserves_yaml_types(tmp_path: Path) -> None:
    from tui.backends.vllm import backend_common, backend_storage

    path = tmp_path / "typed.yaml"
    path.write_text(
        "model: org/model\n"
        "gpu-memory-utilization: 0.85\n"
        "enforce-eager: true\n"
        "compilation-config: [1, 2]\n"
    )
    with patch.object(backend_common, "CONFIG_DIR", tmp_path), patch.object(
        backend_storage, "CONFIG_DIR", tmp_path
    ):
        config = backend_storage.load_config("typed")
        backend_storage.save_config(config)

    saved = yaml.safe_load(path.read_text())
    assert saved["gpu-memory-utilization"] == 0.85
    assert type(saved["gpu-memory-utilization"]) is float
    assert saved["enforce-eager"] is True
    assert saved["compilation-config"] == [1, 2]


def test_recipe_merge_reenables_disabled_key(tmp_path: Path) -> None:
    from tui.backends.vllm import backend_common, backend_storage
    from tui.cli import config as cli_config

    path = tmp_path / "target.yaml"
    path.write_text(
        "model: org/model\n"
        "gpu-memory-utilization: 0.9\n"
        "# llmux:disabled reasoning-parser: old\n"
    )
    recipe = _recipe()
    with patch.object(cli_config, "_config_dir", lambda _backend: tmp_path), patch.object(
        backend_common, "CONFIG_DIR", tmp_path
    ), patch.object(backend_storage, "CONFIG_DIR", tmp_path), patch(
        "tui.common.recipes.fetch_recipe", AsyncMock(return_value=recipe)
    ):
        cli_config.config_from_recipe(
            "org/model",
            recipe_from="",
            variant="small",
            feature=["reasoning"],
            name="target",
            list_only=False,
            json_out=False,
            overwrite=False,
            merge=True,
        )

    assert "reasoning-parser" not in backend_storage.load_config("target").disabled_params
    assert "# llmux:disabled reasoning-parser:" not in path.read_text()
    assert yaml.safe_load(path.read_text())["reasoning-parser"] == "x"


@pytest.mark.asyncio
async def test_vllm_flag_cache_rejects_string_root(tmp_path: Path) -> None:
    from tui.backends.vllm import backend_inspect

    image = "custom/vllm:v1"
    identity = "sha256:one"
    key = hashlib.sha256(f"full-help-v2:{image}@{identity}".encode()).hexdigest()[:16]
    (tmp_path / f".vllm-params-{key}.json").write_text(json.dumps("max-model-len"))
    with patch.object(backend_inspect, "_VLLM_PARAMS_CACHE_DIR", tmp_path), patch(
        "tui.common.docker.image_identity", AsyncMock(return_value=identity)
    ):
        with pytest.raises(RuntimeError, match="invalid vLLM flag cache"):
            await backend_inspect.extract_vllm_params(image)


@pytest.mark.asyncio
async def test_vllm_flags_are_discovered_from_image_help(tmp_path: Path) -> None:
    from tui.backends.vllm import backend_inspect

    calls: list[tuple[str, ...]] = []

    async def run_command(*args: str, **_kwargs):
        calls.append(args)
        return 0, "usage: vllm serve [--dynamic-flag VALUE]\n  --another-flag TEXT\n"

    with patch.object(backend_inspect, "_VLLM_PARAMS_CACHE_DIR", tmp_path), patch.object(
        backend_inspect, "run_command", run_command
    ), patch("tui.common.docker.image_identity", AsyncMock(return_value=None)):
        flags = await backend_inspect.extract_vllm_params("custom/vllm:v1")

    assert flags == {"dynamic-flag", "another-flag"}
    assert calls == [
        (
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--entrypoint",
            "vllm",
            "custom/vllm:v1",
            "serve",
            "--help",
        )
    ]


def test_vllm_cascade_uses_effective_config_link(tmp_path: Path) -> None:
    from tui.backends.vllm import backend_storage
    from tui.common import profile_store

    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        "version: 1\ndefaults: {}\nprofiles:\n"
        "- name: profile-name\n  backend: vllm\n"
    )
    runtime = tmp_path / ".runtime"
    config = tmp_path / "profile-name.yaml"
    config.write_text("model: org/model\n")
    with patch.object(backend_storage, "CONFIG_DIR", tmp_path), patch.object(
        profile_store, "PROFILES_YAML", profiles
    ), patch.object(profile_store, "RUNTIME_DIR", runtime):
        backend_storage.delete_profile("profile-name", delete_config=True)

    assert not config.exists()
    with patch.object(profile_store, "PROFILES_YAML", profiles):
        assert profile_store.load_profile("profile-name", "vllm") is None


def _profile_storage(tmp_path: Path, env_value: str = "ok") -> tuple[Path, Path, Path]:
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "defaults": {},
                "profiles": [
                    {
                        "name": "profile",
                        "backend": "vllm",
                        "config_name": "config",
                        "env_vars": {"CUSTOM": env_value},
                    }
                ],
            },
            sort_keys=False,
        )
    )
    runtime = tmp_path / ".runtime"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("model: org/model\n")
    return profiles, runtime, config_dir


def test_config_rename_validation_failure_leaves_every_file_unchanged(tmp_path: Path) -> None:
    from tui.common import config_store, profile_store

    profiles, runtime, config_dir = _profile_storage(tmp_path, "it's invalid")
    original_profiles = profiles.read_text()
    with patch.object(profile_store, "PROFILES_YAML", profiles), patch.object(
        profile_store, "RUNTIME_DIR", runtime
    ), patch.object(config_store, "config_dir", lambda _backend: config_dir):
        with pytest.raises(ValueError, match="single quote"):
            config_store.rename_config(
                "vllm",
                "config",
                "renamed",
                replacement_text="model: org/edited\n",
            )

    assert (config_dir / "config.yaml").read_text() == "model: org/model\n"
    assert not (config_dir / "renamed.yaml").exists()
    assert profiles.read_text() == original_profiles


def test_config_delete_and_reference_clear_are_one_transaction(tmp_path: Path) -> None:
    from tui.common import config_store, profile_store

    profiles, runtime, config_dir = _profile_storage(tmp_path)
    with patch.object(profile_store, "PROFILES_YAML", profiles), patch.object(
        profile_store, "RUNTIME_DIR", runtime
    ), patch.object(config_store, "config_dir", lambda _backend: config_dir):
        changed = config_store.delete_config("vllm", "config")

    assert changed == ["profile"]
    assert not (config_dir / "config.yaml").exists()
    with patch.object(profile_store, "PROFILES_YAML", profiles):
        stored = profile_store.load_profile("profile", "vllm")
    assert stored is not None
    assert stored.config_name == ""
    assert "CONFIG_NAME=profile" in (runtime / "vllm" / "profile.env").read_text()


def test_config_rename_rechecks_source_after_lock_acquisition(tmp_path: Path) -> None:
    from tui.common import config_store, profile_store

    profiles, runtime, config_dir = _profile_storage(tmp_path)
    source = config_dir / "config.yaml"

    @contextmanager
    def concurrent_delete():
        source.unlink()
        yield

    with patch.object(profile_store, "PROFILES_YAML", profiles), patch.object(
        profile_store, "RUNTIME_DIR", runtime
    ), patch.object(config_store, "config_dir", lambda _backend: config_dir), patch.object(
        profile_store, "_storage_lock", concurrent_delete
    ):
        with pytest.raises(ValueError, match="config not found"):
            config_store.rename_config("vllm", "config", "renamed")

    assert not (config_dir / "renamed.yaml").exists()
