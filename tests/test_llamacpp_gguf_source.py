from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tui.backends.llamacpp import backend
from tui.common import profile_store


@pytest.fixture
def cfg_dir(tmp_path: Path):
    path = tmp_path / "config"
    path.mkdir()
    with patch.object(backend, "CONFIG_DIR", path):
        yield path


def _write_profiles(entries: list[dict]) -> None:
    profile_store.PROFILES_YAML.write_text(
        yaml.safe_dump({"version": 1, "defaults": {}, "profiles": entries})
    )


def _raw_profile(name: str) -> dict:
    data = yaml.safe_load(profile_store.PROFILES_YAML.read_text())
    return next(p for p in data["profiles"] if p["name"] == name)


def test_profile_model_file_moves_to_hf_file(cfg_dir: Path) -> None:
    _write_profiles([{"name": "a", "backend": "llamacpp", "hf_repo": "o/r", "model_file": "a.gguf"}])
    (cfg_dir / "a.yaml").write_text("ctx-size: 1\n")

    messages = backend.migrate_gguf_sources()

    raw = _raw_profile("a")
    assert raw["hf_file"] == "a.gguf"
    assert "model_file" not in raw
    assert messages


def test_config_model_file_moves_to_profile_and_keeps_comments(cfg_dir: Path) -> None:
    _write_profiles([{"name": "a", "backend": "llamacpp", "hf_repo": "o/r", "model_file": "old.gguf"}])
    (cfg_dir / "a.yaml").write_text("# keep me\nmodel-file: cfg.gguf\nctx-size: 1  # ctx note\n")

    backend.migrate_gguf_sources()

    # Old precedence was config model-file over profile model_file.
    assert _raw_profile("a")["hf_file"] == "cfg.gguf"
    text = (cfg_dir / "a.yaml").read_text()
    assert "model-file" not in text
    assert "# keep me" in text and "# ctx note" in text


def test_existing_hf_file_wins_and_dead_config_key_is_dropped(cfg_dir: Path) -> None:
    _write_profiles([{"name": "a", "backend": "llamacpp", "hf_repo": "o/r", "hf_file": "hf.gguf"}])
    (cfg_dir / "a.yaml").write_text("model-file: cfg.gguf\nctx-size: 1\n")

    backend.migrate_gguf_sources()

    assert _raw_profile("a")["hf_file"] == "hf.gguf"
    assert "model-file" not in (cfg_dir / "a.yaml").read_text()


def test_migration_is_idempotent_and_writes_nothing_when_migrated(cfg_dir: Path) -> None:
    _write_profiles([{"name": "a", "backend": "llamacpp", "hf_repo": "o/r", "model_file": "a.gguf"}])
    (cfg_dir / "a.yaml").write_text("model-file: a.gguf\nctx-size: 1\n")
    assert backend.migrate_gguf_sources()
    files = [profile_store.PROFILES_YAML, cfg_dir / "a.yaml"]
    before = [(f.read_text(), f.stat().st_mtime_ns) for f in files]

    assert backend.migrate_gguf_sources() == []
    assert [(f.read_text(), f.stat().st_mtime_ns) for f in files] == before


def test_migration_without_profiles_file_is_a_no_op(cfg_dir: Path) -> None:
    assert backend.migrate_gguf_sources() == []


def test_renderer_uses_only_hf_file() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "llamacpp" / "render-override.py"
    spec = importlib.util.spec_from_file_location("gguf_source_renderer", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="model-file"):
        module.render_command({"model-file": "x.gguf"}, hf_repo="o/r", hf_file="m.gguf")
    with pytest.raises(ValueError, match="hf_file"):
        module.render_command({}, hf_repo="o/r")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("# top\nmodel-file: a\n\n# ctx block\nctx-size: 1  # note\n", "# top\n\n# ctx block\nctx-size: 1  # note\n"),
        ("alias: x  # al\nmodel-file: a  # gone\n\n# ctx block\nctx-size: 1\n", "alias: x  # al\n\n# ctx block\nctx-size: 1\n"),
        ("model-file: a\n# ctx block\nctx-size: 1\n", "# ctx block\nctx-size: 1\n"),
    ],
)
def test_removing_a_key_keeps_the_comment_below_it(text: str, expected: str) -> None:
    from tui.common.config_markers import dump_active_config

    assert dump_active_config(text, {"alias": "x", "ctx-size": 1} if "alias" in text else {"ctx-size": 1}) == expected


def test_gguf_file_extension_migrates_unchanged(cfg_dir: Path) -> None:
    legacy = "Mainline_Support/Inkling-2-Mini-UD-Q2_K_XL.gguf_file"
    _write_profiles([{"name": "i", "backend": "llamacpp", "hf_repo": "u/I", "model_file": legacy}])
    (cfg_dir / "i.yaml").write_text(f"model-file: {legacy}\nctx-size: 1\n")
    backend.migrate_gguf_sources()
    assert _raw_profile("i")["hf_file"] == legacy


_COMMENTED = """\
# my llmux profiles
version: 1
defaults: {}
profiles:
  # chat model
  - name: a
    backend: llamacpp
    hf_repo: o/r
    model_file: a.gguf  # legacy
    # enable_lora: true
    # env_vars:
    #   FOO: bar
  - name: b
    backend: llamacpp
    hf_repo: o/s
    hf_file: b.gguf
    model_file: b.gguf
    # image_tag: llamacpp-dev:x
"""


def test_stale_model_file_alone_writes_nothing(cfg_dir: Path) -> None:
    text = _COMMENTED.replace("model_file: a.gguf  # legacy", "hf_file: a.gguf")
    profile_store.PROFILES_YAML.write_text(text)
    for name in "ab":
        (cfg_dir / f"{name}.yaml").write_text("ctx-size: 1\n")

    assert backend.migrate_gguf_sources() == []
    assert profile_store.PROFILES_YAML.read_text() == text


def test_migration_keeps_comments_and_commented_out_keys(cfg_dir: Path) -> None:
    profile_store.PROFILES_YAML.write_text(_COMMENTED)
    for name in "ab":
        (cfg_dir / f"{name}.yaml").write_text("ctx-size: 1\n")

    assert backend.migrate_gguf_sources()

    text = profile_store.PROFILES_YAML.read_text()
    for kept in ("# my llmux profiles", "# chat model", "# enable_lora: true", "#   FOO: bar", "# image_tag: llamacpp-dev:x"):
        assert kept in text
    assert _raw_profile("a")["hf_file"] == "a.gguf" and "model_file" not in _raw_profile("a")
    assert "model_file: b.gguf" in text  # untouched entry stays byte-identical


def test_migration_is_all_or_nothing(cfg_dir: Path) -> None:
    _write_profiles([
        {"name": "a", "backend": "llamacpp", "hf_repo": "o/r", "model_file": "a.gguf"},
        {"name": "b", "backend": "llamacpp", "hf_repo": "o/s", "model_file": "b.gguf"},
    ])
    (cfg_dir / "a.yaml").write_text("model-file: a.gguf\nctx-size: 1\n")
    (cfg_dir / "b.yaml").write_text("model-file: [unclosed\n")
    before = [f.read_text() for f in (profile_store.PROFILES_YAML, cfg_dir / "a.yaml")]

    with pytest.raises(Exception):
        backend.migrate_gguf_sources()

    assert [f.read_text() for f in (profile_store.PROFILES_YAML, cfg_dir / "a.yaml")] == before


def test_saving_an_unmigrated_profile_keeps_model_file() -> None:
    _write_profiles([{"name": "a", "backend": "llamacpp", "hf_repo": "o/r", "model_file": "a.gguf"}])
    stored = profile_store.load_profile("a", "llamacpp")
    profile_store.save_profile(stored)
    assert _raw_profile("a")["model_file"] == "a.gguf"


def test_edit_fails_when_the_repo_changes_during_validation(monkeypatch) -> None:
    from typer.testing import CliRunner

    from tui.cli import app
    from tui.cli import profile as profile_cli

    _write_profiles([{"name": "a", "backend": "llamacpp", "hf_repo": "o/A", "hf_file": "a.gguf"}])

    def racing_validation(repo, hf_file):
        profile_store.update_profile("a", "llamacpp", lambda sp: setattr(sp, "hf_repo", "o/B"))
        return repo, hf_file

    monkeypatch.setattr(profile_cli, "_require_gguf_source", racing_validation)
    result = CliRunner().invoke(app, ["profile", "edit", "a", "--hf-file", "a2.gguf"])
    assert result.exit_code != 0
    assert (_raw_profile("a")["hf_repo"], _raw_profile("a")["hf_file"]) == ("o/B", "a.gguf")


def test_root_callback_skips_migration_writes_on_dry_run(monkeypatch) -> None:
    import sys
    from unittest.mock import Mock

    from tui.cli import _migrate_legacy_storage

    migrate = Mock(return_value=[])
    monkeypatch.setattr(backend, "migrate_gguf_sources", migrate)
    monkeypatch.setattr(sys, "argv", ["llmux", "up", "lp", "--dry-run"])
    _migrate_legacy_storage()
    migrate.assert_not_called()
    monkeypatch.setattr(sys, "argv", ["llmux", "up", "lp"])
    _migrate_legacy_storage()
    migrate.assert_called_once()
