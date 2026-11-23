from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from tui.backends.llamacpp import backend
from tui.common import profile_store

CONTAINER_CACHE = "/root/.cache/huggingface"


def _renderer():
    script = Path(__file__).resolve().parents[1] / "scripts" / "llamacpp" / "render-override.py"
    spec = importlib.util.spec_from_file_location("lora_renderer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lora_modules_parse_hf_and_container_sources() -> None:
    entries = backend.parse_lora_modules("tone=org/ad-GGUF/sub/tone.gguf:0.5, /root/x.gguf")
    assert [(e.name, e.hf_repo, e.hf_file, e.path, e.scale) for e in entries] == [
        ("tone", "org/ad-GGUF", "sub/tone.gguf", "", "0.5"),
        ("", "", "", "/root/x.gguf", ""),
    ]


@pytest.mark.parametrize(
    "bad", ["org/only.gguf", "org/r/ad.bin", "org/r/ad.gguf:big", "x=", "../r/a.gguf"]
)
def test_bad_lora_modules_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        backend.parse_lora_modules(bad)


def test_llamacpp_profile_stores_lora_fields() -> None:
    profile_store.create_profile(profile_store.StoredProfile(
        name="l", backend="llamacpp", hf_repo="o/r", hf_file="m.gguf",
        enable_lora=True, lora_modules="org/ad/a.gguf:0.7",
    ))
    raw = yaml.safe_load(profile_store.PROFILES_YAML.read_text())["profiles"][0]
    assert (raw["enable_lora"], raw["lora_modules"]) == (True, "org/ad/a.gguf:0.7")
    loaded = profile_store.load_profile("l", "llamacpp")
    assert (loaded.enable_lora, loaded.lora_modules) == (True, "org/ad/a.gguf:0.7")


def test_invalid_llamacpp_lora_modules_fail_validation() -> None:
    with pytest.raises(ValueError, match="lora"):
        profile_store.create_profile(profile_store.StoredProfile(
            name="l", backend="llamacpp", hf_repo="o/r", hf_file="m.gguf",
            enable_lora=True, lora_modules="org/ad.bin",
        ))


def test_cached_hf_adapter_maps_to_its_container_path(tmp_path: Path) -> None:
    snap = tmp_path / "hub" / "models--org--ad" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "a.gguf").write_text("x")
    stored = profile_store.StoredProfile(
        name="l", backend="llamacpp", enable_lora=True, lora_modules="org/ad/a.gguf:0.7,/root/b.gguf",
    )
    assert backend.lora_args(stored, str(tmp_path)) == [
        ("--lora-scaled", f"{CONTAINER_CACHE}/hub/models--org--ad/snapshots/rev1/a.gguf:0.7"),
        ("--lora", "/root/b.gguf"),
    ]


def test_missing_hf_adapter_asks_for_prepare(tmp_path: Path) -> None:
    stored = profile_store.StoredProfile(
        name="l", backend="llamacpp", enable_lora=True, lora_modules="org/ad/a.gguf",
    )
    with pytest.raises(ValueError, match="llmux prepare"):
        backend.lora_args(stored, str(tmp_path))


def test_disabled_lora_renders_nothing(tmp_path: Path) -> None:
    stored = profile_store.StoredProfile(
        name="l", backend="llamacpp", enable_lora=False, lora_modules="org/ad/a.gguf",
    )
    assert backend.lora_args(stored, str(tmp_path)) == []


def test_renderer_appends_lora_args() -> None:
    command = _renderer().render_command(
        {}, hf_repo="o/r", hf_file="m.gguf",
        lora=[("--lora", "/a.gguf"), ("--lora-scaled", "/b.gguf:0.5")],
    )
    assert command[-4:] == ["--lora", "/a.gguf", "--lora-scaled", "/b.gguf:0.5"]


def test_tui_profile_roundtrip_keeps_lora() -> None:
    profile_store.create_profile(profile_store.StoredProfile(
        name="l", backend="llamacpp", hf_repo="o/r", hf_file="m.gguf",
        enable_lora=True, lora_modules="org/ad/a.gguf",
    ))
    p = backend.load_profile("l")
    p.port = 8099
    backend.save_profile(p)
    stored = profile_store.load_profile("l", "llamacpp")
    assert (stored.port, stored.enable_lora, stored.lora_modules) == (8099, True, "org/ad/a.gguf")
