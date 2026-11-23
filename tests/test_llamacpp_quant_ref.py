from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import yaml

from tui.backends.llamacpp import backend, backend_runtime
from tui.common import profile_store

FILES = [
    "X-Q4_K_M.gguf",
    "X-UD-Q4_K_XL.gguf",
    "Q8_0/X-Q8_0-00001-of-00002.gguf",
    "Q8_0/X-Q8_0-00002-of-00002.gguf",
    "mmproj-X-Q4_K_M.gguf",
]


@pytest.mark.parametrize(
    "quant, expected",
    [("Q4_K_M", "X-Q4_K_M.gguf"), ("q8_0", "Q8_0/X-Q8_0-00001-of-00002.gguf"), ("UD-Q4_K_XL", "X-UD-Q4_K_XL.gguf")],
)
def test_quant_tag_picks_the_entry_gguf(quant: str, expected: str) -> None:
    assert backend.pick_gguf_for_quant("o/X", quant, FILES) == expected


def test_unknown_or_ambiguous_quant_is_an_error() -> None:
    with pytest.raises(ValueError, match="Q2_K"):
        backend.pick_gguf_for_quant("o/X", "Q2_K", FILES)
    with pytest.raises(ValueError, match="ambiguous"):
        backend.pick_gguf_for_quant("o/X", "Q4_K_M", [*FILES, "X-Q4_K_M-imat.gguf"])


@pytest.mark.asyncio
async def test_resolve_source_resolves_quant_when_hf_is_reachable() -> None:
    with patch.object(backend, "list_repo_ggufs", AsyncMock(return_value=FILES)):
        assert await backend.resolve_gguf_source("o/X:Q4_K_M", "") == ("o/X", "X-Q4_K_M.gguf", "")


@pytest.mark.asyncio
async def test_resolve_source_keeps_quant_pending_when_offline() -> None:
    offline = AsyncMock(side_effect=backend.HfListingUnavailable("offline"))
    with patch.object(backend, "list_repo_ggufs", offline):
        repo, hf_file, warning = await backend.resolve_gguf_source("o/X:Q4_K_M", "")
    assert (repo, hf_file) == ("o/X:Q4_K_M", "")
    assert "prepare" in warning


@pytest.mark.asyncio
async def test_resolve_source_rejects_quant_plus_file() -> None:
    with pytest.raises(ValueError, match="either"):
        await backend.resolve_gguf_source("o/X:Q4_K_M", "X-Q4_K_M.gguf")


def test_pending_quant_is_a_valid_profile() -> None:
    profile_store.PROFILES_YAML.write_text(yaml.safe_dump({
        "version": 1, "defaults": {},
        "profiles": [{"name": "q", "backend": "llamacpp", "hf_repo": "o/X:Q4_K_M"}],
    }))
    assert profile_store.load_profile("q", "llamacpp").hf_repo == "o/X:Q4_K_M"


@pytest.mark.asyncio
async def test_pending_quant_is_resolved_and_stored_before_render() -> None:
    profile_store.PROFILES_YAML.write_text(yaml.safe_dump({
        "version": 1, "defaults": {},
        "profiles": [{"name": "q", "backend": "llamacpp", "hf_repo": "o/X:Q4_K_M"}],
    }))
    with patch.object(backend, "list_repo_ggufs", AsyncMock(return_value=FILES)):
        message = await backend_runtime._resolve_pending_quant("q")
    stored = profile_store.load_profile("q", "llamacpp")
    assert (stored.hf_repo, stored.hf_file) == ("o/X", "X-Q4_K_M.gguf")
    assert "X-Q4_K_M.gguf" in message


@pytest.mark.asyncio
async def test_nothing_to_resolve_is_a_no_op() -> None:
    profile_store.PROFILES_YAML.write_text(yaml.safe_dump({
        "version": 1, "defaults": {},
        "profiles": [{"name": "q", "backend": "llamacpp", "hf_repo": "o/X", "hf_file": "a.gguf"}],
    }))
    listing = AsyncMock()
    with patch.object(backend, "list_repo_ggufs", listing):
        assert await backend_runtime._resolve_pending_quant("q") == ""
    listing.assert_not_awaited()


def test_gguf_entries_group_shards_and_flag_mmproj() -> None:
    entries = [
        {"type": "file", "path": "X-Q4_K_M.gguf", "size": 10},
        {"type": "file", "path": "Q8_0/X-Q8_0-00001-of-00002.gguf", "lfs": {"size": 5}, "size": 1},
        {"type": "file", "path": "Q8_0/X-Q8_0-00002-of-00002.gguf", "lfs": {"size": 7}, "size": 1},
        {"type": "file", "path": "mmproj-F16.gguf", "size": 3},
        {"type": "file", "path": "README.md", "size": 1},
    ]
    assert backend.group_gguf_entries(entries) == [
        {"file": "Q8_0/X-Q8_0-00001-of-00002.gguf", "shards": 2, "size_bytes": 12, "mmproj": False},
        {"file": "X-Q4_K_M.gguf", "shards": 1, "size_bytes": 10, "mmproj": False},
        {"file": "mmproj-F16.gguf", "shards": 1, "size_bytes": 3, "mmproj": True},
    ]


@pytest.mark.asyncio
async def test_tui_quick_setup_preselects_the_quant_and_hides_shards_and_mmproj() -> None:
    from textual.app import App
    from textual.widgets import Input, Select

    from tui.backends.llamacpp.screens import quick_setup as module

    listing = [{"type": "file", "path": p, "size": 1} for p in FILES]
    app = App()
    screen = module.QuickSetupScreen()
    with patch.object(module, "list_hf_repo_files", AsyncMock(return_value=listing)), \
            patch.object(module, "list_config_names", return_value=[]):
        async with app.run_test(size=(120, 60)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#repo-input", Input).value = "o/X:Q8_0"
            screen.query_one("#fetch-btn").press()
            await pilot.pause()
            await pilot.pause()
            select = screen.query_one("#gguf-select", Select)
            values = [value for _label, value in select._options]
            assert select.value == "Q8_0/X-Q8_0-00001-of-00002.gguf"
    assert "Q8_0/X-Q8_0-00002-of-00002.gguf" not in values
    assert "mmproj-X-Q4_K_M.gguf" not in values


INKLING = [
    "Inkling-2-Mini-UD-Q2_K_XL.gguf",
    "Mainline_Support/Inkling-2-Mini-UD-Q2_K_XL.gguf_file",
    "Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file",
    "mmproj-F16.gguf",
]


def test_gguf_file_extension_is_a_model_file() -> None:
    # unsloth ships mainline-llama.cpp variants as `.gguf_file`; -hff accepts any repo path.
    rows = backend.group_gguf_entries([{"type": "file", "path": p, "size": 1} for p in INKLING])
    assert "Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file" in [r["file"] for r in rows]
    assert backend.gguf_choice_error("u/I", INKLING[1], INKLING) == ""
    assert backend.pick_gguf_for_quant("u/I", "Q8_K_XL", INKLING) == INKLING[2]
    with pytest.raises(ValueError, match="ambiguous"):  # root .gguf and the .gguf_file twin
        backend.pick_gguf_for_quant("u/I", "Q2_K_XL", INKLING)


@pytest.mark.asyncio
async def test_gguf_file_listing_counts_for_validation() -> None:
    listing = [{"type": "file", "path": p} for p in INKLING]
    with patch.object(backend, "list_hf_repo_files", AsyncMock(return_value=listing)):
        assert INKLING[1] in await backend.list_repo_ggufs("u/I")
        assert await backend.resolve_gguf_source("u/I", INKLING[1]) == ("u/I", INKLING[1], "")


def test_cached_gguf_file_shows_in_the_disk_inventory(tmp_path) -> None:
    snap = tmp_path / "hub" / "models--u--I" / "snapshots" / "r1" / "Mainline_Support"
    snap.mkdir(parents=True)
    (snap / "Inkling-2-Mini-UD-Q2_K_XL.gguf_file").write_bytes(b"x" * 10)
    (tmp_path / ".env.common").write_text(f"HF_CACHE_PATH={tmp_path}\n")
    with patch.object(backend, "ROOT", tmp_path):
        names = [e["name"] for e in backend.list_cached_gguf()]
    assert names == ["Inkling-2-Mini-UD-Q2_K_XL.gguf_file"]


@pytest.mark.asyncio
async def test_tui_quick_setup_refuses_a_quant_changed_after_fetch(tmp_path) -> None:
    from unittest.mock import MagicMock

    from textual.app import App
    from textual.widgets import Input

    from tui.backends.llamacpp.screens import quick_setup as module

    listing = [{"type": "file", "path": p, "size": 1} for p in FILES]
    app = App()
    screen = module.QuickSetupScreen()
    with patch.object(module, "list_hf_repo_files", AsyncMock(return_value=listing)), \
            patch.object(module, "list_config_names", return_value=[]), \
            patch.object(backend, "CONFIG_DIR", tmp_path), patch.object(module, "CONFIG_DIR", tmp_path):
        async with app.run_test(size=(120, 60)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#repo-input", Input).value = "o/X:Q4_K_M"
            screen.query_one("#fetch-btn").press()
            await pilot.pause()
            await pilot.pause()
            screen.query_one("#repo-input", Input).value = "o/X:Q8_0"
            screen.notify = MagicMock()
            screen.on_create()
            message = screen.notify.call_args.args[0]
    assert "fetch" in message.lower()
    assert profile_store.list_profiles("llamacpp") == [] and list(tmp_path.iterdir()) == []


def test_default_quant_follows_llama_cpp() -> None:
    # upstream find_best_model: Q4_K_M, then Q8_0, then the first model; first match wins.
    assert backend.pick_gguf_for_quant("o/d", "", ["d-F16.gguf", "d-Q8_0.gguf"]) == "d-Q8_0.gguf"
    assert backend.pick_gguf_for_quant("o/d", "", ["a-Q4_K_M.gguf", "b-Q4_K_M.gguf"]) == "a-Q4_K_M.gguf"
    assert backend.pick_gguf_for_quant("o/d", "", ["d-Q8_0.gguf", "d-Q4_K_M.gguf"]) == "d-Q4_K_M.gguf"
    assert backend.pick_gguf_for_quant("o/d", "", ["mmproj-F16.gguf", "d-Q8_0.gguf"]) == "d-Q8_0.gguf"
