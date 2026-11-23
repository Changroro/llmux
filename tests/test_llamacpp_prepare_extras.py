from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tui.backends.llamacpp import backend, backend_runtime as lrt
from tui.common import profile_store as ps

MMPROJ = "/root/.cache/huggingface/hub/models--o--r/snapshots/abc/mmproj-F16.gguf"


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"spec-draft-hf": "o/d:Q8_0"}, [("draft", "o/d", "Q8_0", "")]),
        ({"extra-args": ["-hfd", "o/d:Q8_0", "--spec-type", "draft-mtp"]}, [("draft", "o/d", "Q8_0", "")]),
        ({"extra-args": "--hf-repo-draft=o/d --fit off"}, [("draft", "o/d", "", "")]),
        ({"mmproj": MMPROJ}, [("mmproj", "o/r", "", "mmproj-F16.gguf")]),
        ({"mmproj": "/models/mmproj.gguf"}, [("mmproj", "", "", "")]),
        ({"ctx-size": 1, "extra-args": []}, []),
    ],
)
def test_prepare_extras_reads_draft_and_mmproj_flags(params: dict, expected: list) -> None:
    assert [
        (e["kind"], e["repo"], e["quant"], e["file"]) for e in lrt._prepare_extras(params)
    ] == expected


def test_draft_quant_takes_the_first_match_like_upstream() -> None:
    assert backend.pick_gguf_for_quant("o/d", "Q8_0", ["a-Q8_0.gguf", "b-Q8_0.gguf"], strict=False) == "a-Q8_0.gguf"

async def _drain(stream):
    logs, rc = [], None
    async for event in stream:
        if event[0] == "rc":
            rc = event[1]
        else:
            logs.append(event[1])
    return logs, rc


async def _prepare(tmp: Path, params: dict, downloads: list, repo_files=("m.gguf",), render=None, **lora):
    stored = ps.StoredProfile(
        name="p", backend="llamacpp", container_name="p", config_name="p",
        hf_repo="o/r", hf_file="m.gguf", **lora,
    )
    profile = lrt.Profile(name="p", container_name="p", config_name="p", hf_repo="o/r", hf_file="m.gguf")

    async def download(*, hf_repo, hf_file, cache_path, **_kw):
        downloads.append((hf_repo, hf_file))
        snap = Path(cache_path) / "hub" / f"models--{hf_repo.replace('/', '--')}" / "snapshots" / "abc"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / hf_file).write_text("x")
        yield ("rc", 0)

    with patch.object(lrt, "validate_common_env", lambda *a, **k: (True, [])), \
         patch.object(lrt.profile_store, "load_profile", lambda *a, **k: stored), \
         patch.object(lrt, "load_profile", lambda _n: profile), \
         patch.object(lrt, "_render_profile_snapshot", lambda _n: (profile, Path("p.env"))), \
         patch.object(lrt, "_ensure_profile_config", lambda *_a: (True, [])), \
         patch.object(lrt, "load_config", lambda _n: backend.Config(name="p", params=params)), \
         patch.object(lrt, "_render_override", render or AsyncMock(return_value=(0, ""))), \
         patch.object(lrt.prepare, "image_present", AsyncMock(return_value=True)), \
         patch.object(lrt.prepare, "hf_cache_path", lambda: str(tmp)), \
         patch.object(lrt.prepare, "hf_token", lambda: ""), \
         patch.object(lrt.prepare, "stream_llamacpp_download", download), \
         patch.object(backend, "list_repo_ggufs", AsyncMock(return_value=["d-Q4_K_M.gguf", "d-Q8_0.gguf"])), \
         patch.object(backend, "list_hf_repo_files", AsyncMock(return_value=[{"type": "file", "path": f} for f in repo_files])):
        return await _drain(lrt.stream_container_prepare("p"))


@pytest.mark.asyncio
async def test_prepare_downloads_draft_and_mmproj(tmp_path: Path) -> None:
    downloads: list = []
    logs, rc = await _prepare(
        tmp_path, {"extra-args": ["-hfd", "o/d:Q8_0"], "mmproj": MMPROJ}, downloads
    )
    assert rc == 0, logs
    assert downloads == [("o/r", "m.gguf"), ("o/r", "mmproj-F16.gguf"), ("o/d", "d-Q8_0.gguf")]


@pytest.mark.asyncio
async def test_stale_mmproj_snapshot_path_fails_with_the_right_path(tmp_path: Path) -> None:
    stale = MMPROJ.replace("/abc/", "/old/")
    logs, rc = await _prepare(tmp_path, {"mmproj": stale}, [])
    assert rc == 1
    assert any(MMPROJ in line for line in logs), logs


@pytest.mark.parametrize(
    "model, files, expected",
    [
        # repo root counts as a shared prefix; a .gguf_file model still pairs (upstream rule)
        ("Mainline_Support/I-UD-Q2_K_XL.gguf_file", ["mmproj-F16.gguf", "I-Q2_K_XL.gguf"], "mmproj-F16.gguf"),
        ("Q4/X-Q4_K_M.gguf", ["mmproj-F16.gguf", "Q4/mmproj-F16.gguf"], "Q4/mmproj-F16.gguf"),  # deeper wins
        ("X-Q4_K_M.gguf", ["mmproj-F16.gguf", "mmproj-Q8_0.gguf"], "mmproj-Q8_0.gguf"),  # closest bits
        ("X-Q4_K_M.gguf", ["other/mmproj-F16.gguf"], ""),  # not on the model's path
        ("X-Q4_K_M.gguf", ["X-Q8_0.gguf"], ""),
    ],
)
def test_mmproj_sibling_follows_upstream(model: str, files: list, expected: str) -> None:
    assert backend.find_best_sibling(files, model, "mmproj") == expected


@pytest.mark.parametrize(
    "params, wanted",
    [
        ({}, True),
        ({"mmproj": MMPROJ}, False),
        ({"no-mmproj": True}, False),
        ({"mmproj-auto": False}, False),
        ({"mmproj-url": "https://x/m.gguf"}, False),
        ({"extra-args": ["--no-mmproj-auto"]}, False),
        ({"extra-args": ["-mm", "/m.gguf"]}, False),
    ],
)
def test_auto_mmproj_only_when_config_leaves_it_on(params: dict, wanted: bool) -> None:
    assert lrt._wants_auto_mmproj(params) is wanted


@pytest.mark.asyncio
async def test_prepare_downloads_the_auto_mmproj(tmp_path: Path) -> None:
    downloads: list = []
    logs, rc = await _prepare(tmp_path, {}, downloads, repo_files=("m.gguf", "mmproj-F16.gguf"))
    assert rc == 0, logs
    assert downloads == [("o/r", "m.gguf"), ("o/r", "mmproj-F16.gguf")]


@pytest.mark.asyncio
async def test_disabled_auto_mmproj_is_not_downloaded(tmp_path: Path) -> None:
    downloads: list = []
    _, rc = await _prepare(tmp_path, {"no-mmproj": True}, downloads, repo_files=("m.gguf", "mmproj-F16.gguf"))
    assert rc == 0 and downloads == [("o/r", "m.gguf")]


def test_explicit_draft_file_wins_over_the_quant() -> None:
    extras = lrt._prepare_extras({"extra-args": ["-hfd", "o/d:Q8_0", "-md", "d-Q4_K_M.gguf"]})
    assert [(e["kind"], e["repo"], e["file"]) for e in extras] == [("draft", "o/d", "d-Q4_K_M.gguf")]
