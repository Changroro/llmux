"""Memory estimate: per-profile GPU set, utilization budget and estimate inputs."""

from __future__ import annotations

import pytest

from tui.common import mem, profile_store
from tui.common.docker import GpuInfo


def _gpu(index: str, total_mib: str = "16384") -> GpuInfo:
    return GpuInfo(index=index, name="g", memory_used="0", memory_total=total_mib, utilization="0", temperature="0")


def test_fit_uses_only_the_profile_gpus_and_budget():
    gpus = [_gpu("0"), _gpu("1"), _gpu("2"), _gpu("3")]
    per_gpu, rows = mem.gpu_fit(20.0, gpus, gpu_ids=["0", "1"], utilization=0.5)
    assert per_gpu == 10.0
    assert [r["index"] for r in rows] == ["0", "1"]
    assert rows[0]["budget_gb"] == 8.0 and rows[0]["over"] is True


def test_fit_without_profile_keeps_all_gpus():
    per_gpu, rows = mem.gpu_fit(8.0, [_gpu("0"), _gpu("1")])
    assert per_gpu == 4.0 and len(rows) == 2 and rows[0]["over"] is False


def test_fit_rejects_unknown_profile_gpu():
    with pytest.raises(ValueError, match="GPU 7"):
        mem.gpu_fit(8.0, [_gpu("0")], gpu_ids=["7"])


def test_vllm_profile_request(monkeypatch, tmp_path):
    from tui.backends.vllm import backend_storage
    from tui.backends.vllm.backend_common import Config

    cfg = Config(
        name="c", model="org/m", gpu_memory_utilization="0.8",
        extra_params={"max-model-len": 8192, "kv-cache-dtype": "fp8", "revision": "r"},
    )
    monkeypatch.setattr(backend_storage, "load_config", lambda _n: cfg)
    stored = profile_store.StoredProfile(
        name="p", backend="vllm", gpu_id="0,1", tensor_parallel_size=2, config_name="c"
    )
    model, options, gpu_ids, util, split = mem.profile_estimate_request(stored)
    assert (model, gpu_ids, util, split) == ("org/m", ["0", "1"], 0.8, 2)
    assert options == {"max_model_len": 8192, "kv_cache_dtype": "fp8", "revision": "r"}


def test_llamacpp_profile_request_selects_the_profile_gguf(monkeypatch):
    from tui.backends.llamacpp import backend as lbackend

    cfg = lbackend.Config(name="c", params={"ctx-size": 32768, "cache-type-k": "q8_0"})
    monkeypatch.setattr(lbackend, "load_config", lambda _n: cfg)
    stored = profile_store.StoredProfile(
        name="p", backend="llamacpp", gpu_id="0", config_name="c",
        hf_repo="unsloth/X-GGUF", hf_file="X-Q4_K_M.gguf",
    )
    model, options, gpu_ids, util, split = mem.profile_estimate_request(stored)
    assert model == "unsloth/X-GGUF" and util == 1.0 and gpu_ids == ["0"] and split is None
    assert options == {"gguf_file": "X-Q4_K_M.gguf", "max_model_len": 32768, "kv_cache_dtype": "Q8_0"}


@pytest.mark.asyncio
async def test_estimate_passes_options_to_hf_mem(monkeypatch):
    import sys
    from types import SimpleNamespace

    seen: dict = {}

    async def arun(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(memory=4 * 1024**3, kv_cache=1024**3, total_memory=None)

    monkeypatch.setitem(sys.modules, "hf_mem", SimpleNamespace(arun=arun))
    result = await mem.estimate_model_memory("org/m", hf_token="", max_model_len=4096)
    assert result.startswith("~5.0GB")
    assert seen["max_model_len"] == 4096


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(32768, 32768), ("32768", 32768), ("32k", 32000), ("32K", 32768), ("25.6k", 25600), ("1M", 1048576), ("auto", None), (-1, None), ("-1", None)],
)
def test_vllm_max_model_len_accepts_vllm_formats(monkeypatch, raw, expected):
    from tui.backends.vllm import backend_storage
    from tui.backends.vllm.backend_common import Config

    cfg = Config(name="c", model="org/m", extra_params={"max-model-len": raw})
    monkeypatch.setattr(backend_storage, "load_config", lambda _n: cfg)
    stored = profile_store.StoredProfile(name="p", backend="vllm", gpu_id="0", config_name="c")
    _model, options, _ids, _util, _split = mem.profile_estimate_request(stored)
    assert options.get("max_model_len") == expected


def test_vllm_max_model_len_rejects_decimal_binary_suffix(monkeypatch):
    from tui.backends.vllm import backend_storage
    from tui.backends.vllm.backend_common import Config

    cfg = Config(name="c", model="org/m", extra_params={"max-model-len": "1.5K"})
    monkeypatch.setattr(backend_storage, "load_config", lambda _n: cfg)
    stored = profile_store.StoredProfile(name="p", backend="vllm", gpu_id="0", config_name="c")
    with pytest.raises(ValueError, match="max-model-len"):
        mem.profile_estimate_request(stored)


def test_llamacpp_pending_quant_must_be_resolved_first(monkeypatch):
    from tui.backends.llamacpp import backend as lbackend

    monkeypatch.setattr(lbackend, "load_config", lambda _n: lbackend.Config(name="c", params={}))
    stored = profile_store.StoredProfile(
        name="p", backend="llamacpp", config_name="c", hf_repo="unsloth/X-GGUF:Q4_K_M",
    )
    with pytest.raises(ValueError, match="llmux prepare p"):
        mem.profile_estimate_request(stored)
