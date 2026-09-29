from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tui.backends.llamacpp import backend as llama_backend
from tui.backends.llamacpp import backend_runtime as llama_runtime
from tui.backends.vllm import backend_common as vllm_common
from tui.backends.vllm import backend_runtime as vllm_runtime
from tui.cli import _runtime as cli_runtime
from tui.common import profile_store


def _load_llama_renderer():
    path = Path(__file__).parents[1] / "scripts" / "llamacpp" / "render-override.py"
    spec = importlib.util.spec_from_file_location("audit_r3_llama_renderer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stream_async_rejects_missing_terminal_rc(capsys):
    async def incomplete_stream():
        yield ("log", "partial output")

    rc = cli_runtime.stream_async(incomplete_stream())

    captured = capsys.readouterr()
    assert rc != 0
    assert "protocol error" in captured.err.lower()
    assert "terminal rc" in captured.err.lower()


def test_stream_async_rejects_nonterminal_rc(capsys):
    async def malformed_stream():
        yield ("rc", 0)
        yield ("log", "late output")

    rc = cli_runtime.stream_async(malformed_stream())

    captured = capsys.readouterr()
    assert rc != 0
    assert "protocol error" in captured.err.lower()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"model": {"org": "repo"}}, "model must be a non-empty string"),
        (
            {"model": "org/repo", "gpu-memory-utilization": {"bad": "value"}},
            "gpu-memory-utilization must be a number",
        ),
        (
            {"model": "org/repo", "gpu-memory-utilization": True},
            "gpu-memory-utilization must be a number",
        ),
        (
            {"model": "org/repo", "gpu-memory-utilization": 1.1},
            "gpu-memory-utilization must be greater than 0 and at most 1",
        ),
    ],
)
def test_vllm_runtime_rejects_invalid_core_config_types(data, message):
    with pytest.raises(ValueError, match=message):
        vllm_runtime._validate_core_config_data(data)


@pytest.mark.parametrize(
    "value",
    [
        {"nested": 1},
        ["valid", {"nested": 1}],
        [["nested"]],
    ],
)
def test_llama_renderer_rejects_mapping_and_nested_values(value):
    renderer = _load_llama_renderer()

    with pytest.raises(ValueError, match="ctx-size.*scalar"):
        renderer.render_command(
            {"ctx-size": value},
            hf_repo="org/repo",
            hf_file="model.gguf",
        )


def test_vllm_auto_repair_reloads_latest_config_inside_transaction(
    monkeypatch, tmp_path
):
    stale = vllm_common.Profile(
        name="race",
        config_name="race",
        model_id="org/stale-model",
    )
    latest = vllm_common.Profile(
        name="race",
        config_name="race",
        model_id="org/latest-model",
    )
    config = vllm_common.Config(
        name="race",
        model="your-org/your-model",
        extra_params={"external-only": True},
    )
    monkeypatch.setattr(vllm_common, "CONFIG_DIR", tmp_path)
    config.path.write_text(
        "model: your-org/your-model\nexternal-only: true\n"
    )
    in_transaction = False

    @contextmanager
    def transaction():
        nonlocal in_transaction
        in_transaction = True
        try:
            yield
        finally:
            in_transaction = False

    saved = []

    def load_latest(name):
        assert name == "race"
        assert in_transaction
        return latest

    def load_config(name):
        assert name == "race"
        assert in_transaction
        return config

    def save_config(value):
        assert in_transaction
        saved.append(value)

    monkeypatch.setattr(profile_store, "storage_transaction", transaction)
    monkeypatch.setattr(vllm_runtime, "load_profile", load_latest)
    monkeypatch.setattr(vllm_runtime, "load_config", load_config)
    monkeypatch.setattr(vllm_runtime, "save_config", save_config)

    ok, messages = vllm_runtime._ensure_profile_config(stale)

    assert ok is True
    assert messages
    assert saved[0].model == "org/latest-model"
    assert saved[0].extra_params == {"external-only": True}


def test_llama_auto_link_reloads_latest_profile_inside_transaction(
    monkeypatch, tmp_path
):
    stale_stored = profile_store.StoredProfile(name="race", backend="llamacpp")
    stale_profile = llama_backend.Profile(name="race")
    latest_stored = profile_store.StoredProfile(
        name="race",
        backend="llamacpp",
        config_name="edited",
    )
    latest_profile = llama_backend.Profile(name="race", config_name="edited")
    (tmp_path / "edited.yaml").write_text("ctx-size: 4096\n")
    monkeypatch.setattr(llama_runtime, "CONFIG_DIR", tmp_path)
    in_transaction = False

    @contextmanager
    def transaction():
        nonlocal in_transaction
        in_transaction = True
        try:
            yield
        finally:
            in_transaction = False

    def load_stored(name, backend):
        assert (name, backend) == ("race", "llamacpp")
        assert in_transaction
        return latest_stored

    def load_profile(name):
        assert name == "race"
        assert in_transaction
        return latest_profile

    monkeypatch.setattr(profile_store, "storage_transaction", transaction)
    monkeypatch.setattr(profile_store, "load_profile", load_stored)
    monkeypatch.setattr(llama_runtime, "load_profile", load_profile)
    monkeypatch.setattr(
        llama_runtime,
        "load_config",
        lambda name: llama_backend.Config(name=name, params={"ctx-size": 4096}),
    )
    monkeypatch.setattr(
        profile_store,
        "save_profile",
        lambda _profile: pytest.fail("latest config link must not be overwritten"),
    )

    ok, messages = llama_runtime._ensure_profile_config(stale_stored, stale_profile)

    assert ok is True
    assert all("config 미링크" not in message for message in messages)


@pytest.mark.parametrize(
    ("runtime", "backend", "profile_factory"),
    [
        (
            vllm_runtime,
            "vllm",
            lambda: vllm_common.Profile(name="snap", config_name="new-config"),
        ),
        (
            llama_runtime,
            "llamacpp",
            lambda: llama_backend.Profile(name="snap", config_name="new-config"),
        ),
    ],
)
def test_runtime_snapshot_renders_and_returns_the_same_latest_profile(
    monkeypatch, tmp_path, runtime, backend, profile_factory
):
    stored = profile_store.StoredProfile(
        name="snap",
        backend=backend,
        config_name="new-config",
    )
    latest = profile_factory()
    rendered = []

    @contextmanager
    def transaction():
        yield

    monkeypatch.setattr(profile_store, "storage_transaction", transaction)
    monkeypatch.setattr(
        profile_store,
        "load_profile",
        lambda name, selected: stored
        if (name, selected) == ("snap", backend)
        else None,
    )
    monkeypatch.setattr(runtime, "load_profile", lambda name: latest)
    monkeypatch.setattr(
        profile_store,
        "render_env",
        lambda value: rendered.append(value) or (tmp_path / f"{backend}.env"),
    )

    profile, path = runtime._render_profile_snapshot("snap")

    assert profile is latest
    assert rendered == [stored]
    assert path == tmp_path / f"{backend}.env"


def test_vllm_compose_env_uses_rendered_config_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(vllm_common, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(vllm_runtime, "_common_env", lambda **kwargs: {})
    stale = vllm_common.Profile(name="snap", config_name="old-config")
    stale.path.write_text(
        "CONTAINER_NAME=snap\n"
        "VLLM_PORT=8000\n"
        "GPU_ID=0\n"
        "TENSOR_PARALLEL_SIZE=1\n"
        "CONFIG_NAME=new-config\n"
        "ENABLE_LORA=false\n"
    )

    env = vllm_runtime._compose_env(stale, use_dev=False)

    assert env["CONFIG_NAME"] == "new-config"


def test_llama_renderer_writes_env_and_override_from_one_locked_snapshot(
    monkeypatch, tmp_path
):
    renderer = _load_llama_renderer()
    stored = profile_store.StoredProfile(
        name="snap",
        backend="llamacpp",
        config_name="cfg",
        hf_repo="org/repo",
        hf_file="model.gguf",
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cfg.yaml").write_text("ctx-size: 4096\n")
    monkeypatch.setattr(renderer, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(renderer, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(renderer.profile_store, "load_profile", lambda *_args: stored)
    rendered = []
    monkeypatch.setattr(
        renderer.profile_store,
        "render_env",
        lambda value: rendered.append(value) or tmp_path / "snap.env",
    )
    monkeypatch.setattr(renderer.sys, "argv", ["render-override.py", "snap"])

    assert renderer.main() == 0
    assert rendered == [stored]


@pytest.mark.asyncio
async def test_llama_prepare_downloads_latest_rendered_profile(
    monkeypatch,
    tmp_path,
):
    stale = llama_backend.Profile(
        name="race",
        container_name="race",
        hf_repo="org/old",
        hf_file="old.gguf",
    )
    latest = llama_backend.Profile(
        name="race",
        container_name="race",
        hf_repo="org/new",
        hf_file="new.gguf",
    )
    downloaded = []

    async def download(**kwargs):
        downloaded.append(kwargs)
        yield "rc", 0

    monkeypatch.setattr(
        profile_store,
        "load_profile",
        lambda *_args: profile_store.StoredProfile(
            name="race",
            backend="llamacpp",
            hf_repo="org/old",
            hf_file="old.gguf",
        ),
    )
    monkeypatch.setattr(
        llama_runtime,
        "validate_common_env",
        lambda *_args, **_kwargs: (True, []),
    )
    monkeypatch.setattr(llama_runtime, "load_profile", lambda _name: stale)
    monkeypatch.setattr(llama_runtime, "_ensure_profile_config", lambda *_args: (True, []))
    monkeypatch.setattr(
        llama_runtime,
        "_render_profile_snapshot",
        lambda _name: (latest, tmp_path / "race.env"),
    )
    monkeypatch.setattr(
        llama_runtime,
        "_render_override",
        AsyncMock(return_value=(0, "")),
    )
    monkeypatch.setattr(
        llama_runtime,
        "_resolve_runtime_image",
        lambda _profile: "repo/image:v1",
    )
    monkeypatch.setattr(
        llama_runtime.prepare,
        "image_present",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        llama_runtime.prepare,
        "hf_cache_path",
        lambda: str(tmp_path / "cache"),
    )
    monkeypatch.setattr(llama_runtime.prepare, "hf_token", lambda: "")
    monkeypatch.setattr(llama_runtime.prepare, "stream_llamacpp_download", download)

    events = [
        event async for event in llama_runtime.stream_container_prepare("race")
    ]

    assert events[-1] == ("rc", 0)
    assert downloaded[0]["hf_repo"] == "org/new"
    assert downloaded[0]["hf_file"] == "new.gguf"
