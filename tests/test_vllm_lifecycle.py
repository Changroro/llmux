"""vLLM `up` readiness outcomes, timeouts and dry-run."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from tui.backends.vllm import backend_runtime as rt
from tui.common import exit_codes, profile_store


class _Store:
    @staticmethod
    def load_profile(name, backend_name):
        return profile_store.StoredProfile(name=name, backend=backend_name)

    @staticmethod
    @contextmanager
    def storage_transaction():
        yield


def _profile():
    return rt.Profile(name="p", container_name="p", port="8000", image_tag="vllm/vllm-openai:v0.20.1")


def _patches(validation, commands, **extra):
    async def compose(cmd, **_kwargs):
        commands.append(cmd)
        yield ("rc", 0)

    profile = _profile()
    values = {
        "load_profile": lambda _name: profile,
        "check_port_conflict": AsyncMock(return_value=None),
        "_ensure_common_env": lambda _profile: (True, []),
        "_ensure_profile_config": lambda _profile, **_kw: (True, []),
        "_render_profile_snapshot": lambda _name, **_k: (profile, "/unused"),
        "_gpu_conflict_messages": AsyncMock(return_value=[]),
        "_compose_env": lambda *_a, **_k: {},
        "_post_start_validation": validation,
        "_verify_vllm_version": _no_events,
        "profile_store": _Store,
        "stream_command": compose,
        **extra,
    }
    return patch.dict(rt.stream_container_up.__globals__, values)


async def _no_events(*_args, **_kwargs):
    if False:
        yield ("log", "")


async def _collect(**kwargs):
    return [event async for event in rt.stream_container_up("p", **kwargs)]


@pytest.mark.asyncio
async def test_crashed_container_is_removed_by_the_id_seen_at_detection():
    async def crashed(*_a, **_k):
        yield ("result", False, ["Error: container 'p' exited during startup (exited)."], "crashed", "abc123")

    run = AsyncMock(return_value=(0, "abc123"))
    down = AsyncMock()
    commands: list = []
    with _patches(crashed, commands, run_command=run, container_down=down):
        events = await _collect()
    run.assert_awaited_once_with("docker", "rm", "-f", "abc123", timeout=30)
    down.assert_not_awaited()
    assert events[-1] == ("rc", exit_codes.ERROR)


@pytest.mark.asyncio
async def test_validation_reports_the_crashed_container_id():
    profile = _profile()

    async def run(*args, **_kwargs):
        if args[:2] == ("docker", "inspect"):
            return 0, "exited\tnone\tabc123"
        return 0, "tail\n"

    with patch.object(rt, "run_command", side_effect=run), patch.object(
        rt, "_models_endpoint_ready", AsyncMock(return_value=False)
    ), patch.object(rt, "load_config", return_value=rt.Config(name="p", model="m")):
        events = [e async for e in rt._post_start_validation(profile, timeout=0.05, poll_interval=0.01)]
    assert events[-1][3:] == ("crashed", "abc123")


@pytest.mark.asyncio
async def test_stalled_container_is_left_running_with_not_ready_code():
    async def stalled(*_a, **_k):
        yield ("result", False, ["Error: container started but /v1/models is not ready within timeout."], "stalled")

    down = AsyncMock()
    commands: list = []
    with _patches(stalled, commands, container_down=down):
        events = await _collect()
    down.assert_not_awaited()
    assert events[-1] == ("rc", exit_codes.NOT_READY)
    assert any("llmux wait p" in str(data) for kind, data in events if kind == "log")


@pytest.mark.asyncio
async def test_timeouts_reach_readiness_validation():
    seen: dict = {}

    async def ready(*_a, **kwargs):
        seen.update(kwargs)
        yield ("result", True, [])

    commands: list = []
    with _patches(ready, commands):
        events = await _collect(stall_timeout=90.0, ready_timeout=600.0)
    assert events[-1] == ("rc", 0)
    assert seen["timeout"] == 90.0 and seen["max_wait"] == 600.0


@pytest.mark.asyncio
async def test_port_conflict_returns_conflict_code():
    async def never(*_a, **_k):
        raise AssertionError("validation must not run")
        yield

    commands: list = []
    with _patches(never, commands, check_port_conflict=AsyncMock(return_value="profile 'q'")):
        events = await _collect()
    assert events[-1] == ("rc", exit_codes.CONFLICT)
    assert commands == []


@pytest.mark.asyncio
async def test_dry_run_plans_without_side_effects():
    async def never(*_a, **_k):
        raise AssertionError("validation must not run")
        yield

    ensure_calls: list = []
    port_probe = AsyncMock(return_value="profile 'q'")
    commands: list = []
    with _patches(
        never,
        commands,
        check_port_conflict=port_probe,
        _ensure_profile_config=lambda _p, **kw: ensure_calls.append(kw) or (True, []),
    ):
        events = await _collect(dry_run=True)
    assert commands == []
    port_probe.assert_not_awaited()
    assert ensure_calls == [{"dry_run": True}]
    plans = [data for kind, data in events if kind == "plan"]
    assert plans and plans[0]["image"] == "vllm/vllm-openai:v0.20.1"
    assert plans[0]["command"][:2] == ["docker", "compose"]
    assert events[-1] == ("rc", 0)


@pytest.mark.asyncio
async def test_validation_reports_crash_and_stall_outcomes():
    profile = _profile()

    async def run(*args, **_kwargs):
        if args[:2] == ("docker", "inspect"):
            return 0, state["value"]
        if args[:2] == ("docker", "logs"):
            return 0, "same\n"
        return 0, ""

    state = {"value": "exited\tnone"}
    with patch.object(rt, "run_command", side_effect=run), patch.object(
        rt, "_models_endpoint_ready", AsyncMock(return_value=False)
    ), patch.object(rt, "load_config", return_value=rt.Config(name="p", model="m")):
        crash = [e async for e in rt._post_start_validation(profile, timeout=0.05, poll_interval=0.01)]
        state["value"] = "running\tstarting"
        stall = [e async for e in rt._post_start_validation(profile, timeout=0.05, poll_interval=0.01)]
    assert crash[-1][3] == "crashed"
    assert stall[-1][3] == "stalled"


@pytest.mark.asyncio
async def test_du_failure_warns_and_keeps_waiting(tmp_path):
    profile = _profile()
    calls = {"ready": 0}

    async def run(*args, **_kwargs):
        if args[0] == "du":
            return 124, "timed out"
        if args[:2] == ("docker", "inspect"):
            return 0, "running\tstarting"
        return 0, f"log {calls['ready']}\n"

    async def ready(*_a, **_k):
        calls["ready"] += 1
        return calls["ready"] > 3

    with patch.object(rt, "run_command", side_effect=run), patch.object(
        rt, "_models_endpoint_ready", side_effect=ready
    ), patch.object(rt, "load_config", return_value=rt.Config(name="p", model="m")):
        events = [
            e
            async for e in rt._post_start_validation(
                profile, timeout=5, poll_interval=0.01, hf_cache_path=str(tmp_path)
            )
        ]
    assert events[-1] == ("result", True, [])
    warnings = [e[1] for e in events if e[0] == "log" and "du" in e[1]]
    assert len(warnings) == 1 and warnings[0].startswith("Warning")


def test_model_cache_dir_targets_one_model(tmp_path):
    assert rt._model_cache_dir(str(tmp_path), "Qwen/Qwen3-8B") == str(
        tmp_path / "hub" / "models--Qwen--Qwen3-8B"
    )
    assert rt._model_cache_dir(str(tmp_path), "/models/local") is None
    assert rt._model_cache_dir("", "Qwen/Qwen3-8B") is None


_TUNABLE = (
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_DISABLE_COMPILE_CACHE",
    "VLLM_USE_V2_MODEL_RUNNER",
)


def test_profile_env_vars_override_vllm_runtime_defaults(tmp_path, monkeypatch):
    common = tmp_path / ".env.common"
    common.write_text("VLLM_DISABLE_COMPILE_CACHE=1\nVLLM_USE_V2_MODEL_RUNNER=1\n")
    profile_env = tmp_path / "p.env"
    profile_env.write_text("CONTAINER_NAME=p\nVLLM_DISABLE_COMPILE_CACHE=0\nVLLM_ENABLE_V1_MULTIPROCESSING=1\n")
    monkeypatch.setattr(rt, "COMMON_ENV", common)
    profile = rt.Profile(name="p")
    monkeypatch.setattr(type(profile), "path", property(lambda self: profile_env))
    env = rt._compose_env(profile, use_dev=False, version_tag="v1")
    assert env["VLLM_DISABLE_COMPILE_CACHE"] == "0"
    assert env["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_compose_reads_tunable_env_from_interpolation_not_literals():
    from pathlib import Path

    compose = (Path(rt.COMPOSE_DIR) / "docker-compose.yaml").read_text()
    for key in _TUNABLE:
        assert f"{key}=${{{key}:-" in compose, key
    assert "/root/.cache/vllm" in compose
    assert "${BIND_HOST:-127.0.0.1}:" in compose


def test_bind_host_must_cover_loopback(tmp_path):
    from tui.common.env import validate_common_env

    common = tmp_path / ".env.common"
    for value, ok in (("", True), ("127.0.0.1", True), ("0.0.0.0", True), ("192.168.0.5", False)):
        common.write_text(f"HF_CACHE_PATH={tmp_path}\nBIND_HOST={value}\n")
        common.chmod(0o600)
        assert validate_common_env(common)[0] is ok, value


@pytest.mark.asyncio
async def test_exposure_warning_only_for_open_bind_without_key(tmp_path, monkeypatch):
    from tui.common import http

    common = tmp_path / ".env.common"
    monkeypatch.setattr(http, "_common_env_path", lambda: common)
    monkeypatch.setattr(http, "profile_api_key", AsyncMock(return_value=""))
    common.write_text("BIND_HOST=0.0.0.0\n")
    assert (await http.exposure_warning("vllm", "p")).startswith("Warning")
    http.profile_api_key.return_value = "k"
    assert await http.exposure_warning("vllm", "p") == ""
    common.write_text("BIND_HOST=127.0.0.1\n")
    http.profile_api_key.return_value = ""
    assert await http.exposure_warning("vllm", "p") == ""


def test_compile_cache_defaults_next_to_hf_cache(tmp_path, monkeypatch):
    common = tmp_path / ".env.common"
    common.write_text("HF_CACHE_PATH=/data/hf/\n")
    monkeypatch.setattr(rt, "COMMON_ENV", common)
    profile = rt.Profile(name="p")
    monkeypatch.setattr(type(profile), "path", property(lambda self: tmp_path / "missing.env"))
    assert rt._compose_env(profile, use_dev=False)["VLLM_CACHE_PATH"] == "/data/vllm"
    common.write_text("HF_CACHE_PATH=/data/hf\nVLLM_CACHE_PATH=/fast/vllm\n")
    assert rt._compose_env(profile, use_dev=False)["VLLM_CACHE_PATH"] == "/fast/vllm"


@pytest.mark.parametrize("key", ["port", "host", "tensor-parallel-size", "tensor_parallel_size"])
def test_reserved_config_keys_are_rejected_on_save_and_up(key, tmp_path, monkeypatch):
    from tui.backends.vllm import backend_storage

    monkeypatch.setattr(backend_storage, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(rt.Config, "path", property(lambda self: tmp_path / f"{self.name}.yaml"))
    config = rt.Config(name="c", model="org/m", extra_params={key: 2})
    with pytest.raises(ValueError, match="profile"):
        backend_storage.save_config(config)
    assert not (tmp_path / "c.yaml").exists()
    with pytest.raises(ValueError, match="profile"):
        rt._validate_core_config_data({"model": "org/m", key: 2})


def test_disabled_reserved_key_is_allowed(tmp_path, monkeypatch):
    from tui.backends.vllm import backend_storage

    monkeypatch.setattr(backend_storage, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(rt.Config, "path", property(lambda self: tmp_path / f"{self.name}.yaml"))
    backend_storage.save_config(rt.Config(name="c", model="org/m", disabled_params={"port": 9000}))
    assert (tmp_path / "c.yaml").exists()


def test_recipe_import_drops_reserved_keys():
    from tui.common.recipes import Recipe, build_config

    recipe = Recipe(
        model_id="org/m",
        base_args=["--tensor-parallel-size", "8", "--port", "9000", "--max-num-seqs", "64"],
    )
    _model, params = build_config(recipe, None, [])
    assert params == {"max-num-seqs": 64}


def test_prepare_lists_draft_and_tokenizer_repos():
    config = rt.Config(
        name="c",
        model="org/main",
        extra_params={
            "revision": "r1",
            "tokenizer": "org/tok",
            "tokenizer-revision": "t1",
            "speculative-config": '{"method": "eagle3", "model": "org/draft", "num_speculative_tokens": 3}',
        },
    )
    assert rt._prepare_repos(config) == [
        ("model", "org/main", "r1"),
        ("tokenizer", "org/tok", "t1"),
        ("draft model", "org/draft", ""),
    ]


def test_prepare_skips_local_paths_mtp_and_duplicates():
    config = rt.Config(
        name="c",
        model="org/main",
        extra_params={
            "tokenizer": "org/main",
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
        },
    )
    assert rt._prepare_repos(config) == [("model", "org/main", "")]
    local = rt.Config(name="c", model="/models/x", extra_params={"tokenizer": "./tok"})
    assert rt._prepare_repos(local) == []


@pytest.mark.asyncio
async def test_prepare_downloads_every_repo_and_skips_tokenizer_weights(monkeypatch, tmp_path):
    from tui.common import prepare

    config = rt.Config(name="p", model="org/main", extra_params={"tokenizer": "org/tok"})
    calls: list = []

    async def download(**kwargs):
        calls.append((kwargs["model_id"], kwargs.get("ignore_patterns")))
        yield ("rc", 0)

    profile = _profile()
    monkeypatch.setattr(rt.profile_store, "load_profile", lambda *_a: object())
    monkeypatch.setattr(rt, "load_profile", lambda _n: profile)
    monkeypatch.setattr(rt, "_ensure_common_env", lambda _p: (True, []))
    monkeypatch.setattr(rt, "_ensure_profile_config", lambda _p: (True, []))
    monkeypatch.setattr(rt, "_render_profile_snapshot", lambda _n: (profile, "/x"))
    monkeypatch.setattr(rt, "load_config", lambda _n: config)
    monkeypatch.setattr(prepare, "image_present", AsyncMock(return_value=True))
    monkeypatch.setattr(prepare, "hf_cache_path", lambda: str(tmp_path))
    monkeypatch.setattr(prepare, "stream_vllm_download", download)
    events = [e async for e in rt.stream_container_prepare("p")]
    assert events[-1] == ("rc", 0)
    assert calls[0] == ("org/main", None)
    assert calls[1][0] == "org/tok" and "*.safetensors" in calls[1][1]


@pytest.mark.asyncio
async def test_running_but_unhealthy_is_still_loading_not_crashed():
    """Compose marks a slow first load unhealthy (~690s) while it keeps running."""
    profile = _profile()

    async def run(*args, **_kwargs):
        if args[:2] == ("docker", "inspect"):
            return 0, "running\tunhealthy"
        if args[:2] == ("docker", "logs"):
            return 0, "same\n"
        return 0, ""

    with patch.object(rt, "run_command", side_effect=run), patch.object(
        rt, "_models_endpoint_ready", AsyncMock(return_value=False)
    ), patch.object(rt, "load_config", return_value=rt.Config(name="p", model="m")):
        events = [e async for e in rt._post_start_validation(profile, timeout=0.05, poll_interval=0.01)]
    assert events[-1][3] == "stalled"


@pytest.mark.asyncio
async def test_wait_keeps_waiting_on_running_unhealthy(monkeypatch):
    from tui.common import docker, readiness

    snapshot = docker.ContainerSnapshot(
        name="p", lifecycle=docker.ContainerLifecycle.RUNNING,
        health=docker.ContainerHealth.UNHEALTHY, raw_state="running", raw_status="running",
    )
    monkeypatch.setattr(readiness.docker, "container_snapshots", AsyncMock(return_value={"p": snapshot}))
    monkeypatch.setattr(readiness, "probe_served", AsyncMock(return_value=("loading", [], "")))
    stored = profile_store.StoredProfile(name="p", backend="vllm", port=8000)
    rc, _msg, _models = await readiness.wait_ready(stored, timeout=0, interval=0.01)
    assert rc == exit_codes.NOT_READY


@pytest.mark.parametrize("source", ["shell", "profile"])
def test_bind_host_reaching_compose_is_the_validated_common_value(tmp_path, monkeypatch, source):
    common = tmp_path / ".env.common"
    common.write_text("HF_CACHE_PATH=/data/hf\n")
    profile_env = tmp_path / "p.env"
    profile_env.write_text("BIND_HOST=10.0.0.9\n" if source == "profile" else "")
    if source == "shell":
        monkeypatch.setenv("BIND_HOST", "10.0.0.9")
    monkeypatch.setattr(rt, "COMMON_ENV", common)
    profile = rt.Profile(name="p")
    monkeypatch.setattr(type(profile), "path", property(lambda self: profile_env))
    assert rt._compose_env(profile, use_dev=False)["BIND_HOST"] == "127.0.0.1"
    common.write_text("HF_CACHE_PATH=/data/hf\nBIND_HOST=0.0.0.0\n")
    assert rt._compose_env(profile, use_dev=False)["BIND_HOST"] == "0.0.0.0"


@pytest.mark.parametrize("key", ["BIND_HOST", "VLLM_CACHE_PATH"])
def test_profile_env_vars_cannot_set_host_level_keys(key):
    assert profile_store.is_protected_profile_env_key(key, "vllm")
    assert profile_store.is_protected_profile_env_key(key, "llamacpp")


@pytest.mark.asyncio
async def test_tui_not_ready_says_still_loading_and_follows_logs(monkeypatch):
    from textual.app import App
    from tui.backends.vllm.screens import container as screen_mod

    profile = rt.Profile(name="p", container_name="p", port="8000", config_name="p", image_tag="vllm/vllm-openai:v1")
    monkeypatch.setattr(screen_mod, "load_profile", lambda _n: profile)
    monkeypatch.setattr(screen_mod.ContainerUpScreen, "on_mount", lambda self: None)
    monkeypatch.setattr(screen_mod, "check_port_conflict", AsyncMock(return_value=None))
    monkeypatch.setattr("tui.common.widgets.confirm_model_switch", AsyncMock(return_value=True))
    followed: list = []

    async def up(*_a, **_k):
        yield ("rc", exit_codes.NOT_READY)

    async def logs(name, **_k):
        followed.append(name)
        yield "loading..."

    monkeypatch.setattr(screen_mod, "stream_container_up", up)
    monkeypatch.setattr(screen_mod, "stream_container_logs", logs)
    app = App()
    async with app.run_test(size=(120, 45)) as pilot:
        screen = screen_mod.ContainerUpScreen("p")
        await app.push_screen(screen)
        await pilot.pause()
        screen.action_confirm_start()
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = str(screen.query_one("#startup-status").render())
    assert "still loading" in text.lower() or "로딩" in text
    assert followed == ["p"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("present", "pull_rc", "stage", "rc", "pulled"),
    [
        (True, 0, True, 0, False),
        (False, 0, True, 0, True),
        (False, 1, True, exit_codes.ERROR, True),
        (False, 0, False, 0, False),
    ],
)
async def test_dry_run_secures_the_image_when_staging(monkeypatch, present, pull_rc, stage, rc, pulled):
    from tui.common import prepare

    pulls: list = []

    async def pull(ref):
        pulls.append(ref)
        yield ("log", "pulling")
        yield ("rc", pull_rc)

    monkeypatch.setattr(prepare, "image_present", AsyncMock(return_value=present))
    monkeypatch.setattr(prepare, "stream_pull", pull)

    async def never(*_a, **_k):
        raise AssertionError("validation must not run")
        yield

    commands: list = []
    with _patches(never, commands):
        events = await _collect(dry_run=True, stage_image=stage)
    assert events[-1] == ("rc", rc)
    assert bool(pulls) is pulled
    assert commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tag", ["", "vllm/vllm-openai:v0.21.0"])
@pytest.mark.parametrize(("staged", "policy"), [(False, "always"), (True, "never")])
async def test_staged_start_reuses_the_image_the_precheck_secured(tag, staged, policy):
    """After --replace stopped the old model, the real start must not pull again."""

    async def ready(*_a, **_k):
        yield ("result", True, [])

    commands: list = []
    with _patches(ready, commands):
        events = await _collect(pull=True, tag=tag, image_staged=staged)
    assert events[-1] == ("rc", 0)
    assert commands[0][commands[0].index("--pull") + 1] == policy


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "outcome"), [(401, "error"), (503, None)])
async def test_models_probe_auth_failure_is_a_config_error_not_loading(code, outcome):
    import urllib.error

    def answer(*_a, **_k):
        raise urllib.error.HTTPError("http://127.0.0.1:8000/v1/models", code, "x", {}, None)

    async def run(*args, **_kwargs):
        if args[:2] == ("docker", "inspect"):
            return 0, "running\thealthy\tabc"
        return 0, "same\n"

    with patch.object(rt, "open_url", side_effect=answer), patch.object(
        rt, "run_command", side_effect=run
    ), patch.object(rt, "load_config", return_value=rt.Config(name="p", model="m")):
        events = [e async for e in rt._post_start_validation(_profile(), timeout=0.05, poll_interval=0.01)]
    result = events[-1]
    if outcome is None:
        assert result[3] == "stalled"
    else:
        assert result[1] is False and result[3] == outcome
        assert "401" in " ".join(result[2])


@pytest.mark.asyncio
async def test_misconfigured_server_exits_error_and_keeps_running():
    async def misconfigured(*_a, **_k):
        yield ("result", False, ["Error: /v1/models answered HTTP 401"], "error")

    commands: list = []
    with _patches(misconfigured, commands):
        events = await _collect()
    assert events[-1] == ("rc", exit_codes.ERROR)


@pytest.mark.asyncio
async def test_dry_run_plan_names_the_profile_env_the_real_up_would_render(tmp_path, monkeypatch):
    """The plan is the exact compose command, even before the runtime .env exists."""
    from tui.backends.vllm import backend_common

    monkeypatch.setattr(backend_common, "RUNTIME_DIR", tmp_path)

    async def never(*_a, **_k):
        raise AssertionError("validation must not run")
        yield

    commands: list = []
    with _patches(never, commands):
        events = await _collect(dry_run=True)
    command = [data for kind, data in events if kind == "plan"][0]["command"]
    env_file = str(tmp_path / "p.env")
    assert not (tmp_path / "p.env").exists()
    assert command[command.index(env_file) - 1] == "--env-file"
