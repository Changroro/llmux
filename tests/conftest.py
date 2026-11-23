import pytest

from tui.common import profile_store


def pytest_configure(config):
    config.addinivalue_line("markers", "migration: let the CLI root callback migrate legacy GGUF fields")


@pytest.fixture(autouse=True)
def _isolated_profile_store(tmp_path, monkeypatch):
    # Tests must never read or write the developer's real profiles.yaml / .runtime.
    monkeypatch.setattr(profile_store, "PROFILES_YAML", tmp_path / "profiles.yaml")
    monkeypatch.setattr(profile_store, "RUNTIME_DIR", tmp_path / ".runtime")


@pytest.fixture(autouse=True)
def _no_real_project_writes(request, tmp_path_factory, monkeypatch):
    # A subprocess that does not pass its own LLMUX_ROOT gets an empty checkout, never the real one.
    root = tmp_path_factory.mktemp("llmux-root")
    (root / "compose").mkdir()
    monkeypatch.setenv("LLMUX_ROOT", str(root))
    # In-process CLI runs would migrate configs in the real config/llamacpp/.
    if request.node.get_closest_marker("migration") is None:
        import tui.cli

        monkeypatch.setattr(tui.cli, "_migrate_legacy_storage", lambda: None)
