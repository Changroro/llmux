import pytest

from tui.common import profile_store


@pytest.fixture(autouse=True)
def _isolated_profile_store(tmp_path, monkeypatch):
    # Tests must never read or write the developer's real profiles.yaml / .runtime.
    monkeypatch.setattr(profile_store, "PROFILES_YAML", tmp_path / "profiles.yaml")
    monkeypatch.setattr(profile_store, "RUNTIME_DIR", tmp_path / ".runtime")
