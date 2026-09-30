"""Width-guard (TooNarrowScreen) behaviour, driven through Textual's pilot.

Textual delivers the initial Resize *before* on_mount, so a terminal that
starts out narrow used to bury the guard underneath the dashboard: the stack
came out as [_default, TooNarrow, Dashboard], the guard was invisible, and F1
was a permanent no-op for the rest of the session.
"""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tui.app import LlmuxApp
from tui.screens.dashboard import DashboardScreen
from tui.screens.too_narrow import TooNarrowScreen

NARROW = (60, 30)
WIDE = (120, 30)


class WidthGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_narrow_launch_puts_guard_on_top(self) -> None:
        app = LlmuxApp()
        async with app.run_test(size=NARROW) as pilot:
            await pilot.pause()
            self.assertIsInstance(app.screen, TooNarrowScreen)
            # Exactly one guard, and it sits above the dashboard.
            guards = [s for s in app.screen_stack if isinstance(s, TooNarrowScreen)]
            self.assertEqual(len(guards), 1)
            self.assertTrue(
                any(isinstance(s, DashboardScreen) for s in app.screen_stack)
            )

    async def test_widening_after_narrow_launch_releases_guard(self) -> None:
        app = LlmuxApp()
        async with app.run_test(size=NARROW) as pilot:
            await pilot.pause()
            self.assertIsInstance(app.screen, TooNarrowScreen)

            await pilot.resize_terminal(*WIDE)
            await pilot.pause()

            self.assertIsInstance(app.screen, DashboardScreen)
            self.assertIsNone(app._too_narrow)
            self.assertFalse(
                any(isinstance(s, TooNarrowScreen) for s in app.screen_stack)
            )

    async def test_f1_works_after_narrow_launch_and_widen(self) -> None:
        app = LlmuxApp()
        async with app.run_test(size=NARROW) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(*WIDE)
            await pilot.pause()

            # F1 used to be dead for the whole session because `_too_narrow`
            # stayed set on a guard that could never surface.
            app.push_screen("vllm_configs")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, DashboardScreen)

            await pilot.press("f1")
            await pilot.pause()
            self.assertIsInstance(app.screen, DashboardScreen)

    async def test_renarrow_does_not_stack_a_second_guard(self) -> None:
        app = LlmuxApp()
        async with app.run_test(size=NARROW) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(*WIDE)
            await pilot.pause()
            await pilot.resize_terminal(*NARROW)
            await pilot.pause()

            self.assertIsInstance(app.screen, TooNarrowScreen)
            guards = [s for s in app.screen_stack if isinstance(s, TooNarrowScreen)]
            self.assertEqual(len(guards), 1)

    async def test_wide_launch_has_no_guard(self) -> None:
        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            self.assertIsInstance(app.screen, DashboardScreen)
            self.assertIsNone(app._too_narrow)

    async def test_f1_leaves_guard_in_place_while_narrow(self) -> None:
        app = LlmuxApp()
        async with app.run_test(size=NARROW) as pilot:
            await pilot.pause()
            await pilot.press("f1")
            await pilot.pause()
            # The guard outranks F1 — popping it would expose the clipped
            # screens it exists to hide.
            self.assertIsInstance(app.screen, TooNarrowScreen)


@contextlib.contextmanager
def _patched_vllm_config_dir(path: Path):
    """Point every module that holds a vLLM CONFIG_DIR reference at `path`."""
    from tui.backends.vllm import backend, backend_common, backend_storage
    from tui.backends.vllm.screens import config as config_screen

    with contextlib.ExitStack() as stack:
        for mod in (backend, backend_common, backend_storage, config_screen):
            stack.enter_context(patch.object(mod, "CONFIG_DIR", path))
        yield


@contextlib.contextmanager
def _patched_llamacpp_config_dir(path: Path):
    from tui.backends.llamacpp import backend
    from tui.backends.llamacpp.screens import quick_setup

    with patch.object(backend, "CONFIG_DIR", path), patch.object(
        quick_setup, "CONFIG_DIR", path
    ):
        yield


class ConfigFormDisableSwitchTests(unittest.IsolatedAsyncioTestCase):
    """The config form's per-row Switch decides active vs disabled; toggling it
    off and saving must write a disabled marker, and re-opening must restore the
    off state."""

    async def test_toggle_off_saves_marker_and_restores(self) -> None:
        from textual.widgets import Input, Switch
        from tui.backends.vllm.backend_common import CONFIG_DIR as _real  # noqa: F401
        from tui.backends.vllm.screens.config import ConfigFormScreen
        from tui.backends.vllm import backend_storage as vs

        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            with _patched_vllm_config_dir(cdir):
                # Seed a config with two active params.
                (cdir / "cfg.yaml").write_text(
                    "model: m/x\ngpu-memory-utilization: '0.9'\nmax-model-len: 4096\n"
                )

                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ConfigFormScreen("cfg")
                    await app.push_screen(screen)
                    await pilot.pause()

                    # Find the max-model-len row's switch and turn it off.
                    target = None
                    for row in screen.query(".param-row"):
                        if row.query_one(".param-key", Input).value == "max-model-len":
                            target = row.query_one(".param-switch", Switch)
                    self.assertIsNotNone(target)
                    target.value = False
                    await pilot.pause()

                    screen.query_one("#save-btn").press()
                    await pilot.pause()

                text = (cdir / "cfg.yaml").read_text()
                self.assertIn("# llmux:disabled max-model-len", text)

                # Reload: the param is disabled, so a fresh form shows it off.
                cfg = vs.load_config("cfg")
                self.assertIn("max-model-len", cfg.disabled_params)
                self.assertNotIn("max-model-len", cfg.extra_params)


class ConfigFlagImageSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_profile_keeps_its_runtime_context(self) -> None:
        from tui.common import flag_discovery

        with patch.object(flag_discovery, "profile_target", AsyncMock(return_value=("sha256:actual", "p"))) as resolve:
            self.assertEqual(await flag_discovery.config_target("vllm", "shared", "p"), ("sha256:actual", "p"))
        resolve.assert_awaited_once_with("vllm", "p")

    async def test_config_rejects_ambiguous_runtime_images(self) -> None:
        from tui.common import flag_discovery, profile_store

        profiles = [profile_store.StoredProfile(name=name, backend="llamacpp", config_name="cfg") for name in ("a", "b")]
        with patch.object(profile_store, "list_profiles", return_value=profiles), patch.object(flag_discovery, "profile_target", AsyncMock(side_effect=[("sha256:a", "a"), ("sha256:b", "b")])):
            with self.assertRaisesRegex(RuntimeError, "multiple images"):
                await flag_discovery.config_target("llamacpp", "cfg")

    async def test_new_config_does_not_inherit_an_unlinked_profile_image(self) -> None:
        from tui.common import flag_discovery

        with patch.object(flag_discovery, "profile_target", AsyncMock()) as resolve:
            self.assertEqual(await flag_discovery.config_target("vllm", ""), ("", ""))
        resolve.assert_not_awaited()


class ConfigFlagIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_vllm_flag_sets_are_isolated_per_form(self) -> None:
        from tui.backends.vllm.screens import config as screen_module

        screens = []
        for flag in ("image-a-only", "image-b-only"):
            with patch.object(screen_module, "list_profile_names", return_value=[]), \
                patch.object(
                    screen_module,
                    "extract_vllm_params",
                    AsyncMock(return_value={flag}),
                ):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = screen_module.ConfigFormScreen()
                    screens.append(screen)
                    await app.push_screen(screen)
                    await app.workers.wait_for_complete()
        self.assertEqual(screens[0]._known_params, {"image-a-only"})
        self.assertEqual(screens[1]._known_params, {"image-b-only"})

    async def test_llamacpp_flag_sets_are_isolated_per_form(self) -> None:
        from tui.backends.llamacpp.screens import config as screen_module

        screens = []
        for flag in ("image-a-only", "image-b-only"):
            with patch.object(screen_module, "list_profile_names", return_value=[]), \
                patch.object(
                    screen_module,
                    "extract_llama_server_flags",
                    AsyncMock(return_value={flag}),
                ):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = screen_module.ConfigFormScreen()
                    screens.append(screen)
                    await app.push_screen(screen)
                    await app.workers.wait_for_complete()
        self.assertEqual(screens[0]._known_flags, {"image-a-only"})
        self.assertEqual(screens[1]._known_flags, {"image-b-only"})


class ConfigListRenameTests(unittest.IsolatedAsyncioTestCase):
    """`R` on the config list renames the YAML and repoints referencing
    profiles — and refuses while a referencing profile's container is up."""

    async def test_vllm_params_count_matches_cli_yaml_key_count(self) -> None:
        from textual.widgets import DataTable
        from tui.backends.vllm.screens.config import ConfigListScreen

        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            (cdir / "cfg.yaml").write_text(
                "model: org/model\ngpu-memory-utilization: 0.9\nmax-model-len: 4096\n"
            )
            with _patched_vllm_config_dir(cdir):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ConfigListScreen()
                    await app.push_screen(screen)
                    await pilot.pause()
                    row = screen.query_one("#config-table", DataTable).get_row_at(0)

            self.assertEqual(str(row[3]), "3")

    async def test_rename_moves_file_and_notifies(self) -> None:
        from tui.backends.vllm.screens.config import ConfigListScreen

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cdir = root / "config"
            cdir.mkdir()
            (cdir / "cfg.yaml").write_text("model: m/x\n")
            profiles_yaml = root / "profiles.yaml"
            profiles_yaml.write_text(
                "version: 1\ndefaults: {}\nprofiles:\n"
                "- name: p\n  backend: vllm\n  config_name: cfg\n"
            )

            with _patched_vllm_config_dir(cdir), \
                patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), \
                patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"), \
                patch("tui.common.config_store.config_dir", lambda b: cdir), \
                patch(
                    "tui.common.docker.running_container_names",
                    AsyncMock(return_value=set()),
                ):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ConfigListScreen()
                    await app.push_screen(screen)
                    await pilot.pause()

                    screen._rename_config("cfg", "cfg2")
                    await pilot.pause()
                    await pilot.pause()

                self.assertFalse((cdir / "cfg.yaml").exists())
                self.assertTrue((cdir / "cfg2.yaml").exists())
                from tui.common import profile_store

                self.assertEqual(
                    profile_store.load_profile("p", "vllm").config_name, "cfg2"
                )

    async def test_rename_refuses_while_referencing_container_runs(self) -> None:
        from tui.backends.vllm.screens.config import ConfigListScreen

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cdir = root / "config"
            cdir.mkdir()
            (cdir / "cfg.yaml").write_text("model: m/x\n")
            profiles_yaml = root / "profiles.yaml"
            profiles_yaml.write_text(
                "version: 1\ndefaults: {}\nprofiles:\n"
                "- name: p\n  backend: vllm\n  config_name: cfg\n"
            )

            with _patched_vllm_config_dir(cdir), \
                patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), \
                patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"), \
                patch("tui.common.config_store.config_dir", lambda b: cdir), \
                patch(
                    "tui.common.docker.running_container_names",
                    AsyncMock(return_value={"p"}),
                ):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ConfigListScreen()
                    await app.push_screen(screen)
                    await pilot.pause()

                    screen._rename_config("cfg", "cfg2")
                    await pilot.pause()
                    await pilot.pause()

                self.assertTrue((cdir / "cfg.yaml").exists())
                self.assertFalse((cdir / "cfg2.yaml").exists())


class ProfileFormRenameTests(unittest.IsolatedAsyncioTestCase):
    """The edit form's Name field is editable: changing it renames the profile
    instead of creating a second one."""

    @contextlib.contextmanager
    def _env(self, root: Path, running: set[str]):
        profiles_yaml = root / "profiles.yaml"
        profiles_yaml.write_text(
            "version: 1\ndefaults: {}\nprofiles:\n"
            "- name: solo\n  backend: vllm\n  port: 8123\n"
        )
        with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), \
            patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"), \
            patch(
                "tui.common.docker.running_container_names",
                AsyncMock(return_value=running),
            ):
            yield

    async def test_editing_name_renames_in_place(self) -> None:
        from textual.widgets import Input
        from tui.backends.vllm.backend import load_profile
        from tui.backends.vllm.screens.profile import ProfileFormScreen
        from tui.common import profile_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._env(root, set()):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ProfileFormScreen(load_profile("solo"))
                    await app.push_screen(screen)
                    await pilot.pause()

                    name_input = screen.query_one("#name-input", Input)
                    self.assertFalse(name_input.disabled)  # the whole point
                    name_input.value = "renamed"
                    await pilot.pause()

                    screen.query_one("#save-btn").press()
                    await pilot.pause()
                    await pilot.pause()

                # Renamed, not duplicated.
                self.assertIsNone(profile_store.load_profile("solo", "vllm"))
                moved = profile_store.load_profile("renamed", "vllm")
                self.assertIsNotNone(moved)
                self.assertEqual(moved.port, 8123)
                self.assertEqual(
                    len(profile_store.list_profiles("vllm")), 1
                )

    async def test_editing_name_refused_while_running(self) -> None:
        from textual.widgets import Input
        from tui.backends.vllm.backend import load_profile
        from tui.backends.vllm.screens.profile import ProfileFormScreen
        from tui.common import profile_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._env(root, {"solo"}):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ProfileFormScreen(load_profile("solo"))
                    await app.push_screen(screen)
                    await pilot.pause()

                    screen.query_one("#name-input", Input).value = "renamed"
                    await pilot.pause()
                    screen.query_one("#save-btn").press()
                    await pilot.pause()
                    await pilot.pause()

                # Nothing moved, and no stray second profile was written.
                self.assertIsNotNone(profile_store.load_profile("solo", "vllm"))
                self.assertIsNone(profile_store.load_profile("renamed", "vllm"))
                self.assertEqual(len(profile_store.list_profiles("vllm")), 1)


class ProfileDeleteSafetyTests(unittest.IsolatedAsyncioTestCase):
    @contextlib.contextmanager
    def _env(self, root: Path, running: set[str]):
        profiles_yaml = root / "profiles.yaml"
        profiles_yaml.write_text(
            "version: 1\ndefaults: {}\nprofiles:\n"
            "- name: delete-me\n  backend: vllm\n  port: 8123\n  config_name: cfg\n"
        )
        config_dir = root / "config"
        config_dir.mkdir()
        (config_dir / "cfg.yaml").write_text("model: m/x\n")
        with _patched_vllm_config_dir(config_dir), \
            patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), \
            patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"), \
            patch(
                "tui.common.docker.running_container_names",
                AsyncMock(return_value=running),
            ):
            yield profiles_yaml, config_dir

    async def test_delete_confirmation_rechecks_running_state(self) -> None:
        from tui.backends.vllm.screens.profile import ProfileDeleteScreen
        from tui.common import profile_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._env(root, {"delete-me"}):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    await app.push_screen(ProfileDeleteScreen("delete-me"))
                    await pilot.pause()
                    app.screen.query_one("#delete-btn").press()
                    await pilot.pause()
                    await pilot.pause()
                self.assertIsNotNone(profile_store.load_profile("delete-me", "vllm"))

    async def test_tui_delete_keeps_linked_config_by_default(self) -> None:
        from tui.backends.vllm.screens.profile import ProfileDeleteScreen
        from tui.common import profile_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._env(root, set()) as (_, config_dir):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    await app.push_screen(ProfileDeleteScreen("delete-me"))
                    await pilot.pause()
                    app.screen.query_one("#delete-btn").press()
                    await pilot.pause()
                    await pilot.pause()
                self.assertIsNone(profile_store.load_profile("delete-me", "vllm"))
                self.assertTrue((config_dir / "cfg.yaml").exists())


class VllmProfileGpuEditTests(unittest.IsolatedAsyncioTestCase):
    async def test_gpu_edit_recomputes_untouched_tensor_parallel(self) -> None:
        from textual.widgets import Input
        from tui.backends.vllm.backend import Profile
        from tui.backends.vllm.screens import profile as screen_module

        saved = []
        profile = Profile(
            name="gpu-edit",
            container_name="gpu-edit",
            port="8123",
            gpu_id="0",
            tensor_parallel="1",
        )
        with patch.object(screen_module, "list_config_names", return_value=[]), \
            patch.object(screen_module, "save_profile", side_effect=saved.append):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = screen_module.ProfileFormScreen(profile)
                await app.push_screen(screen)
                await pilot.pause()
                screen.query_one("#gpu-input", Input).value = "0,1"
                screen.query_one("#save-btn").press()
                await pilot.pause()

        self.assertEqual(saved[0].gpu_id, "0,1")
        self.assertEqual(saved[0].tensor_parallel, "2")


class ProfileParityTests(unittest.IsolatedAsyncioTestCase):
    async def test_action_menus_offer_clone_for_both_backends(self) -> None:
        from tui.backends.llamacpp import backend as lbackend
        from tui.backends.llamacpp.screens.dashboard import ActionModal
        from tui.backends.vllm.screens.dashboard import ProfileActionScreen

        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            vllm = ProfileActionScreen("v", False)
            await app.push_screen(vllm)
            await pilot.pause()
            self.assertIn("clone", {option.id for option in vllm.query_one("#action-list").options})
            app.pop_screen()
            await pilot.pause()

            profile = lbackend.Profile(name="l", container_name="l", port=8080)
            llama = ActionModal(profile)
            await app.push_screen(llama)
            await pilot.pause()
            self.assertIn("clone-profile", {option.id for option in llama.query_one("#action-list").options})

    async def test_action_menus_offer_runtime_env_rendering_for_both_backends(self) -> None:
        from tui.backends.llamacpp import backend as lbackend
        from tui.backends.llamacpp.screens.dashboard import ActionModal
        from tui.backends.vllm.screens.dashboard import ProfileActionScreen

        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            vllm = ProfileActionScreen("v", False)
            await app.push_screen(vllm)
            await pilot.pause()
            self.assertIn("render_env", {option.id for option in vllm.query_one("#action-list").options})
            app.pop_screen()
            await pilot.pause()

            profile = lbackend.Profile(name="l", container_name="l", port=8080)
            llama = ActionModal(profile)
            await app.push_screen(llama)
            await pilot.pause()
            self.assertIn("render-env", {option.id for option in llama.query_one("#action-list").options})

    async def test_vllm_profile_form_saves_environment_variables(self) -> None:
        from textual.widgets import TextArea
        from tui.backends.vllm.backend import Profile
        from tui.backends.vllm.screens import profile as screen_module

        saved = []
        profile = Profile(name="v", container_name="v", port="8123")
        with patch.object(screen_module, "list_config_names", return_value=[]), \
            patch.object(screen_module, "save_profile", side_effect=saved.append):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = screen_module.ProfileFormScreen(profile)
                await app.push_screen(screen)
                await pilot.pause()
                screen.query_one("#env-vars-input", TextArea).text = "OMP_NUM_THREADS=4\nEMPTY="
                screen.query_one("#save-btn").press()
                await pilot.pause()

        self.assertEqual(saved[0].env_vars, {"OMP_NUM_THREADS": "4", "EMPTY": ""})

    async def test_llamacpp_profile_form_saves_environment_variables(self) -> None:
        from textual.widgets import TextArea
        from tui.backends.llamacpp import backend
        from tui.backends.llamacpp.screens import profile as screen_module

        saved = []
        profile = backend.Profile(name="l", container_name="l", port=8124)
        with patch.object(screen_module, "list_config_names", return_value=[]), \
            patch.object(screen_module, "save_profile", side_effect=saved.append):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = screen_module.ProfileFormScreen(profile)
                await app.push_screen(screen)
                await pilot.pause()
                screen.query_one("#env-vars-input", TextArea).text = "LLAMA_LOG_COLORS=1"
                screen.query_one("#save-btn").press()
                await pilot.pause()

        self.assertEqual(saved[0].env_vars, {"LLAMA_LOG_COLORS": "1"})

    async def test_vllm_quick_setup_uses_custom_profile_name(self) -> None:
        from textual.widgets import Input
        from tui.backends.vllm.screens import quick_setup as screen_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config" / "vllm"
            with _patched_vllm_config_dir(config_dir), patch.object(
                screen_module.profile_store, "PROFILES_YAML", root / "profiles.yaml"
            ), patch.object(
                screen_module.profile_store, "RUNTIME_DIR", root / ".runtime"
            ):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = screen_module.QuickSetupScreen()
                    await app.push_screen(screen)
                    await pilot.pause()
                    screen.query_one("#model-input", Input).value = "org/Model"
                    screen.query_one("#name-input", Input).value = "custom-name"
                    screen.query_one("#create-btn").press()
                    await pilot.pause()

                stored = screen_module.profile_store.load_profile(
                    "custom-name", "vllm"
                )
                self.assertIsNotNone(stored)
                self.assertEqual(stored.name, "custom-name")
                self.assertTrue((config_dir / "custom-name.yaml").exists())


class LlamacppQuickSetupManualFileTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_gguf_file_allows_creation_without_listing(self) -> None:
        from textual.widgets import Input
        from tui.backends.llamacpp.screens import quick_setup as screen_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config" / "llamacpp"
            with _patched_llamacpp_config_dir(config_dir), patch.object(
                screen_module.profile_store, "PROFILES_YAML", root / "profiles.yaml"
            ), patch.object(
                screen_module.profile_store, "RUNTIME_DIR", root / ".runtime"
            ):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = screen_module.QuickSetupScreen()
                    await app.push_screen(screen)
                    await pilot.pause()
                    screen.query_one("#repo-input", Input).value = "org/model-GGUF"
                    screen._listing_failed = True
                    manual = screen.query_one("#manual-file-input", Input)
                    manual.disabled = False
                    manual.value = "model.gguf"
                    screen.query_one("#name-input", Input).value = "manual-model"
                    screen.query_one("#create-btn").press()
                    await pilot.pause()

                stored = screen_module.profile_store.load_profile(
                    "manual-model", "llamacpp"
                )
                config = screen_module.load_config("manual-model")
                self.assertIsNotNone(stored)
                self.assertEqual(stored.hf_file, "model.gguf")
                self.assertEqual(config.params["model-file"], "model.gguf")


class ConfigFormRenameTests(unittest.IsolatedAsyncioTestCase):
    """The config form's Name field is editable too: changing it renames the
    YAML and repoints referencing profiles, instead of forking a copy."""

    async def test_editing_name_renames_and_repoints(self) -> None:
        from textual.widgets import Input
        from tui.backends.vllm.screens.config import ConfigFormScreen
        from tui.common import profile_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cdir = root / "config"
            cdir.mkdir()
            (cdir / "cfg.yaml").write_text("model: m/x\ngpu-memory-utilization: '0.9'\n")
            profiles_yaml = root / "profiles.yaml"
            profiles_yaml.write_text(
                "version: 1\ndefaults: {}\nprofiles:\n"
                "- name: p\n  backend: vllm\n  config_name: cfg\n"
            )

            with _patched_vllm_config_dir(cdir), \
                patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), \
                patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"), \
                patch("tui.common.config_store.config_dir", lambda b: cdir), \
                patch(
                    "tui.common.docker.running_container_names",
                    AsyncMock(return_value=set()),
                ):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ConfigFormScreen("cfg")
                    await app.push_screen(screen)
                    await pilot.pause()

                    name_input = screen.query_one("#name-input", Input)
                    self.assertFalse(name_input.disabled)
                    name_input.value = "cfg2"
                    await pilot.pause()

                    screen.query_one("#save-btn").press()
                    await pilot.pause()
                    await pilot.pause()

                # Renamed, not duplicated, and the profile follows.
                self.assertFalse((cdir / "cfg.yaml").exists())
                self.assertTrue((cdir / "cfg2.yaml").exists())
                self.assertEqual(
                    profile_store.load_profile("p", "vllm").config_name, "cfg2"
                )


class MonitorScreenTests(unittest.IsolatedAsyncioTestCase):
    """The live monitor renders real metric values from /metrics + nvidia-smi."""

    def _row(self, backend="vllm"):
        from tui.common.adapter import DashboardRow
        return DashboardRow(
            backend=backend, profile_name="m1", container_name="m1",
            port=8000, running=True, model="m1", detail="", gpu_id="0",
        )

    async def test_renders_metrics_and_gpu(self) -> None:
        import io
        from rich.console import Console
        from tui.common.metrics import MetricsSnapshot
        from tui.common.docker import GpuInfo
        from tui.screens.monitor import MonitorScreen
        snap = MetricsSnapshot(
            backend="vllm", prompt_tokens=100.0, generation_tokens=200.0,
            requests_running=3.0, requests_waiting=1.0, kv_cache_usage=0.34,
        )
        gpus = [GpuInfo("0", "RTX", "8000", "16000", "78", "71", "210")]

        with patch("tui.common.plain_monitor.fetch_snapshot",
                   AsyncMock(return_value=snap)), \
            patch("tui.common.plain_monitor._running_rows",
                  AsyncMock(return_value=([self._row()], []))), \
            patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=gpus)), \
            patch("tui.common.docker.get_pcie_stats", AsyncMock(return_value={})):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = MonitorScreen(self._row())
                await app.push_screen(screen)
                for _ in range(30):
                    await pilot.pause()
                    if screen._rendered is not None:
                        break

                con = Console(width=120, file=io.StringIO())
                con.print(screen._rendered)
                out = con.file.getvalue()
                self.assertIn("34%", out)
                self.assertIn("GPU0", out)
                self.assertIn("210W", out)
                self.assertIn("REQUESTS", out)

    async def test_unreachable_server_does_not_crash(self) -> None:
        from tui.screens.monitor import MonitorScreen
        from textual.widgets import Static

        with patch("tui.common.plain_monitor.fetch_snapshot",
                   AsyncMock(return_value=None)), \
            patch("tui.common.plain_monitor._running_rows",
                  AsyncMock(return_value=([self._row()], []))), \
            patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[])), \
            patch("tui.common.docker.get_pcie_stats", AsyncMock(return_value={})):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = MonitorScreen(self._row())
                await app.push_screen(screen)
                await pilot.pause()
                await pilot.pause()
                # Renders (with unreachable server) rather than raising.
                self.assertIsInstance(screen.query_one("#mon", Static), Static)

    async def test_opens_with_nothing_running_and_shows_gpu(self) -> None:
        """`v` must not be gated on a running container — the GPU panel is the
        whole point of opening the monitor when nothing is up."""
        import io
        from rich.console import Console
        from tui.common.docker import GpuInfo
        from tui.screens.monitor import MonitorScreen

        gpus = [GpuInfo("0", "RTX", "8000", "16000", "78", "71", "210")]
        with patch("tui.common.plain_monitor._running_rows", AsyncMock(return_value=([], []))), \
            patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=gpus)), \
            patch("tui.common.docker.get_pcie_stats", AsyncMock(return_value={})):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = MonitorScreen(None)
                await app.push_screen(screen)
                for _ in range(30):
                    await pilot.pause()
                    if screen._rendered is not None:
                        break
                con = Console(width=120, file=io.StringIO())
                con.print(screen._rendered)
                out = con.file.getvalue()
                self.assertIn("GPU0", out)
                self.assertIn("210W", out)


class VersionScreenEnterTests(unittest.IsolatedAsyncioTestCase):
    """Enter on the version picker commits the highlighted option and starts,
    instead of only selecting the radio."""

    def _profile(self):
        import types
        return types.SimpleNamespace(
            name="p", image_tag="", port=8000, container_name="p",
            config_name="p", gpu_id="0", model_id="", enable_lora="false",
            tensor_parallel="1", extra_pip_packages="",
        )

    async def test_enter_commits_selection_and_starts(self) -> None:
        from tui.backends.vllm.screens import container as mod
        from tui.backends.vllm.screens.container import ContainerUpScreen, VER_OFFICIAL
        from textual.widgets import RadioSet, RadioButton

        with patch.object(mod, "load_profile", lambda n: self._profile()), \
            patch.object(mod, "get_local_latest_tag", AsyncMock(return_value="v0.1.0")), \
            patch.object(mod, "get_dockerhub_release_version", AsyncMock(return_value="v0.1.0")), \
            patch.object(mod, "get_dockerhub_nightly_date", AsyncMock(return_value="available")), \
            patch.object(mod, "get_gpu_info", AsyncMock(return_value=[])):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = ContainerUpScreen("p")
                await app.push_screen(screen)
                await pilot.pause()
                await pilot.pause()

                rs = screen.query_one("#version-radio", RadioSet)
                rs.query_one(f"#{VER_OFFICIAL}", RadioButton).value = True
                await pilot.pause()

                called = []
                screen._do_start = lambda: called.append(True)
                screen.action_confirm_start()
                self.assertEqual(called, [True])

    async def test_enter_on_empty_custom_focuses_input_not_start(self) -> None:
        from tui.backends.vllm.screens import container as mod
        from tui.backends.vllm.screens.container import ContainerUpScreen, VER_CUSTOM
        from textual.widgets import RadioSet, RadioButton, Input

        with patch.object(mod, "load_profile", lambda n: self._profile()), \
            patch.object(mod, "get_local_latest_tag", AsyncMock(return_value="v0.1.0")), \
            patch.object(mod, "get_dockerhub_release_version", AsyncMock(return_value="v0.1.0")), \
            patch.object(mod, "get_dockerhub_nightly_date", AsyncMock(return_value="available")), \
            patch.object(mod, "get_gpu_info", AsyncMock(return_value=[])):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = ContainerUpScreen("p")
                await app.push_screen(screen)
                await pilot.pause()
                await pilot.pause()

                rs = screen.query_one("#version-radio", RadioSet)
                rs.query_one(f"#{VER_CUSTOM}", RadioButton).value = True
                await pilot.pause()

                called = []
                screen._do_start = lambda: called.append(True)
                screen.action_confirm_start()
                # Empty custom tag → don't start, focus the input.
                self.assertEqual(called, [])
                self.assertIs(app.focused, screen.query_one("#custom-tag-input", Input))

    async def test_vllm_custom_latest_does_not_start(self) -> None:
        from tui.backends.vllm.screens import container as mod
        from tui.backends.vllm.screens.container import ContainerUpScreen, VER_CUSTOM
        from textual.widgets import Input, RadioButton, RadioSet

        with patch.object(mod, "load_profile", lambda n: self._profile()), \
            patch.object(mod, "get_local_latest_tag", AsyncMock(return_value="v0.1.0")), \
            patch.object(mod, "get_dockerhub_release_version", AsyncMock(return_value="v0.1.0")), \
            patch.object(mod, "get_dockerhub_nightly_date", AsyncMock(return_value="available")), \
            patch.object(mod, "get_gpu_info", AsyncMock(return_value=[])):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = ContainerUpScreen("p")
                await app.push_screen(screen)
                await pilot.pause()
                await pilot.pause()
                radio = screen.query_one("#version-radio", RadioSet)
                radio.query_one(f"#{VER_CUSTOM}", RadioButton).value = True
                await pilot.pause()
                custom = screen.query_one("#custom-tag-input", Input)
                custom.value = "custom/vllm:latest"
                called = []
                screen._do_start = lambda: called.append(True)
                screen.action_confirm_start()
                self.assertEqual(called, [])
                self.assertIs(app.focused, custom)

    async def test_llamacpp_custom_latest_does_not_start(self) -> None:
        from tui.backends.llamacpp.screens import container as mod
        from tui.backends.llamacpp.screens.container import ContainerUpScreen, VER_CUSTOM
        from textual.widgets import Input, RadioButton, RadioSet

        profile = mod.backend.Profile(name="p", container_name="p", port=8080)
        with patch.object(mod.backend, "load_profile", return_value=profile), \
            patch.object(mod, "get_dev_build_defaults", return_value=("repo", "main")), \
            patch.object(mod, "get_gpu_info", AsyncMock(return_value=[])), \
            patch.object(mod, "list_local_dev_images", AsyncMock(return_value=[])):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = ContainerUpScreen("p")
                await app.push_screen(screen)
                await pilot.pause()
                radio = screen.query_one("#version-radio", RadioSet)
                radio.query_one(f"#{VER_CUSTOM}", RadioButton).value = True
                await pilot.pause()
                custom = screen.query_one("#custom-tag-input", Input)
                custom.value = "ghcr.io/example/llama:latest"
                called = []
                screen._do_start = lambda: called.append(True)
                screen.action_confirm_start()
                self.assertEqual(called, [])
                self.assertIs(app.focused, custom)


class PlainModeToggleTests(unittest.IsolatedAsyncioTestCase):
    """The `t` key suspends the TUI, runs the plain dashboard, then reloads."""

    async def test_plain_mode_suspends_runs_monitor_and_reloads(self) -> None:
        import contextlib
        from unittest.mock import AsyncMock, MagicMock, patch
        from tui.screens.dashboard import DashboardScreen
        from tui.common.adapter import DashboardRow

        row = DashboardRow(backend="vllm", profile_name="m", container_name="m",
                           port=8000, running=True, model="m", detail="", gpu_id="0")
        with patch("tui.common.docker.running_container_names", AsyncMock(return_value=set())), \
            patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[])), \
            patch("tui.common.plain_monitor.run_plain_monitor", AsyncMock()) as run, \
            patch("tui.app.LlmuxApp.suspend", lambda self: contextlib.nullcontext()):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                dash = next(s for s in app.screen_stack if isinstance(s, DashboardScreen))
                dash._selected_row = MagicMock(return_value=row)
                dash._reload = MagicMock()
                await dash.action_plain_mode()
                run.assert_awaited_once()
                dash._reload.assert_called_once()


class DashboardReloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_docker_scan_failure_preserves_last_verified_state(self) -> None:
        from textual.widgets import Static
        from tui.common.adapter import DashboardRow

        row = DashboardRow(
            backend="vllm",
            profile_name="verified",
            container_name="verified",
            port=8000,
            running=True,
            model="org/model",
            detail="",
        )

        def rows_for_status(running):
            return [
                DashboardRow(
                    backend=row.backend,
                    profile_name=row.profile_name,
                    container_name=row.container_name,
                    port=row.port,
                    running=row.container_name in running,
                    model=row.model,
                    detail=row.detail,
                )
            ]

        with (
            patch(
                "tui.common.docker.running_container_names",
                AsyncMock(return_value={"verified"}),
            ) as docker_status,
            patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[])),
            patch(
                "tui.screens.dashboard.VllmAdapter.rows",
                side_effect=rows_for_status,
            ),
            patch("tui.screens.dashboard.LlamacppAdapter.rows", return_value=[]),
        ):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                dash = next(
                    screen
                    for screen in app.screen_stack
                    if isinstance(screen, DashboardScreen)
                )
                docker_status.side_effect = RuntimeError("daemon offline")
                dash._reload()
                await pilot.pause()

                self.assertTrue(dash._rows[0].running)
                status = str(dash.query_one("#status-bar", Static).render())
                self.assertIn("Docker status unavailable", status)

    async def test_created_profile_becomes_selected(self) -> None:
        from tui.common.adapter import DashboardRow

        rows = [
            DashboardRow(
                backend="vllm",
                profile_name="old",
                container_name="old",
                port=8000,
                running=False,
                model="org/old",
                detail="",
            )
        ]
        with (
            patch(
                "tui.common.docker.running_container_names",
                AsyncMock(return_value=set()),
            ),
            patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[])),
            patch(
                "tui.screens.dashboard.VllmAdapter.rows",
                side_effect=lambda _running: list(rows),
            ),
            patch("tui.screens.dashboard.LlamacppAdapter.rows", return_value=[]),
        ):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                dash = next(
                    screen
                    for screen in app.screen_stack
                    if isinstance(screen, DashboardScreen)
                )
                rows.append(
                    DashboardRow(
                        backend="vllm",
                        profile_name="new",
                        container_name="new",
                        port=8001,
                        running=False,
                        model="org/new",
                        detail="",
                    )
                )
                dash._after_mutation("new")
                await pilot.pause()

                self.assertEqual(dash._selected_row().profile_name, "new")


class ConfigFormRecipeMergeTests(unittest.IsolatedAsyncioTestCase):
    """Edit Config can pull a recipe: its params merge into the open form,
    updating rows that already exist instead of duplicating them."""

    async def test_apply_recipe_updates_and_adds_rows(self) -> None:
        from textual.widgets import Input, Switch
        from tui.backends.vllm.screens.config import ConfigFormScreen

        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            with _patched_vllm_config_dir(cdir):
                (cdir / "cfg.yaml").write_text(
                    "model: m/x\ngpu-memory-utilization: '0.9'\nmax-model-len: 4096\n"
                )
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ConfigFormScreen("cfg")
                    await app.push_screen(screen)
                    await pilot.pause()

                    screen._apply_recipe(
                        "org/other", {"max-model-len": 8192, "quantization": "awq"}
                    )
                    await pilot.pause()

                    rows = {
                        row.query_one(".param-key", Input).value: row
                        for row in screen.query(".param-row")
                    }
                    self.assertEqual(
                        rows["max-model-len"].query_one(".param-value", Input).value,
                        "8192",
                    )
                    self.assertTrue(
                        rows["max-model-len"].query_one(".param-switch", Switch).value
                    )
                    self.assertIn("quantization", rows)
                    self.assertEqual(
                        screen.query_one("#model-input", Input).value, "org/other"
                    )


class ActionMenuPrepareTests(unittest.IsolatedAsyncioTestCase):
    """`prepare` is offered only while the container is stopped — there is
    nothing to pre-download for a profile that is already serving."""

    async def test_vllm_menu_offers_prepare_only_when_stopped(self) -> None:
        from tui.backends.vllm.screens.dashboard import ProfileActionScreen

        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            for running, expected in ((False, True), (True, False)):
                screen = ProfileActionScreen("p", running)
                await app.push_screen(screen)
                await pilot.pause()
                ids = {o.id for o in screen.query_one("#action-list").options}
                self.assertEqual("prepare" in ids, expected)
                app.pop_screen()
                await pilot.pause()

    async def test_llamacpp_menu_offers_prepare_only_when_stopped(self) -> None:
        from tui.backends.llamacpp import backend as lbackend
        from tui.backends.llamacpp.screens.dashboard import ActionModal

        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            for running, expected in ((False, True), (True, False)):
                profile = lbackend.Profile(name="p", container_name="p", port=8080)
                profile.running = running
                screen = ActionModal(profile)
                await app.push_screen(screen)
                await pilot.pause()
                ids = {o.id for o in screen.query_one("#action-list").options}
                self.assertEqual("prepare" in ids, expected)
                app.pop_screen()
                await pilot.pause()


class QuickSetupRecipeSourceTests(unittest.IsolatedAsyncioTestCase):
    """Quick Setup can pull the recipe of a different model than the one being
    configured (a quantized checkpoint borrowing its base model's recipe)."""

    async def test_source_input_drives_the_fetch(self) -> None:
        from unittest.mock import MagicMock

        from textual.widgets import Input
        from tui.backends.vllm.screens.quick_setup import QuickSetupScreen

        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            screen = QuickSetupScreen()
            await app.push_screen(screen)
            await pilot.pause()

            screen.query_one("#model-input", Input).value = "cpatonn/Qwen3-32B-AWQ"
            screen.query_one("#recipe-model-input", Input).value = "Qwen/Qwen3-32B"
            screen._fetch_recipe = MagicMock()
            screen.query_one("#fetch-recipe-btn").press()
            await pilot.pause()

            screen._fetch_recipe.assert_called_once_with(
                "Qwen/Qwen3-32B", "cpatonn/Qwen3-32B-AWQ"
            )

    async def test_blank_source_falls_back_to_the_model(self) -> None:
        from unittest.mock import MagicMock

        from textual.widgets import Input
        from tui.backends.vllm.screens.quick_setup import QuickSetupScreen

        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            screen = QuickSetupScreen()
            await app.push_screen(screen)
            await pilot.pause()

            screen.query_one("#model-input", Input).value = "Qwen/Qwen3-32B"
            screen._fetch_recipe = MagicMock()
            screen.query_one("#fetch-recipe-btn").press()
            await pilot.pause()

            screen._fetch_recipe.assert_called_once_with(
                "Qwen/Qwen3-32B", "Qwen/Qwen3-32B"
            )


class DashboardUpdateCheckTests(unittest.IsolatedAsyncioTestCase):
    """`U` on the dashboard checks for a release on demand, ignoring the
    failure back-off that the startup check honors."""

    async def _run_check(self, status, blocked=""):
        from unittest.mock import MagicMock

        from tui.common import version_check as vc

        with patch("tui.common.docker.running_container_names", AsyncMock(return_value=set())), \
             patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[])), \
             patch.object(vc, "resolve_status", return_value=status) as resolve, \
             patch.object(vc, "update_blocked_reason", return_value=blocked):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                dash = next(s for s in app.screen_stack if isinstance(s, DashboardScreen))
                dash.notify = MagicMock()
                dash.action_check_update()
                await app.workers.wait_for_complete()
                await pilot.pause()
                pushed = [type(s).__name__ for s in app.screen_stack]
                return dash.notify, resolve, pushed

    async def test_up_to_date_notifies_and_asks_nothing(self) -> None:
        from tui.common import version_check as vc

        notify, resolve, pushed = await self._run_check(
            vc.UpdateStatus(vc.CURRENT, tag="v2.6.0", local_version="2.6.0")
        )
        messages = " ".join(str(c.args[0]) for c in notify.call_args_list)
        self.assertIn("v2.6.0", messages)
        self.assertNotIn("ConfirmModal", pushed)
        self.assertEqual(resolve.call_args.kwargs, {"respect_cooldown": False})

    async def test_behind_offers_the_update(self) -> None:
        from tui.common import version_check as vc

        _, _, pushed = await self._run_check(
            vc.UpdateStatus(vc.BEHIND, tag="v9.9.9", url="u")
        )
        self.assertIn("ConfirmModal", pushed)

    async def test_dirty_checkout_warns_instead_of_offering(self) -> None:
        from tui.common import version_check as vc

        notify, _, pushed = await self._run_check(
            vc.UpdateStatus(vc.BEHIND, tag="v9.9.9", url="u"),
            blocked="checkout has uncommitted changes to tracked files",
        )
        messages = " ".join(str(c.args[0]) for c in notify.call_args_list)
        self.assertIn("uncommitted", messages)
        self.assertNotIn("ConfirmModal", pushed)

    async def test_successful_update_exits_the_old_process(self) -> None:
        from unittest.mock import MagicMock

        from tui.common import version_check as vc

        with patch("tui.common.docker.running_container_names", AsyncMock(return_value=set())), \
             patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[])), \
             patch.object(vc, "apply_update", return_value=(True, "updated")):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                dash = next(s for s in app.screen_stack if isinstance(s, DashboardScreen))
                dash.notify = MagicMock()
                app.exit = MagicMock()
                dash._apply_update("v9.9.9")
                await app.workers.wait_for_complete()

                app.exit.assert_called_once_with()


class SystemOperationTests(unittest.IsolatedAsyncioTestCase):
    async def test_pull_image_collects_stream_result(self) -> None:
        from tui.common import system_operations

        async def events():
            yield "log", "pulling"
            yield "rc", 0

        with patch.object(system_operations, "stream_pull", return_value=events()):
            rc, lines = await system_operations.pull_image("registry.example/model:v1")

        self.assertEqual(rc, 0)
        self.assertEqual(lines, ["pulling"])

    async def test_llamacpp_build_passes_multi_arch_to_backend(self) -> None:
        from tui.common import system_operations

        captured = {}

        async def events():
            yield "rc", 0

        def build(branch, **kwargs):
            captured.update({"branch": branch, **kwargs})
            return events()

        with patch(
            "tui.backends.llamacpp.backend_runtime._stream_build_dev_image",
            side_effect=build,
        ):
            rc, _ = await system_operations.build_dev_image(
                "llamacpp",
                repo_url="https://example.test/llama.cpp.git",
                branch="main",
                custom_tag="dev",
                cuda_arch="89",
                multi_arch=True,
            )

        self.assertEqual(rc, 0)
        self.assertTrue(captured["use_multi_arch"])

    def test_environment_status_never_returns_token_value(self) -> None:
        from tui.common import system_operations

        with tempfile.TemporaryDirectory() as tmp:
            common = Path(tmp) / ".env.common"
            secret = "hf_do_not_render_this"
            common.write_text(
                f"HF_TOKEN={secret}\nHF_CACHE_PATH={Path(tmp) / 'cache'}\n"
            )
            _, messages = system_operations.environment_status(common)

        self.assertNotIn(secret, "\n".join(messages))

    def test_render_backend_envs_reports_each_failed_profile(self) -> None:
        from types import SimpleNamespace
        from tui.common import system_operations

        profiles = [
            SimpleNamespace(name="ok", backend="vllm"),
            SimpleNamespace(name="bad", backend="vllm"),
        ]

        def render(name, backend):
            self.assertEqual(backend, "vllm")
            if name == "bad":
                raise ValueError("invalid config")
            return Path("/runtime/ok.env")

        with patch.object(system_operations.profile_store, "list_profiles", return_value=profiles), \
            patch.object(
                system_operations.profile_store,
                "render_env_for_profile",
                side_effect=render,
            ):
            rendered, failures = system_operations.render_backend_envs("vllm")

        self.assertEqual(rendered, [Path("/runtime/ok.env")])
        self.assertEqual(failures, ["bad: invalid config"])


class SystemScreenParityTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_system_screens_expose_image_and_environment_actions(self) -> None:
        from unittest.mock import MagicMock
        from tui.backends.llamacpp.screens.system import SystemScreen as LlamaSystem
        from tui.backends.vllm.screens.system import SystemScreen as VllmSystem

        ids = (
            "#btn-pull-image",
            "#btn-remove-image",
            "#btn-build-image",
            "#btn-validate-env",
            "#btn-render-envs",
        )
        with patch("tui.common.docker.running_container_names", AsyncMock(return_value=set())), \
            patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[])):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                for screen_type in (VllmSystem, LlamaSystem):
                    with patch.object(screen_type, "_refresh_gpu", MagicMock()), \
                        patch.object(screen_type, "_refresh_images", MagicMock()), \
                        patch.object(screen_type, "_refresh_containers", MagicMock()), \
                        patch.object(screen_type, "_refresh_disk", MagicMock()):
                        screen = screen_type()
                        await app.push_screen(screen)
                        await pilot.pause()
                        for selector in ids:
                            self.assertIsNotNone(screen.query_one(selector))
                        app.pop_screen()
                        await pilot.pause()
