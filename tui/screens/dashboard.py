"""Unified Dashboard — vLLM + llama.cpp 프로필을 한 DataTable 에 통합."""

from __future__ import annotations

import asyncio
import math
import re
from time import monotonic

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from tui.backends.llamacpp import backend as lbackend
from tui.backends.llamacpp import backend_runtime as lruntime
from tui.backends.llamacpp.adapter import LlamacppAdapter
from tui.backends.vllm import backend as vbackend
from tui.backends.vllm.adapter import VllmAdapter
from tui.common import docker as common_docker
from tui.common.adapter import DashboardRow
from tui.common.conflicts import (
    external_port_conflicts,
    gpu_conflicts,
    port_conflicts,
)
from tui.common.http import BENCH_RUNS, BENCH_WARMUP, list_served_models, run_bench
from tui.common.i18n import t
from tui.common.mem import estimate_model_memory
from tui.common.metrics import (
    MetricsUnavailableError,
    ThroughputTracker,
    fetch_token_counters,
)
from tui.common.profile_store import (
    clone_profile,
    load_profile,
    render_env_for_profile,
)
from tui.common.widgets import BackendPickerModal, ConfirmModal, TextPromptModal


class DashboardScreen(Screen):
    """두 backend 프로필을 단일 DataTable 로 통합 표시."""

    BINDINGS = [
        Binding("enter", "action_menu", t("Action", "작업")),
        Binding("n", "new_profile", t("New", "새로")),
        Binding("m", "mem_estimate", t("Memory", "메모리")),
        Binding("C", "config_list", t("Configs", "설정")),
        Binding("s", "system_info", t("System", "시스템")),
        Binding("t", "plain_mode", t("Terminal", "터미널")),
        Binding("r", "refresh", t("Refresh", "새로고침")),
        Binding("q", "quit", t("Quit", "종료")),
        Binding("u", "start_container", show=False),
        Binding("p", "prepare_profile", show=False),
        Binding("U", "check_update", show=False),
        Binding("d", "stop_container", show=False),
        Binding("l", "view_logs", show=False),
        Binding("v", "monitor", show=False),
        Binding("e", "edit_profile", show=False),
        Binding("c", "edit_config", show=False),
        Binding("x", "delete_profile", show=False),
        Binding("escape", "hide_mem_search", show=False),
        Binding("question_mark", "help", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._vllm = VllmAdapter()
        self._llamacpp = LlamacppAdapter()
        self._rows: list[DashboardRow] = []
        self._container_snapshots: dict[str, common_docker.ContainerSnapshot] = {}
        self._gpus = []
        self._gpu_scan_error = ""
        self._scan_errors: set[str] = set()
        self._refresh_timer = None
        self._tps: dict[str, str] = {}
        self._tps_errors: dict[str, str] = {}
        self._tps_tracker = ThroughputTracker()
        self._preferred_profile_name: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="status-bar")
        yield DataTable(id="profile-table", cursor_type="row")
        yield Static(
            t(
                "\n  No profiles yet — press [b]n[/b] to create one\n",
                "\n  아직 프로필이 없습니다 — [b]n[/b] 키로 하나 만드세요\n",
            ),
            id="empty-state",
        )
        yield Static("", id="gpu-bar")
        with Horizontal(id="mem-search-area"):
            yield Static(" 🔍 ", id="search-icon")
            yield Input(
                placeholder="Estimate HF model memory (press m then type, Enter to run)",
                id="mem-search-input",
            )
        yield Static("", id="mem-result-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#profile-table", DataTable)
        self._col_keys = table.add_columns(
            "Status", "Backend", "Profile", "Port", "tok/s", "Model", "Detail"
        )
        self._tps_col_key = self._col_keys[4]
        self._reload()
        self._poll_gpu()
        self._poll_throughput()
        self._tps_timer = self.set_interval(3.0, lambda: self._poll_throughput())
        self._refresh_timer = self.set_interval(5.0, lambda: self._reload())
        self._gpu_timer = self.set_interval(2.0, lambda: self._poll_gpu())

    def on_screen_suspend(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.pause()
        if getattr(self, "_gpu_timer", None) is not None:
            self._gpu_timer.pause()
        if getattr(self, "_tps_timer", None) is not None:
            self._tps_timer.pause()

    def on_screen_resume(self) -> None:
        self._reload()
        self._poll_gpu()
        if self._refresh_timer is not None:
            self._refresh_timer.resume()
        if getattr(self, "_gpu_timer", None) is not None:
            self._gpu_timer.resume()
        if getattr(self, "_tps_timer", None) is not None:
            self._tps_timer.resume()

    @work(exclusive=True, group="dashboard-reload")
    async def _reload(self) -> None:
        try:
            running = await common_docker.running_container_names()
        except Exception as exc:
            self._scan_errors = {"docker"}
            self.notify(
                t(f"Docker status scan failed: {exc}",
                  f"Docker 상태 스캔 실패: {exc}"),
                severity="error",
                timeout=8,
            )
            self.query_one("#status-bar", Static).update(
                t(
                    " [red]Docker status unavailable[/] · showing last verified state",
                    " [red]Docker 상태 확인 불가[/] · 마지막 확인 상태 표시 중",
                )
            )
            return

        self._container_snapshots = getattr(running, "snapshots", {})

        rows: list[DashboardRow] = []
        scan_errors: set[str] = set()
        try:
            rows.extend(self._vllm.rows(running))
        except Exception as exc:
            scan_errors.add("vllm")
            rows.extend(row for row in self._rows if row.backend == "vllm")
            self.notify(t(f"vLLM scan failed: {exc}", f"vLLM 스캔 실패: {exc}"),
                        severity="error")
        try:
            rows.extend(self._llamacpp.rows(running))
        except Exception as exc:
            scan_errors.add("llamacpp")
            rows.extend(row for row in self._rows if row.backend == "llamacpp")
            self.notify(t(f"llama.cpp scan failed: {exc}", f"llama.cpp 스캔 실패: {exc}"),
                        severity="error")
        self._scan_errors = scan_errors
        rows.sort(key=lambda r: (not r.running, r.backend, r.profile_name))
        self._rows = rows
        self._render_rows(rows)

    def _render_rows(self, rows: list[DashboardRow]) -> None:
        table = self.query_one("#profile-table", DataTable)
        empty = self.query_one("#empty-state")
        status_bar = self.query_one("#status-bar", Static)

        prev_key: str | None = None
        if table.row_count > 0:
            try:
                row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
                prev_key = str(row_key.value)
            except (KeyError, IndexError):
                pass

        table.clear()

        if not rows:
            table.styles.display = "none"
            empty.styles.display = "block"
            if self._scan_errors:
                failed = ", ".join(sorted(self._scan_errors))
                status_bar.update(f" [red]Profile status unavailable:[/] {failed}")
            else:
                status_bar.update("")
            return

        table.styles.display = "block"
        empty.styles.display = "none"

        v_run = sum(1 for r in rows if r.backend == "vllm" and r.running)
        l_run = sum(1 for r in rows if r.backend == "llamacpp" and r.running)
        v_total = sum(1 for r in rows if r.backend == "vllm")
        l_total = sum(1 for r in rows if r.backend == "llamacpp")
        status = (
            f" [#4c8dff]vLLM[/] {v_run}/{v_total}  ·  "
            f"[green]llama.cpp[/] {l_run}/{l_total}  ·  "
            + t("[dim]Enter = actions[/dim]", "[dim]Enter = 작업 메뉴[/dim]")
        )
        if self._scan_errors:
            failed = ", ".join(sorted(self._scan_errors))
            status += f"  ·  [red]{failed} status unknown[/]"
        if self._gpu_scan_error:
            status += "  ·  [red]GPU status unknown[/]"
        status_bar.update(status)

        for r in rows:
            backend_cell = (
                "[#4c8dff]vLLM[/]" if r.backend == "vllm" else "[green]llama.cpp[/]"
            )
            status_cell = self._container_status_cell(r)
            port_cell = str(r.port) if r.port is not None else "—"
            model_short = r.model.split("/")[-1] if "/" in r.model else (r.model or "—")
            detail = r.detail or "—"
            key = f"{r.backend}:{r.profile_name}"
            tps_cell = self._tps.get(key, "—") if r.running else "—"
            table.add_row(
                status_cell,
                backend_cell,
                r.profile_name,
                port_cell,
                tps_cell,
                model_short,
                detail,
                key=key,
            )

        preferred_key: str | None = None
        if self._preferred_profile_name is not None:
            for row in rows:
                if row.profile_name == self._preferred_profile_name:
                    preferred_key = f"{row.backend}:{row.profile_name}"
                    break
            self._preferred_profile_name = None

        target_key = preferred_key or prev_key
        if target_key is not None:
            for idx, r in enumerate(rows):
                if f"{r.backend}:{r.profile_name}" == target_key:
                    try:
                        table.move_cursor(row=idx)
                    except Exception:
                        pass
                    break

    def _container_status_cell(self, row: DashboardRow) -> str:
        snapshot = self._container_snapshots.get(row.container_name)
        status = snapshot.display_status if snapshot is not None else (
            "running" if row.running else "stopped"
        )
        if status in {"running", "healthy"}:
            return f"[green]● {status}[/]"
        if status in {"starting", "paused", "restarting", "removing"}:
            return f"[yellow]● {status}[/]"
        if status in {"unhealthy", "dead", "unknown"}:
            return f"[red]● {status}[/]"
        return f"[dim]○ {status}[/]"

    @work(exclusive=True, group="dashboard-gpu")
    async def _poll_gpu(self) -> None:
        try:
            gpus = await common_docker.get_gpu_info()
        except Exception as exc:
            self._gpu_scan_error = str(exc)
            last_known = ""
            if self._gpus:
                try:
                    last_known = common_docker.format_gpu_bar(self._gpus)
                except Exception as render_exc:
                    self._gpu_scan_error += f"; last verified GPU rendering failed: {render_exc}"
            message = t(
                f" [red]GPU status unavailable:[/] {exc}",
                f" [red]GPU 상태 확인 불가:[/] {exc}",
            )
            if last_known:
                message += t(" · last verified: ", " · 마지막 확인: ") + last_known
            self.query_one("#gpu-bar", Static).update(message)
            self.notify(
                t(f"GPU scan failed: {exc}", f"GPU 스캔 실패: {exc}"),
                severity="error",
                timeout=8,
            )
            return
        try:
            rendered = common_docker.format_gpu_bar(gpus)
        except Exception as exc:
            self._gpu_scan_error = f"GPU rendering failed: {exc}"
            self.query_one("#gpu-bar", Static).update(
                t(
                    f" [red]GPU rendering failed:[/] {exc}",
                    f" [red]GPU 렌더링 실패:[/] {exc}",
                )
            )
            self.notify(self._gpu_scan_error, severity="error", timeout=8)
            return
        self._gpus = gpus
        self._gpu_scan_error = ""
        self.query_one("#gpu-bar", Static).update(rendered)

    @work(exclusive=True, group="dashboard-tps")
    async def _poll_throughput(self) -> None:
        live: list[DashboardRow] = []
        for r in list(self._rows):
            key = f"{r.backend}:{r.profile_name}"
            if not r.running or not r.port:
                self._tps.pop(key, None)
                self._tps_errors.pop(key, None)
                self._tps_tracker.forget(key)
                continue
            live.append(r)

        samples = await asyncio.gather(
            *(fetch_token_counters(r.port) for r in live),
            return_exceptions=True,
        )
        now = monotonic()

        for r, counters in zip(live, samples):
            key = f"{r.backend}:{r.profile_name}"
            if isinstance(counters, MetricsUnavailableError):
                message = str(counters)
                if self._tps_errors.get(key) != message:
                    self.notify(message, severity="error", timeout=8)
                self._tps_errors[key] = message
                self._tps[key] = "[red]error[/]"
                self._tps_tracker.forget(key)
                cell = "[red]error[/]"
            elif isinstance(counters, BaseException):
                raise counters
            if counters is None:
                self._tps_errors.pop(key, None)
                self._tps.pop(key, None)
                self._tps_tracker.forget(key)
                cell = "—"
            elif not isinstance(counters, BaseException):
                self._tps_errors.pop(key, None)
                rate = self._tps_tracker.update(key, counters, now)
                if rate is None:
                    self._tps.pop(key, None)
                    cell = "—"
                else:
                    cell = f"{rate[1]:.1f}"
                    self._tps[key] = cell

            try:
                table = self.query_one("#profile-table", DataTable)
                table.update_cell(key, self._tps_col_key, cell)
            except Exception:
                pass

    def _selected_row(self) -> DashboardRow | None:
        table = self.query_one("#profile-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            key_val = str(row_key.value) if row_key is not None else None
        except (KeyError, IndexError):
            return None
        if not key_val:
            return None
        for r in self._rows:
            if f"{r.backend}:{r.profile_name}" == key_val:
                return r
        return None

    @on(DataTable.RowSelected, "#profile-table")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.action_action_menu()

    def action_action_menu(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.backend == "vllm":
            self._push_vllm_action(row)
        else:
            self._push_llamacpp_action(row)

    def _push_vllm_action(self, row: DashboardRow) -> None:
        from tui.backends.vllm.screens.dashboard import ProfileActionScreen

        def after(action: str | None) -> None:
            if action:
                self._dispatch_vllm(action, row)

        self.app.push_screen(ProfileActionScreen(row.profile_name, row.running), after)

    def _push_llamacpp_action(self, row: DashboardRow) -> None:
        from tui.backends.llamacpp.screens.dashboard import ActionModal

        profile = lbackend.load_profile(row.profile_name)
        profile.running = row.running

        def after(action: str | None) -> None:
            if action:
                self._dispatch_llamacpp(action, row, profile)

        self.app.push_screen(ActionModal(profile), after)

    def _confirm_conflicts_before_start(self, row: DashboardRow, on_ok) -> None:
        self._check_and_confirm(row, on_ok)

    @work(exclusive=False, group="conflict-check")
    async def _check_and_confirm(self, row: DashboardRow, on_ok) -> None:
        modal_generation = self.app.modal_generation
        port_msgs = port_conflicts(row, self._rows)
        gpu_msgs = gpu_conflicts(row, self._rows)
        probe_msgs: list[str] = []
        if self._scan_errors:
            failed = ", ".join(sorted(self._scan_errors))
            probe_msgs.append(
                "Could not verify every profile's port/GPU assignment because "
                f"these scans failed: {failed}."
            )
        try:
            ext_ports = await common_docker.running_container_ports()
            ext_msgs = external_port_conflicts(row, self._rows, ext_ports)
        except Exception as exc:
            ext_msgs = []
            probe_msgs.append(
                "Could not inspect running Docker container ports. "
                f"Runtime port check will still run before start. ({exc})"
            )

        if not self.app.can_push_modal(modal_generation):
            return

        if not port_msgs and not gpu_msgs and not ext_msgs and not probe_msgs:
            on_ok()
            return

        lines: list[str] = []
        if probe_msgs:
            lines.append("[b]Port probe warning:[/b]")
            lines += [f"  • {m}" for m in probe_msgs]
        if port_msgs:
            if lines:
                lines.append("")
            lines.append("[b]Port conflict (llmux):[/b]")
            lines += [f"  • {m}" for m in port_msgs]
        if ext_msgs:
            if lines:
                lines.append("")
            lines.append("[b]Port conflict (external):[/b]")
            lines += [f"  • {m}" for m in ext_msgs]
        if gpu_msgs:
            if lines:
                lines.append("")
            lines.append("[b]GPU conflict:[/b]")
            lines += [f"  • {m}" for m in gpu_msgs]
        hard_conflict = bool(port_msgs or ext_msgs or probe_msgs)
        lines.append("")
        lines.append(
            "Resolve the port check before starting."
            if hard_conflict
            else "Proceed despite the GPU overlap?"
        )
        message = "\n".join(lines)

        if hard_conflict:
            self.app.push_screen(
                ConfirmModal(message, confirm_label="Close", variant="error"),
                lambda _confirmed: None,
            )
            return

        def after(proceed: bool) -> None:
            if proceed:
                on_ok()

        self.app.push_screen(
            ConfirmModal(message, confirm_label="Start with GPU overlap", variant="warning"),
            after,
        )


    def _dispatch_vllm(self, action: str, row: DashboardRow) -> None:
        name = row.profile_name
        if action == "start":
            from tui.backends.vllm.screens.container import ContainerUpScreen

            def launch() -> None:
                self.app.push_screen(
                    ContainerUpScreen(name),
                    self._after_mutation,
                )

            self._confirm_conflicts_before_start(row, launch)
            return
        elif action == "prepare":
            from tui.screens.prepare import PrepareScreen

            self.app.push_screen(PrepareScreen(name, "vllm"), self._after_mutation)
        elif action == "stop":
            self._confirm_vllm_stop(name)
        elif action == "logs":
            from tui.backends.vllm.screens.container import LogScreen

            p = vbackend.load_profile(name)
            self.app.push_screen(LogScreen(p.container_name))
        elif action == "monitor":
            from tui.screens.monitor import MonitorScreen
            self.app.push_screen(MonitorScreen(row))
        elif action == "benchmark":
            self._run_vllm_bench(row)
        elif action == "edit_profile":
            from tui.backends.vllm.screens.profile import ProfileFormScreen

            p = vbackend.load_profile(name)
            self.app.push_screen(ProfileFormScreen(p), self._after_mutation)
        elif action == "clone":
            self._prompt_clone(name, "vllm")
        elif action == "render_env":
            self._render_profile_env(name, "vllm")
        elif action == "edit_config":
            from tui.backends.vllm.screens.config import ConfigFormScreen

            p = vbackend.load_profile(name)
            cfg = p.config_name or name
            self.app.push_screen(
                ConfigFormScreen(config_name=cfg), self._after_mutation
            )
        elif action == "delete":
            if row.running:
                self.notify(
                    t("Cannot delete: container is running. Stop it first.",
                      "삭제 불가: 컨테이너가 실행 중입니다. 먼저 중지하세요."),
                    severity="error",
                )
                return
            from tui.backends.vllm.screens.profile import ProfileDeleteScreen

            self.app.push_screen(ProfileDeleteScreen(name), self._after_mutation)

    def _confirm_vllm_stop(self, name: str) -> None:
        def on_ok(ok: bool) -> None:
            if ok:
                self._run_vllm_stop(name)

        self.app.push_screen(
            ConfirmModal(
                t(f"Stop vLLM container [b]{name}[/b]?",
                  f"vLLM 컨테이너 [b]{name}[/b] 을 중지할까요?"),
                confirm_label=t("Yes, stop", "네, 중지"),
            ),
            on_ok,
        )

    @work(exclusive=True, group="benchmark")
    async def _run_vllm_bench(self, row: DashboardRow) -> None:
        if not row.port:
            self.notify(t("No port information", "포트 정보 없음"), severity="error")
            return
        try:
            from tui.common.http import profile_api_key

            api_key = await profile_api_key("vllm", row.profile_name)
            models = await list_served_models(row.port, **({"api_key": api_key} if api_key else {}))
            model = models[0] if models else (row.model or "")
            if not model:
                raise RuntimeError(
                    "could not identify a served model (/v1/models returned nothing)"
                )
            self.notify(
                t(
                    f"Benchmarking (warmup + {BENCH_RUNS} runs, {model})…",
                    f"벤치마크 실행 (warmup+{BENCH_RUNS}회, {model})…",
                )
            )
            r = await run_bench(
                row.port,
                model,
                **({"api_key": api_key} if api_key else {}),
                runs=BENCH_RUNS,
                warmup=BENCH_WARMUP,
            )
            self.notify(
                f"✓ median [b]{r['median_tps']:.1f} tok/s[/b] "
                f"({r['min_tps']:.1f}–{r['max_tps']:.1f})",
                title=row.profile_name,
                timeout=10,
            )
        except Exception as exc:
            self.notify(t(f"✗ benchmark failed: {exc}", f"✗ 벤치마크 실패: {exc}"),
                        severity="error")

    @work(exclusive=False)
    async def _run_vllm_stop(self, name: str) -> None:
        self.notify(t(f"Stopping {name}…", f"{name} 중지 중…"))
        rc, output = await vbackend.container_down(name)
        if rc == 0:
            self.notify(t(f"Stopped {name}.", f"{name} 중지됨."))
        else:
            self.notify(t(f"Error stopping {name}: {output}",
                          f"{name} 중지 오류: {output}"), severity="error")
        self._reload()
        self._poll_gpu()


    def _dispatch_llamacpp(
        self, action: str, row: DashboardRow, profile
    ) -> None:
        name = row.profile_name
        if action == "start":
            from tui.backends.llamacpp.screens.container import ContainerUpScreen

            def launch() -> None:
                self.app.push_screen(ContainerUpScreen(name), self._after_mutation)

            self._confirm_conflicts_before_start(row, launch)
            return
        elif action == "prepare":
            from tui.screens.prepare import PrepareScreen

            self.app.push_screen(PrepareScreen(name, "llamacpp"), self._after_mutation)
        elif action == "stop":
            self._confirm_llamacpp_stop(name)
        elif action == "logs":
            from tui.backends.llamacpp.screens.dashboard import LogViewer

            self.app.push_screen(LogViewer(profile.container_name))
        elif action == "monitor":
            from tui.screens.monitor import MonitorScreen
            self.app.push_screen(MonitorScreen(row))
        elif action == "benchmark":
            self._run_llamacpp_bench(profile)
        elif action == "edit-config":
            from tui.backends.llamacpp.screens.config import ConfigFormScreen

            self.app.push_screen(
                ConfigFormScreen(profile.config_name or name), self._after_mutation
            )
        elif action == "edit-profile":
            from tui.backends.llamacpp.screens.profile import ProfileFormScreen

            self.app.push_screen(ProfileFormScreen(profile), self._after_mutation)
        elif action == "clone-profile":
            self._prompt_clone(name, "llamacpp")
        elif action == "render-env":
            self._render_profile_env(name, "llamacpp")
        elif action == "delete-profile":
            from tui.backends.llamacpp.screens.profile import ProfileDeleteScreen

            self.app.push_screen(ProfileDeleteScreen(name), self._after_mutation)

    def _prompt_clone(self, source: str, backend: str) -> None:
        def after(destination: str | None) -> None:
            if not destination:
                return
            try:
                clone = clone_profile(source, destination, backend)
            except ValueError as exc:
                self.notify(str(exc), severity="error", timeout=8)
                return
            self.notify(
                t(
                    f"Cloned {source} → {clone.name}. Change port/GPU before running both.",
                    f"{source} → {clone.name} 복제됨. 둘 다 실행하기 전에 포트/GPU를 변경하세요.",
                ),
                timeout=8,
            )
            self._after_mutation(clone.name)

        self.app.push_screen(
            TextPromptModal(
                t(f"Clone {source} as", f"{source} 복제 이름"),
                placeholder="new-profile",
            ),
            after,
        )

    def _render_profile_env(self, name: str, backend: str) -> None:
        profile = load_profile(name, backend)
        if profile is None:
            self.notify(
                t(f"Profile not found: {name}", f"프로필을 찾을 수 없습니다: {name}"),
                severity="error",
            )
            return
        try:
            path = render_env_for_profile(profile.name, profile.backend)
        except (OSError, RuntimeError, ValueError) as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(t(f"Rendered {path}", f"렌더링 완료: {path}"), timeout=8)

    def _confirm_llamacpp_stop(self, name: str) -> None:
        def on_ok(ok: bool) -> None:
            if ok:
                self._run_llamacpp_stop(name)

        self.app.push_screen(
            ConfirmModal(
                t(f"Stop llama.cpp container [b]{name}[/b]?",
                  f"llama.cpp 컨테이너 [b]{name}[/b] 을 중지할까요?"),
                confirm_label=t("Yes, stop", "네, 중지"),
            ),
            on_ok,
        )

    @work(exclusive=False)
    async def _run_llamacpp_stop(self, name: str) -> None:
        self.notify(t(f"Stopping {name}…", f"{name} 중지 중…"))
        code, out = await lruntime.container_down(name)
        if code == 0:
            self.notify(t(f"✓ Stopped '{name}'", f"✓ '{name}' 중지"))
        else:
            tail = out.splitlines()[-3:] if out else []
            msg = " / ".join(tail) if tail else f"code={code}"
            self.notify(t(f"✗ stop failed: {msg}", f"✗ 중지 실패: {msg}"),
                        severity="error")
        self._reload()
        self._poll_gpu()

    @work(exclusive=True, group="benchmark")
    async def _run_llamacpp_bench(self, profile) -> None:
        try:
            config_name = profile.config_name or profile.name
            cfg = lbackend.load_config(config_name)
            from tui.common.http import config_api_key

            api_key = await config_api_key(cfg.params, profile.container_name, backend="llamacpp", env=profile.env_vars)
            alias = cfg.get("alias", config_name)
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(f"invalid benchmark alias in config {config_name!r}")
            if not profile.port:
                raise ValueError(f"profile {profile.name!r} has no metrics port")
            self.notify(
                t(
                    f"Benchmarking (warmup + {BENCH_RUNS} runs, {alias})…",
                    f"벤치마크 실행 (warmup+{BENCH_RUNS}회, {alias})…",
                )
            )
            r = await run_bench(
                profile.port,
                alias,
                **({"api_key": api_key} if api_key else {}),
                runs=BENCH_RUNS,
                warmup=BENCH_WARMUP,
            )
            self.notify(
                f"✓ median [b]{r['median_tps']:.1f} tok/s[/b] "
                f"({r['min_tps']:.1f}–{r['max_tps']:.1f})",
                title=profile.name,
                timeout=10,
            )
        except Exception as exc:
            self.notify(t(f"✗ benchmark failed: {exc}", f"✗ 벤치마크 실패: {exc}"),
                        severity="error")

    def action_start_container(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.running:
            self.notify(t("Container already running.", "컨테이너가 이미 실행 중입니다."),
                        severity="warning", timeout=3)
            return
        if row.backend == "vllm":
            self._dispatch_vllm("start", row)
        else:
            self._dispatch_llamacpp(
                "start", row, lbackend.load_profile(row.profile_name)
            )

    def action_prepare_profile(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.running:
            self.notify(
                t("Container is running — nothing to prepare.",
                  "컨테이너가 실행 중입니다 — 준비할 것이 없습니다."),
                severity="warning", timeout=3,
            )
            return
        if row.backend == "vllm":
            self._dispatch_vllm("prepare", row)
        else:
            self._dispatch_llamacpp(
                "prepare", row, lbackend.load_profile(row.profile_name)
            )

    def action_check_update(self) -> None:
        self._check_update()

    @work(exclusive=True, group="update-check")
    async def _check_update(self) -> None:
        from tui.common import version_check as vc

        modal_generation = self.app.modal_generation
        self.notify(t("Checking for updates…", "업데이트 확인 중…"), timeout=3)
        status = await asyncio.to_thread(vc.resolve_status, respect_cooldown=False)

        if status.state == vc.UNKNOWN:
            self.notify(
                t(f"Could not check for updates — {status.detail}",
                  f"업데이트를 확인할 수 없습니다 — {status.detail}"),
                severity="error", timeout=8,
            )
            return
        if status.state == vc.CURRENT:
            version = status.tag or status.local_version
            self.notify(
                t(f"llmux is up to date ({version}).",
                  f"llmux 가 최신입니다 ({version})."),
                timeout=5,
            )
            return

        blocked = vc.update_blocked_reason()
        if blocked:
            self.notify(
                t(f"{status.tag} is available, but auto-update is refused — {blocked}",
                  f"{status.tag} 이(가) 있지만 자동 업데이트 불가 — {blocked}"),
                severity="warning", timeout=10,
            )
            return

        if not self.app.can_push_modal(modal_generation):
            return

        def after(confirmed: bool) -> None:
            if confirmed:
                self._apply_update(status.tag)

        self.app.push_screen(
            ConfirmModal(
                t(f"llmux [b]{status.tag}[/b] is available. Update now?",
                  f"llmux [b]{status.tag}[/b] 이(가) 있습니다. 지금 업데이트할까요?"),
                confirm_label=t("Update", "업데이트"),
            ),
            after,
        )

    @work(exclusive=True, group="update-apply")
    async def _apply_update(self, tag: str) -> None:
        from tui.common import version_check as vc

        self.notify(t(f"Updating to {tag}…", f"{tag} 로 업데이트 중…"), timeout=5)
        ok, message = await asyncio.to_thread(vc.apply_update, tag)
        self.notify(
            message,
            severity="information" if ok else "error",
            timeout=15 if ok else 20,
        )
        if ok:
            self.app.exit()

    def action_stop_container(self) -> None:
        row = self._selected_row()
        if row is None or not row.running:
            return
        if row.backend == "vllm":
            self._confirm_vllm_stop(row.profile_name)
        else:
            self._confirm_llamacpp_stop(row.profile_name)

    def action_view_logs(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.backend == "vllm":
            self._dispatch_vllm("logs", row)
        else:
            self._dispatch_llamacpp(
                "logs", row, lbackend.load_profile(row.profile_name)
            )

    async def action_plain_mode(self) -> None:
        row = self._selected_row()
        from tui.common.plain_monitor import run_plain_monitor

        focus = row.profile_name if row is not None and row.running else None
        with self.app.suspend():
            await run_plain_monitor(focus)
        self._reload()

    def action_monitor(self) -> None:
        from tui.screens.monitor import MonitorScreen

        self.app.push_screen(MonitorScreen(self._selected_row()))

    def action_edit_profile(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.backend == "vllm":
            self._dispatch_vllm("edit_profile", row)
        else:
            self._dispatch_llamacpp(
                "edit-profile", row, lbackend.load_profile(row.profile_name)
            )

    def action_edit_config(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.backend == "vllm":
            self._dispatch_vllm("edit_config", row)
        else:
            self._dispatch_llamacpp(
                "edit-config", row, lbackend.load_profile(row.profile_name)
            )

    def action_delete_profile(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.running:
            self.notify(
                t("Cannot delete: container running. Stop first.",
                  "삭제 불가: 컨테이너가 실행 중입니다. 먼저 중지하세요."),
                severity="error",
            )
            return
        if row.backend == "vllm":
            self._dispatch_vllm("delete", row)
        else:
            self._dispatch_llamacpp(
                "delete-profile", row, lbackend.load_profile(row.profile_name)
            )

    def action_refresh(self) -> None:
        self._reload()

    def action_new_profile(self) -> None:
        def after(backend_name: str) -> None:
            if backend_name == "vllm":
                from tui.backends.vllm.screens.quick_setup import QuickSetupScreen

                self.app.push_screen(QuickSetupScreen(), self._after_mutation)
            elif backend_name == "llamacpp":
                from tui.backends.llamacpp.screens.quick_setup import QuickSetupScreen

                self.app.push_screen(QuickSetupScreen(), self._after_mutation)

        self.app.push_screen(BackendPickerModal(), after)

    def action_system_info(self) -> None:
        row = self._selected_row()
        if row is not None:
            screen_id = "vllm_system" if row.backend == "vllm" else "llamacpp_system"
            self.app.push_screen(screen_id)
            return

        def after(backend_name: str) -> None:
            if backend_name == "vllm":
                self.app.push_screen("vllm_system")
            elif backend_name == "llamacpp":
                self.app.push_screen("llamacpp_system")

        self.app.push_screen(BackendPickerModal(), after)

    def action_config_list(self) -> None:
        row = self._selected_row()
        if row is not None:
            screen_id = "vllm_configs" if row.backend == "vllm" else "llamacpp_configs"
            self.app.push_screen(screen_id)
            return

        def after(backend_name: str) -> None:
            if backend_name == "vllm":
                self.app.push_screen("vllm_configs")
            elif backend_name == "llamacpp":
                self.app.push_screen("llamacpp_configs")

        self.app.push_screen(BackendPickerModal(), after)

    def action_help(self) -> None:
        self.notify(
            t(
                "[b]Dashboard[/b]\n"
                "  Enter   action menu\n"
                "  u/d/l   start/stop/logs\n"
                "  p       prepare (download only, no start)\n"
                "  v/t     live monitor (in TUI / plain terminal)\n"
                "  e/c/x   edit profile/config, delete\n"
                "  C       config list (clone/rename/edit/delete)\n"
                "  m       estimate model memory\n"
                "  n s r q new/system/refresh/quit\n"
                "  U       check for a newer llmux release",
                "[b]대시보드[/b]\n"
                "  Enter   작업 메뉴\n"
                "  u/d/l   시작/중지/로그\n"
                "  p       준비 (다운로드만, 시작 안 함)\n"
                "  v/t     라이브 모니터 (TUI 안 / 일반 터미널)\n"
                "  e/c/x   프로필/config 편집, 삭제\n"
                "  C       config 목록 (복제/이름변경/편집/삭제)\n"
                "  m       모델 메모리 추정\n"
                "  n s r q 새로/시스템/새로고침/종료\n"
                "  U       llmux 새 릴리스 확인",
            ),
            title=t("Keys", "단축키"),
            timeout=10,
        )

    def _after_mutation(self, result: object = None) -> None:
        if isinstance(result, str) and result:
            self._preferred_profile_name = result
        self._reload()

    def _mem_area_visible(self) -> bool:
        return self.query_one("#mem-search-area").styles.display != "none"

    def _set_mem_area(self, visible: bool) -> None:
        area = self.query_one("#mem-search-area")
        bar = self.query_one("#mem-result-bar")
        area.styles.display = "block" if visible else "none"
        bar.styles.display = "block" if visible else "none"

    def action_mem_estimate(self) -> None:
        if self._mem_area_visible():
            self._set_mem_area(False)
            self.query_one("#profile-table", DataTable).focus()
            return
        self._set_mem_area(True)
        self.query_one("#mem-search-input", Input).focus()

    def action_hide_mem_search(self) -> None:
        if not self._mem_area_visible():
            return
        self._set_mem_area(False)
        self.query_one("#profile-table", DataTable).focus()

    @on(Input.Submitted, "#mem-search-input")
    def _on_mem_search(self, event: Input.Submitted) -> None:
        model = event.value.strip()
        if model:
            self._do_mem_estimate(model)

    @work(exclusive=True, group="mem-estimate")
    async def _do_mem_estimate(self, model_id: str) -> None:
        try:
            result_bar = self.query_one("#mem-result-bar", Static)
            result_bar.update(f"  [dim]⏳ Estimating {model_id}...[/dim]")
        except Exception:
            return

        result = await estimate_model_memory(model_id)

        match = re.search(r"~([\d.]+)GB", result)
        model_short = model_id.split("/")[-1] if "/" in model_id else model_id
        if match is None:
            result_bar.update(
                f"  📦 [bold]{model_short}[/bold]  [red]{result}[/red]"
            )
            return
        est_gb = float(match.group(1))
        if not math.isfinite(est_gb) or est_gb <= 0:
            result_bar.update(
                f"  📦 [bold]{model_short}[/bold]  [red]GPU fit UNKNOWN: {result}[/red]"
            )
            return

        if self._gpu_scan_error:
            result_bar.update(
                f"  📦 [bold]{model_short}[/bold]  {result}  "
                f"[yellow](GPU fit UNKNOWN: {self._gpu_scan_error})[/yellow]"
            )
            return

        if not self._gpus:
            result_bar.update(
                f"  📦 [bold]{model_short}[/bold]  {result}  "
                "[yellow](GPU fit UNKNOWN: no GPUs detected)[/yellow]"
            )
            return

        n_gpus = len(self._gpus)
        per_gpu_gb = est_gb / n_gpus if n_gpus > 1 else est_gb
        tp_note = (
            f"  [dim]TP={n_gpus}: {per_gpu_gb:.1f}GB/GPU[/dim]"
            if n_gpus > 1
            else ""
        )
        parts = []
        any_unknown = False
        for g in self._gpus:
            try:
                total_gb = common_docker.parse_gpu_reading(
                    g.memory_total, "memory total"
                ) / 1024
                if total_gb <= 0:
                    raise ValueError(
                        f"invalid GPU memory total: {g.memory_total!r}"
                    )
            except ValueError as exc:
                any_unknown = True
                parts.append(
                    f"GPU{g.index} [dim]{'░' * 12}[/dim] "
                    f"[yellow]UNKNOWN[/yellow] {per_gpu_gb:.1f}/?GB ({exc})"
                )
                continue
            ratio = per_gpu_gb / total_gb
            bar_w = 12
            if ratio > 1.0:
                bar = f"[red bold]{'✗' * bar_w}[/red bold]"
                label = (
                    f"[red bold]OVER[/red bold] "
                    f"{per_gpu_gb:.1f}/{total_gb:.0f}GB"
                )
            else:
                filled = round(ratio * bar_w)
                empty = bar_w - filled
                color = (
                    "green"
                    if ratio < 0.7
                    else ("yellow" if ratio < 0.9 else "red")
                )
                bar = (
                    f"[{color}]{'━' * filled}[/{color}]"
                    f"[dim]{'╌' * empty}[/dim]"
                )
                label = (
                    f"[{color}]{ratio * 100:.0f}%[/{color}] "
                    f"{per_gpu_gb:.1f}/{total_gb:.0f}GB"
                )
            parts.append(f"GPU{g.index} {bar} {label}")
        unknown_note = "  [yellow]FIT UNKNOWN[/yellow]" if any_unknown else ""
        gpu_line = " [dim]│[/dim] ".join(parts)
        text = (
            f"  📦 [bold]{model_short}[/bold] {result}{tp_note}{unknown_note}\n"
            f"     {gpu_line}"
        )
        result_bar.update(text)
