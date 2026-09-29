from __future__ import annotations

from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from tui.backends.llamacpp.backend import (
    COMMON_ENV,
    GpuInfo,
    ROOT,
    _get_hf_cache_dir,
    _get_model_dir,
    get_disk_usage,
    get_docker_images,
    get_gpu_info,
    list_cached_gguf,
    list_profile_names,
    load_profile,
)
from tui.backends.llamacpp.backend_runtime import (
    LLAMACPP_DEV_SPEC,
    get_dev_build_defaults,
)
from tui.common import profile_store, system_operations
from tui.backends.llamacpp.backend import LLAMACPP_OFFICIAL_REPO
from tui.common.dev_build import image_tag_error, list_local_dev_images
from tui.common.docker import (
    GpuReadingLevel,
    container_snapshots,
    gpu_temperature_level,
    gpu_utilization_level,
    parse_gpu_reading,
)
from tui.common.env import parse_env_file
from tui.common.i18n import t
from tui.common.widgets import ConfirmModal, DevBuildPromptModal, TextPromptModal


class SystemScreen(Screen):
    BINDINGS = [
        Binding("escape,backspace,s", "go_back", t("Back", "뒤로"), show=True),
        Binding("q", "go_back", "Back", show=False),
        Binding("r", "refresh_all", t("Refresh", "새로고침"), show=True),
    ]

    DEFAULT_CSS = """
    SystemScreen { layout: vertical; }
    #refresh-bar {
        height: auto;
        dock: bottom;
        padding: 0 2;
        align: right middle;
    }
    .section-title {
        text-style: bold;
        color: $primary;
        margin: 1 0 0 1;
    }
    #images-scroll {
        height: 1fr;
    }
    #images-scroll DataTable {
        max-height: 12;
    }
    #image-action-log {
        height: 8;
    }
    #environment-actions {
        height: auto;
        padding: 1 2;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._gpu_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("GPU", id="gpu-tab"):
                yield DataTable(id="gpu-table")
            with TabPane(t("Docker Images", "Docker 이미지"), id="images-tab"):
                with VerticalScroll(id="images-scroll"):
                    yield Static(
                        t(f"Official Images ({LLAMACPP_OFFICIAL_REPO})", f"공식 이미지 ({LLAMACPP_OFFICIAL_REPO})"),
                        classes="section-title",
                    )
                    yield DataTable(id="llama-images")
                    yield Static(t("Dev Images (llamacpp-dev)", "개발 이미지 (llamacpp-dev)"), classes="section-title")
                    yield DataTable(id="dev-images")
                yield Horizontal(
                    Button(t("Refresh Images", "이미지 새로고침"), id="btn-refresh-images", variant="primary"),
                    Button(t("Pull", "받기"), id="btn-pull-image"),
                    Button(t("Remove", "삭제"), id="btn-remove-image"),
                    Button(t("Build Dev", "개발 이미지 빌드"), id="btn-build-image"),
                    id="refresh-bar",
                )
                yield RichLog(id="image-action-log", highlight=True)
            with TabPane(t("Containers", "컨테이너"), id="containers-tab"):
                yield RichLog(id="container-info", highlight=True)
            with TabPane(t("Disk / Model Dir", "디스크 / 모델 폴더"), id="disk-tab"):
                yield RichLog(id="disk-info", highlight=True)
            with TabPane(t("Environment", "환경"), id="environment-tab"):
                yield Horizontal(
                    Button(t("Validate", "검증"), id="btn-validate-env", variant="primary"),
                    Button(t("Render Runtime Envs", "런타임 환경 렌더링"), id="btn-render-envs"),
                    id="environment-actions",
                )
                yield RichLog(id="environment-log", highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        gpu_table = self.query_one("#gpu-table", DataTable)
        gpu_table.add_columns("GPU", "Name", "Mem Used", "Mem Total", "Util", "Temp")
        images_table = self.query_one("#llama-images", DataTable)
        images_table.add_columns("Tag", "Size", "Created")
        dev_images_table = self.query_one("#dev-images", DataTable)
        dev_images_table.add_columns("Tag", "Size", "Created")

        self._refresh_gpu()
        self._refresh_images()
        self._refresh_containers()
        self._refresh_disk()
        self._gpu_timer = self.set_interval(3, self._refresh_gpu)

    def on_screen_suspend(self) -> None:
        if self._gpu_timer is not None:
            self._gpu_timer.pause()

    def on_screen_resume(self) -> None:
        if self._gpu_timer is not None:
            self._gpu_timer.resume()
        self._refresh_gpu()


    @work(exclusive=True, group="sys-gpu")
    async def _refresh_gpu(self) -> None:
        try:
            gpus = await get_gpu_info()
        except RuntimeError as exc:
            self._update_gpu_error(exc)
            return
        self._update_gpu_table(gpus)

    def _update_gpu_error(self, exc: RuntimeError) -> None:
        table = self.query_one("#gpu-table", DataTable)
        table.clear()
        table.add_row("--", t("GPU query failed", "GPU 조회 실패"), "--", "--", "--", "--")
        self.notify(
            t(f"GPU query failed: {exc}", f"GPU 조회 실패: {exc}"),
            severity="error",
            timeout=8,
        )

    def _update_gpu_table(self, gpus: list[GpuInfo]) -> None:
        table = self.query_one("#gpu-table", DataTable)
        table.clear()
        if not gpus:
            table.add_row("--", "감지된 GPU 없음", "--", "--", "--", "--")
            return
        for g in gpus:
            try:
                util_level = gpu_utilization_level(g.utilization)
                util_color = {
                    GpuReadingLevel.NORMAL: "green",
                    GpuReadingLevel.WARNING: "yellow",
                    GpuReadingLevel.CRITICAL: "red",
                }[util_level]
                util = f"[{util_color}]{g.utilization}%[/]"
            except ValueError:
                util = "--"
            try:
                temp_level = gpu_temperature_level(g.temperature)
                temp_color = {
                    GpuReadingLevel.NORMAL: "green",
                    GpuReadingLevel.WARNING: "yellow",
                    GpuReadingLevel.CRITICAL: "red",
                }[temp_level]
                temp = f"[{temp_color}]{g.temperature}°C[/]"
            except ValueError:
                temp = "--"
            try:
                used_gb = parse_gpu_reading(g.memory_used, "memory used") / 1024
                mem_used = f"{used_gb:.1f} GB"
            except ValueError:
                mem_used = "--"
            try:
                total_gb = parse_gpu_reading(g.memory_total, "memory total") / 1024
                if total_gb <= 0:
                    raise ValueError
                mem_total = f"{total_gb:.1f} GB"
            except ValueError:
                mem_total = "--"
            table.add_row(g.index, g.name, mem_used, mem_total, util, temp)


    @work(exclusive=True, group="sys-images")
    async def _refresh_images(self) -> None:
        for table_id, fetch in (
            ("#llama-images",
             lambda: get_docker_images(LLAMACPP_OFFICIAL_REPO)),
            ("#dev-images",
             lambda: list_local_dev_images(LLAMACPP_DEV_SPEC)),
        ):
            try:
                self._update_image_table(table_id, await fetch())
            except Exception as exc:
                self._image_table_error(table_id, exc)

    def _update_image_table(self, table_id: str, images) -> None:
        table = self.query_one(table_id, DataTable)
        table.clear()
        if not images:
            table.add_row("(없음)", "--", "--")
            return
        for img in images:
            table.add_row(img.tag, img.size, img.created)

    def _image_table_error(self, table_id: str, exc: Exception) -> None:
        table = self.query_one(table_id, DataTable)
        table.clear()
        table.add_row(t("[red](query failed)[/]", "[red](조회 실패)[/]"), "--", "--")
        self.notify(t(f"Image list failed: {exc}", f"이미지 조회 실패: {exc}"),
                    severity="error", timeout=8)


    @work(exclusive=True, group="sys-containers")
    async def _refresh_containers(self) -> None:
        log = self.query_one("#container-info", RichLog)
        log.clear()
        try:
            known = {load_profile(n).container_name for n in list_profile_names()}
            for stored in profile_store.list_profiles("vllm"):
                known.add(stored.container_name or stored.name)
            snapshots = await container_snapshots(include_stopped=True)
        except (OSError, RuntimeError, ValueError) as exc:
            log.write(t("[red]docker ps failed[/]", "[red]docker ps 실패[/]"))
            log.write(str(exc))
            self.notify(
                t(
                    f"Container inventory failed: {exc}",
                    f"컨테이너 목록 조회 실패: {exc}",
                ),
                severity="error",
                timeout=8,
            )
            return
        selected = [snapshots[name] for name in sorted(known & snapshots.keys())]
        if not selected:
            log.write(t("[dim]No profile containers[/]", "[dim]프로필 컨테이너 없음[/]"))
            return
        log.write("NAME\tSTATE\tHEALTH\tSTATUS")
        for snapshot in selected:
            health = snapshot.health.value if snapshot.health.value != "none" else "--"
            log.write(
                f"{snapshot.name}\t{snapshot.raw_state}\t{health}\t{snapshot.raw_status}"
            )


    @work(exclusive=True, group="sys-disk")
    async def _refresh_disk(self) -> None:
        log = self.query_one("#disk-info", RichLog)
        log.clear()
        try:
            model_dir = _get_model_dir()
            model_dir_exists = model_dir.exists()
            files: list[tuple[Path, int]] = []
            if model_dir_exists:
                for path in model_dir.glob("*.gguf"):
                    files.append((path, path.stat().st_size))
                files.sort(key=lambda item: item[1], reverse=True)
            hf_cache_dir = _get_hf_cache_dir()
            cached = list_cached_gguf()
            from tui.cli.system import _nearest_existing_path

            probe_path = _nearest_existing_path(model_dir)
            used, avail, pct = await get_disk_usage(str(probe_path))
        except Exception as exc:
            log.write(f"[red]{exc}[/]")
            self.notify(
                t(f"Disk inventory failed: {exc}", f"디스크 조회 실패: {exc}"),
                severity="error",
                timeout=8,
            )
            return

        log.write(f"[b]Project root:[/b] {ROOT}")
        log.write(f"[b]Model dir:[/b] {model_dir}")
        log.write("")

        if model_dir_exists:
            log.write(t(f"[b]GGUF files ({len(files)})[/b]", f"[b]GGUF 파일 ({len(files)} 개)[/b]"))
            total = 0
            for path, sz in files:
                total += sz
                log.write(f"  {path.name}  [dim]{sz / 1024**3:.1f} GB[/dim]")
            log.write(t(f"  [dim]Total: {total / 1024**3:.1f} GB[/dim]", f"  [dim]합계: {total / 1024**3:.1f} GB[/dim]"))
            log.write("")
        else:
            log.write(t(f"[yellow]Model directory does not exist: {model_dir}[/]", f"[yellow]모델 디렉토리 존재하지 않음: {model_dir}[/]"))
            log.write("")

        log.write(t(f"[b]HF cache GGUF ({len(cached)})[/b]  [dim]{hf_cache_dir}[/dim]", f"[b]HF cache GGUF ({len(cached)} 개)[/b]  [dim]{hf_cache_dir}[/dim]"))
        if cached:
            for c in cached:
                log.write(
                    f"  {c['repo']}/{c['name']}  [dim]{c['size_gb']:.1f} GB[/dim]"
                )
            total_gb = sum(c["size_bytes"] for c in cached) / 1024**3
            log.write(t(f"  [dim]Total: {total_gb:.1f} GB[/dim]", f"  [dim]합계: {total_gb:.1f} GB[/dim]"))
        else:
            log.write(t("  [dim](no GGUF downloaded in HF cache)[/dim]", "  [dim](HF 캐시에 받아둔 GGUF 없음)[/dim]"))
        log.write("")

        log.write(t("[b]Disk usage[/b]", "[b]디스크 사용량[/b]"))
        log.write(t(f"  Used: {used}  Free: {avail}  ({pct})", f"  사용: {used}  남음: {avail}  ({pct})"))


    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_refresh_all(self) -> None:
        self._refresh_gpu()
        self._refresh_images()
        self._refresh_containers()
        self._refresh_disk()
        self.notify(t("Refreshing", "새로고침"))

    def _prompt_pull_image(self) -> None:
        configured = parse_env_file(COMMON_ENV).get("LLAMACPP_IMAGE", "").strip()
        default = configured or f"{LLAMACPP_OFFICIAL_REPO}:server-cuda"

        def after(image_ref: str | None) -> None:
            if image_ref:
                self._pull_image(image_ref)

        self.app.push_screen(
            TextPromptModal(
                t("Image reference to pull", "받을 이미지 참조"),
                default=default,
                placeholder=f"{LLAMACPP_OFFICIAL_REPO}:server-cuda",
            ),
            after,
        )

    def _prompt_remove_image(self) -> None:
        def after_ref(image_ref: str | None) -> None:
            if not image_ref:
                return

            def after_confirm(confirmed: bool) -> None:
                if confirmed:
                    self._remove_image(image_ref)

            self.app.push_screen(
                ConfirmModal(
                    t(
                        f"Remove local image [b]{image_ref}[/b]?",
                        f"로컬 이미지 [b]{image_ref}[/b] 을 삭제할까요?",
                    ),
                    confirm_label=t("Remove", "삭제"),
                ),
                after_confirm,
            )

        self.app.push_screen(
            TextPromptModal(
                t("Image reference or ID to remove", "삭제할 이미지 참조 또는 ID"),
                placeholder=f"{LLAMACPP_OFFICIAL_REPO}:server-cuda",
            ),
            after_ref,
        )

    def _prompt_build_image(self) -> None:
        repo_url, branch = get_dev_build_defaults()

        def after(options: dict[str, object] | None) -> None:
            if options:
                self._build_image(options)

        self.app.push_screen(
            DevBuildPromptModal("llamacpp", repo_url, branch),
            after,
        )

    @work(exclusive=True, group="sys-image-action")
    async def _pull_image(self, image_ref: str) -> None:
        log = self.query_one("#image-action-log", RichLog)
        log.clear()
        error = image_tag_error(image_ref)
        if error:
            log.write(f"[red]{error}[/]")
            self.notify(error, severity="error", timeout=8)
            return
        log.write(t(f"Pulling {image_ref}", f"{image_ref} 받는 중"))
        try:
            rc, lines = await system_operations.pull_image(image_ref)
        except (OSError, RuntimeError, ValueError) as exc:
            log.write(f"[red]{exc}[/]")
            self.notify(str(exc), severity="error", timeout=8)
            return
        for line in lines:
            log.write(line)
        if rc != 0:
            self.notify(t("Image pull failed", "이미지 받기 실패"), severity="error", timeout=8)
            return
        self._refresh_images()
        self.notify(t("Image pulled", "이미지를 받았습니다"))

    @work(exclusive=True, group="sys-image-action")
    async def _remove_image(self, image_ref: str) -> None:
        log = self.query_one("#image-action-log", RichLog)
        log.clear()
        rc, output = await system_operations.remove_image(image_ref)
        if output.strip():
            log.write(output)
        if rc != 0:
            self.notify(t("Image removal failed", "이미지 삭제 실패"), severity="error", timeout=8)
            return
        self._refresh_images()
        self.notify(t("Image removed", "이미지를 삭제했습니다"))

    @work(exclusive=True, group="sys-image-action")
    async def _build_image(self, options: dict[str, object]) -> None:
        log = self.query_one("#image-action-log", RichLog)
        log.clear()
        try:
            rc, lines = await system_operations.build_dev_image(
                "llamacpp",
                repo_url=str(options["repo_url"]),
                branch=str(options["branch"]),
                custom_tag=str(options["custom_tag"]),
                cuda_arch=str(options["cuda_arch"]),
                multi_arch=bool(options["multi_arch"]),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            log.write(f"[red]{exc}[/]")
            self.notify(str(exc), severity="error", timeout=8)
            return
        for line in lines:
            log.write(line)
        if rc != 0:
            self.notify(t("Development image build failed", "개발 이미지 빌드 실패"), severity="error", timeout=8)
            return
        self._refresh_images()
        self.notify(t("Development image built", "개발 이미지를 빌드했습니다"))

    def _validate_environment(self) -> None:
        log = self.query_one("#environment-log", RichLog)
        log.clear()
        ok, messages = system_operations.environment_status(COMMON_ENV)
        for message in messages:
            log.write(message)
        if ok:
            log.write(t("[green]Status: OK[/]", "[green]상태: 정상[/]"))
            self.notify(t("Environment is valid", "환경 설정이 유효합니다"))
        else:
            self.notify(t("Environment validation failed", "환경 설정 검증 실패"), severity="error", timeout=8)

    def _render_environment(self) -> None:
        log = self.query_one("#environment-log", RichLog)
        log.clear()
        try:
            rendered, failures = system_operations.render_backend_envs("llamacpp")
        except (OSError, RuntimeError, ValueError) as exc:
            log.write(f"[red]{exc}[/]")
            self.notify(str(exc), severity="error", timeout=8)
            return
        for path in rendered:
            log.write(t(f"Rendered {path}", f"렌더링 완료: {path}"))
        for failure in failures:
            log.write(f"[red]{failure}[/]")
        if failures:
            self.notify(t("Some runtime envs failed", "일부 런타임 환경 렌더링 실패"), severity="error", timeout=8)
        else:
            self.notify(t("Runtime envs rendered", "런타임 환경을 렌더링했습니다"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-refresh-images":
            self._refresh_images()
            self.notify(t("Refreshing image list", "이미지 목록 새로고침"))
        elif event.button.id == "btn-pull-image":
            self._prompt_pull_image()
        elif event.button.id == "btn-remove-image":
            self._prompt_remove_image()
        elif event.button.id == "btn-build-image":
            self._prompt_build_image()
        elif event.button.id == "btn-validate-env":
            self._validate_environment()
        elif event.button.id == "btn-render-envs":
            self._render_environment()
