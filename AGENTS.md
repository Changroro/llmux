# llmux — notes for AI coding agents

llmux is a terminal multiplexer for LLM servers. One Textual TUI and one
headless CLI drive both **vLLM** (HF Transformers models) and **llama.cpp**
(GGUF) via Docker Compose. Profiles live in `profiles.yaml` and are rendered
to per-profile `.env` files at runtime.

## Prefer the CLI

For any scripted or agent-driven change, **use the CLI, not the TUI.** The CLI
mirrors every TUI capability and is the only interface usable headlessly.

Start from `llmux --help`. Top-level shortcuts and sub-apps:

- Container lifecycle: `llmux up <profile>`, `prepare`, `down`, `wait`, `logs`, `ps`, `bench`, `render-env`
  (`prepare` = everything `up` does except starting the server: renders the
  runtime files, ensures the image is local, downloads the weights.)
- Profiles: `llmux profile {list,show,new,edit,rename,clone,delete,quick-setup}`
- Configs: `llmux config {list,show,new,edit,clone,rename,flags,from-recipe,delete}`
- Images: `llmux image {list,pull,remove,build-dev}`
- System: `llmux system {gpu,mem-estimate,disk,env-check}` (`gpu` / `env-check` also available top-level)
- Self-update: `llmux update [--check|--yes|--json]` — on-demand release check (the startup check runs on every interactive invocation; only a failed lookup backs off). `U` in the TUI.
- Plain monitor: `llmux top [profile]` — the btop-style system monitor in a plain terminal, no TUI (also `t` in the TUI). GPUs always; a profile argument narrows it to one running model.

Every `list` / `show` / `check` command takes `--json` for machine-readable
output, and `llmux --version` prints the version — the CLI is meant to be
driven from scripts, agents, and CI. Running `llmux` with no arguments launches
the TUI (intended for humans).

**Exit codes carry meaning.** A query that couldn't reach docker, the registry,
or huggingface.co exits non-zero instead of returning an empty result — `ps`
never shows `stopped` because the probe failed; `--log-level DEBUG` surfaces the
exception. Keep it that way: `run_command` returns `(1, "")` for a missing binary
*and* a failed call, so emptiness never proves absence (cf. `gpu_probe_failed`).

| Code | Meaning (`tui/common/exit_codes.py`) |
|---|---|
| 0 | success (`down` on a stopped profile included) |
| 1 | operation failed; `up` stops a container that crashed during startup |
| 2 | usage error (unknown profile, bad option) |
| 3 | conflict — port or GPUs taken |
| 4 | not ready — server still running but not answering `/v1/models`; follow with `llmux wait` |
| 5 | unreachable — docker / registry / network could not be queried |
| 130 | interrupted |

**Agent contract.** With `--json`, stdout holds exactly one record —
`{"ok": true, "code": 0, …}` or `{"ok": false, "code": N, "error": …}` (the
failure record is guaranteed by `tui.cli.main`, usage errors included). All
human output and every `Error`/`✗`/`Warning`/`⚠` line go to stderr; runtime
streams must keep those prefixes on diagnostic lines. Write commands get the
record via `@json_command` in `tui/cli/_runtime.py`. `up --dry-run` validates
in memory and returns the compose command as `plan`; it writes no file and
never builds, pulls, stops or starts anything. `--replace` runs that dry run
(plus a host-socket port check and the image pull/build) before stopping anything;
the start then reuses that image (`--pull never`).

## Repo layout (high-level)

- `tui/cli/` — Typer CLI (the headless entrypoint).
- `tui/screens/`, `tui/app.py` — Textual TUI.
- `tui/backends/{vllm,llamacpp}/` — backend runtimes; keep parity between them.
- `compose/{vllm,llamacpp}/` — Docker Compose stacks.
- `config/{vllm,llamacpp}/` — per-profile YAML overrides (gitignored; only `example.yaml` is tracked).
- `profiles.yaml`, `.env.common` — user-local, gitignored; copy from `profiles.example.yaml` / `.env.common.example` before first run.

## Conventions

- **Python env: `uv`** — `uv sync`, `uv run llmux …`, `uv run pytest`.
- **Commits: gitmoji + Conventional Commits** — `✨ feat(scope): subject`,
  `🐛 fix(...)`, `♻️ refactor(...)`, `📝 docs(...)`.
- **Branching: `git worktree`** — feature work happens on `feat/<thing>`
  branches in a worktree, never directly on `main`.
- **Parity rules:**
  - **TUI ⇄ CLI parity** — any feature added to one interface belongs in the other.
  - **vLLM ⇄ llama.cpp parity** — features should land on both backends unless the asymmetry is deliberate and documented.
  - **Deliberate asymmetry:** the live monitor is a visual view, not a JSON command. Its plain-terminal twin is `llmux top <profile>` (same `monitor_render` renderable as `v`); both share `tui/common/monitor_render.py`. Don't add a `--json` monitor to "restore parity".

## Per-developer notes

If you maintain personal scratch notes for your own AI agent (workflow
preferences, session handoff, in-flight TODOs), put them in `*.local.md`
files — `AGENTS.local.md`, `CLAUDE.local.md`, `HANDOFF.md` are all gitignored
by convention. Claude Code auto-loads `CLAUDE.local.md`; other AGENTS.md-aware
tools follow the same `.local.md` override pattern.
