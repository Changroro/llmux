from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def _help(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "tui", *args, "--help"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "LLMUX_NONINTERACTIVE": "1",
            "NO_COLOR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def test_ci_negative_matrix_covers_every_required_compose_interpolation() -> None:
    required = Counter()
    for path in sorted((REPO_ROOT / "compose").glob("*/*.yaml")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        for variable in re.findall(r"\$\{([A-Z0-9_]+):\?", path.read_text()):
            required[(f"{relative}:{variable}", variable)] += 1

    workflow = _read(".github/workflows/ci.yml")
    covered = Counter(
        re.findall(r"^\s+expect_missing\s+(\S+)\s+([A-Z0-9_]+)\s+", workflow, re.MULTILINE)
    )

    assert sum(required.values()) == 16
    assert covered == required
    assert 'echo "compose accepted missing required $label" >&2' in workflow
    assert 'echo "compose rejected missing required $label"' in workflow
    assert '[[ $output != *"$variable"* ]]' in workflow
    assert "exit 1" in workflow


def test_cli_help_and_reference_document_the_same_options() -> None:
    top_help = _help("top")
    profile_new_help = _help("profile", "new")
    profile_edit_help = _help("profile", "edit")
    mem_help = _help("system", "mem-estimate")
    cli_reference = _read("docs/reference/cli.html")

    assert "[PROFILE]" in top_help
    assert "--tensor-parallel" in profile_new_help
    assert "--tensor-parallel" in profile_edit_help
    for option in ("--max-loras", "--max-lora-rank", "--lora-modules"):
        assert option in profile_new_help
        assert option in profile_edit_help
    assert "--json" in mem_help
    assert "llmux top [PROFILE]" in cli_reference
    assert cli_reference.count("<code>--tensor-parallel</code>") >= 2
    for option in ("--max-loras", "--max-lora-rank", "--lora-modules"):
        assert cli_reference.count(f"<code>{option}</code>") >= 2
    assert "llmux system mem-estimate MODEL_ID [OPTIONS]" in cli_reference
    assert "<code>--json</code>" in cli_reference[
        cli_reference.index('id="system-mem-estimate"') :
        cli_reference.index('id="system-disk"')
    ]


def test_json_support_table_and_readmes_match_the_public_cli() -> None:
    cli_reference = _read("docs/reference/cli.html")
    json_section = cli_reference[
        cli_reference.index('id="json-output"') :
        cli_reference.index('id="exit-codes"')
    ]
    expected = {
        "container ps",
        "container benchmark",
        "container stats",
        "profile list",
        "profile show",
        "config list",
        "config show",
        "config flags",
        "config from-recipe",
        "image list",
        "system gpu",
        "system mem-estimate",
        "system disk",
        "system env-check",
        "update",
    }
    documented = set(re.findall(r"<code>([^<]+)</code></td><td>", json_section))

    json_help_commands = {
        "container ps": ("container", "ps"),
        "container benchmark": ("container", "benchmark"),
        "container stats": ("container", "stats"),
        "profile list": ("profile", "list"),
        "profile show": ("profile", "show"),
        "config list": ("config", "list"),
        "config show": ("config", "show"),
        "config flags": ("config", "flags"),
        "config from-recipe": ("config", "from-recipe"),
        "image list": ("image", "list"),
        "system gpu": ("system", "gpu"),
        "system mem-estimate": ("system", "mem-estimate"),
        "system disk": ("system", "disk"),
        "system env-check": ("system", "env-check"),
        "update": ("update",),
    }
    text_only_commands = (
        ("container", "up"),
        ("container", "prepare"),
        ("container", "down"),
        ("container", "logs"),
        ("profile", "new"),
        ("profile", "edit"),
        ("profile", "delete"),
        ("config", "new"),
        ("config", "edit"),
        ("config", "delete"),
        ("image", "pull"),
        ("image", "remove"),
        ("image", "build-dev"),
        ("top",),
    )

    assert documented == expected
    assert set(json_help_commands) == expected
    for args in json_help_commands.values():
        assert "--json" in _help(*args)
    for args in text_only_commands:
        assert "--json" not in _help(*args)
    assert "Every TUI action is also a headless `llmux` subcommand." in _read(
        "README.md"
    )
    assert "Structured output is available only on commands that document `--json`." in _read(
        "README.md"
    )
    assert "TUI의 모든 동작은 headless `llmux` 서브커맨드로도 제공됩니다." in _read(
        "README.ko.md"
    )
    assert "구조화 출력은 `--json`을 명시한 명령에서만 지원합니다." in _read(
        "README.ko.md"
    )


def test_llamacpp_profile_env_metrics_and_conflicts_are_documented() -> None:
    profiles = _read("docs/guide/profiles.html")
    cli_reference = _read("docs/reference/cli.html")
    configs = _read("docs/guide/configs.html")
    tui = _read("docs/guide/tui.html")

    assert "<td>both</td><td>Arbitrary" in profiles
    assert "<code>--set</code></td><td>both</td>" in cli_reference
    assert "<code>--unset</code></td><td>both</td>" in cli_reference
    assert "llmux always enables the metrics endpoint" in configs
    assert "cannot override llmux-managed server options" in configs
    assert "Port conflicts and failed port probes block startup" in tui
    assert "offers <strong>Stop and start</strong>" in tui


def test_manual_secret_files_use_owner_only_permissions() -> None:
    template = _read(".env.common.example")
    env_reference = _read("docs/reference/env-common.html")
    installation = _read("docs/getting-started/installation.html")
    profiles = _read("docs/guide/profiles.html")

    assert "install -m 600 .env.common.example .env.common" in template
    assert "install -m 600 .env.common.example .env.common" in env_reference
    assert "install -m 600 .env.common.example .env.common" in installation
    assert "install -m 600 .env.common.example .env.common" in _read(
        "docs/troubleshooting.html"
    )
    assert "install -m 600 profiles.example.yaml profiles.yaml" in profiles
    assert "install -m 600 profiles.example.yaml profiles.yaml" in installation
    assert "owner-only" in env_reference
    assert "owner-only" in profiles
    assert "group/world-accessible" in _read("docs/reference/cli.html")
    cli_reference = _read("docs/reference/cli.html")
    assert "Other environment values remain visible for automation compatibility" in cli_reference
    assert "<code>HF_TOKEN</code>, <code>HF_ENDPOINT</code>" in profiles
    assert "<code>DOCKER_</code> or <code>COMPOSE_</code>" in profiles
    for key in ("PATH", "HTTP_PROXY", "SSL_CERT_FILE", "LD_PRELOAD"):
        assert f"<code>{key}</code>" in profiles
    assert "are not promoted into the host Compose process" in profiles
    assert "Existing environment values are shown as <code>&lt;redacted&gt;</code>" in _read(
        "docs/guide/tui.html"
    )


def test_failure_and_input_security_contracts_are_documented() -> None:
    cli_reference = _read("docs/reference/cli.html")
    tui = _read("docs/guide/tui.html")
    architecture = _read("docs/reference/architecture.html")
    dev_build = _read("docs/guide/dev-build.html")

    assert "aggregate <code>any_over</code> is also <code>null</code>" in cli_reference
    assert "<code>hf_cache_exists: false</code>" in cli_reference
    assert "<code>df_target</code>" in cli_reference
    assert "HTTP polling failure displays <code>error</code>" in tui
    assert "the command exits 2 because the target is unknown" in tui
    assert "rechecks the approved targets" in architecture
    assert "HTTP(S) URL containing userinfo, a query, or a fragment is rejected" in dev_build
    assert "Docker image inputs reject URL schemes" in cli_reference
    assert "pagination safety cap is reached" in tui
    assert "never to an off-origin pagination or redirect target" in tui
    llamacpp = _read("docs/backends/llamacpp.html")
    assert "--no-webui --metrics" in llamacpp
    assert "cannot override the managed command" in llamacpp
