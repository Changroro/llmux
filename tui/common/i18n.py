"""Tiny two-language switch for user-facing strings (TUI notifies, labels, CLI).

Wrap any string a user reads in ``t(en, ko)``. The active language resolves from
``LLMUX_LANG`` (``en`` / ``ko``); when unset it follows the system locale
(``LC_ALL`` / ``LC_MESSAGES`` / ``LANG``), then falls back to English.

There is no message catalog on purpose — the two variants sit right at the call
site, so they can't drift out of sync with the surrounding code the way a
separate ``.po`` file does. Interpolate the natural way and let ``t`` pick:

    self.notify(t(f"Stopping {name}…", f"{name} 중지 중…"))
"""

from __future__ import annotations

import os

_VALID = ("en", "ko")


def lang() -> str:
    """Resolve the active UI language: ``LLMUX_LANG`` > locale > ``en``.

    Read fresh each call (a handful of env lookups) so a test — or a user who
    exports the var and relaunches — always sees the current value.
    """
    explicit = os.environ.get("LLMUX_LANG", "").strip().lower()
    if explicit in _VALID:
        return explicit
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        locale = os.environ.get(var, "").strip()
        if locale:
            return "ko" if locale[:2].lower() == "ko" else "en"
    return "en"


def t(en: str, ko: str) -> str:
    """Return the Korean variant in ``ko`` mode, English otherwise."""
    return ko if lang() == "ko" else en
