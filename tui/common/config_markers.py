from __future__ import annotations

import re
from typing import Any

import yaml

_MARKER_PREFIX = "# llmux:disabled "
_MARKER_RE = re.compile(r"^#\s*llmux:disabled\s+([^:]+):\s?(.*)$")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate mapping key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml_mapping(text: str, source: object) -> dict[str, Any]:
    try:
        parsed = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"{source}: invalid YAML: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(
            f"{source} must be a mapping; value is not a mapping "
            f"({type(parsed).__name__})"
        )
    return dict(parsed)


def _inline(value: Any) -> str:
    dumped = yaml.safe_dump(
        value,
        default_flow_style=True,
        allow_unicode=True,
        sort_keys=False,
        width=10**9,
    )
    lines = [ln for ln in dumped.strip().split("\n") if ln.strip() != "..."]
    return " ".join(lines)


def render_disabled_markers(disabled: dict[str, Any]) -> str:
    lines = [f"{_MARKER_PREFIX}{key}: {_inline(value)}" for key, value in disabled.items()]
    return ("\n".join(lines) + "\n") if lines else ""


def dump_active_config(existing_text: str | None, data: dict[str, Any]) -> str:
    active = strip_disabled_markers(existing_text or "")

    def plain() -> str:
        return yaml.dump(
            data, default_flow_style=False, allow_unicode=True, sort_keys=False
        )
    if "#" not in active:
        return plain()

    try:
        from io import StringIO

        from ruamel.yaml import YAML
    except ImportError as exc:
        raise RuntimeError(
            "ruamel.yaml is required to preserve config comments but is not "
            "installed; a plain dump would silently drop disabled-param markers. "
            "Reinstall dependencies (uv sync)."
        ) from exc

    ry = YAML()
    ry.preserve_quotes = True
    ry.width = 10**9
    try:
        existing = ry.load(active)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"the existing config could not be parsed ({exc}); saving would "
            "drop every comment in it. Fix or delete the file first."
        ) from exc
    if not isinstance(existing, dict):
        raise RuntimeError(
            "the existing config is not a mapping; saving would drop every "
            "comment in it. Fix or delete the file first."
        )

    def merge(target, values):
        if isinstance(target, dict) and isinstance(values, dict):
            for key in [k for k in target if k not in values]:
                _delete_keeping_following_comment(target, key)
            for key, value in values.items():
                target[key] = merge(target.get(key), value)
            return target
        if isinstance(target, list) and isinstance(values, list):
            del target[len(values):]
            for index, value in enumerate(values):
                if index < len(target):
                    target[index] = merge(target[index], value)
                else:
                    target.append(value)
            return target
        return values

    merge(existing, data)

    buf = StringIO()
    ry.dump(existing, buf)
    return buf.getvalue()


def _delete_keeping_following_comment(mapping: Any, key: Any) -> None:
    """Delete `key` from a ruamel mapping without losing the comment lines below it.

    ruamel stores the lines between a value and the next key as that key's
    trailing comment, so a plain delete would take them along.
    """
    from ruamel.yaml.error import CommentMark
    from ruamel.yaml.tokens import CommentToken

    keys = list(mapping)
    index = keys.index(key)
    slot = mapping.ca.items.pop(key, None)
    del mapping[key]
    post = slot[2] if slot else None
    # The text before the first newline is the deleted key's own inline comment.
    tail = post.value.split("\n", 1)[1] if post is not None and "\n" in post.value else ""
    if not tail.strip():
        return
    if index > 0:
        prev = mapping.ca.items.setdefault(keys[index - 1], [None, None, None, None])
        if prev[2] is None:
            prev[2] = CommentToken("\n" + tail, CommentMark(0), None)
        else:
            prev[2].value += tail
        return
    if mapping.ca.comment is None:
        mapping.ca.comment = [None, []]
    if mapping.ca.comment[1] is None:
        mapping.ca.comment[1] = []
    for line in tail.splitlines():
        mapping.ca.comment[1].append(CommentToken(line + "\n", CommentMark(0), None))


def parse_disabled_markers(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in text.splitlines():
        m = _MARKER_RE.match(line.strip())
        if not m:
            continue
        key = m.group(1).strip()
        raw = m.group(2)
        try:
            out[key] = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"disabled marker {key!r} contains invalid YAML: {exc}"
            ) from exc
    return out


def strip_disabled_markers(text: str) -> str:
    kept = [
        line for line in text.splitlines()
        if not _MARKER_RE.match(line.strip())
    ]
    result = "\n".join(kept)
    return result + "\n" if result and not result.endswith("\n") else result
