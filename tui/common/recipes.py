from __future__ import annotations

import asyncio
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import yaml

from tui.common.ssl_ctx import open_url

RECIPES_RAW_BASE = "https://raw.githubusercontent.com/vllm-project/recipes/main/models"


@dataclass
class RecipeVariant:
    name: str
    precision: str = ""
    vram_minimum_gb: float | None = None
    model_id: str = ""          # overrides the base model (e.g. an FP8 checkpoint)
    extra_args: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class RecipeFeature:
    name: str
    description: str = ""
    args: list[str] = field(default_factory=list)


@dataclass
class Recipe:
    model_id: str
    title: str = ""
    description: str = ""
    min_vllm_version: str = ""
    context_length: int | None = None
    base_args: list[str] = field(default_factory=list)
    variants: list[RecipeVariant] = field(default_factory=list)
    features: list[RecipeFeature] = field(default_factory=list)


class RecipeSchemaError(ValueError):
    pass


def recipe_url(model_id: str) -> str:
    org, _, repo = model_id.strip().strip("/").partition("/")
    return f"{RECIPES_RAW_BASE}/{org}/{repo}.yaml"


def _mapping(raw: dict, key: str, *, required: bool = False) -> dict:
    if key not in raw:
        if required:
            raise RecipeSchemaError(f"{key} is required")
        return {}
    value = raw[key]
    if not isinstance(value, dict):
        raise RecipeSchemaError(f"{key} must be a mapping")
    return value


def _text(raw: dict, key: str, path: str, *, required: bool = False) -> str:
    if key not in raw:
        if required:
            raise RecipeSchemaError(f"{path}.{key} is required")
        return ""
    value = raw[key]
    if not isinstance(value, str) or (required and not value.strip()):
        suffix = "a non-empty string" if required else "a string"
        raise RecipeSchemaError(f"{path}.{key} must be {suffix}")
    return value


def _args(raw: dict, key: str, path: str) -> list[str]:
    value = raw.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RecipeSchemaError(f"{path}.{key} must be a list of strings")
    return list(value)


def _parse(raw: dict, model_id: str) -> Recipe:
    if not isinstance(raw, dict):
        raise RecipeSchemaError("recipe root must be a mapping")
    model = _mapping(raw, "model", required=True)
    meta = _mapping(raw, "meta")
    variants_raw = _mapping(raw, "variants")
    features_raw = _mapping(raw, "features")

    variants: list[RecipeVariant] = []
    for name, v in variants_raw.items():
        if not isinstance(name, str) or not name:
            raise RecipeSchemaError("variants keys must be non-empty strings")
        if not isinstance(v, dict):
            raise RecipeSchemaError(f"variants.{name} must be a mapping")
        vram = v.get("vram_minimum_gb")
        if vram is not None and (
            isinstance(vram, bool)
            or not isinstance(vram, (int, float))
            or vram <= 0
        ):
            raise RecipeSchemaError(
                f"variants.{name}.vram_minimum_gb must be a positive number"
            )
        variants.append(
            RecipeVariant(
                name=name,
                precision=_text(v, "precision", f"variants.{name}"),
                vram_minimum_gb=float(vram) if vram is not None else None,
                model_id=_text(v, "model_id", f"variants.{name}"),
                extra_args=_args(v, "extra_args", f"variants.{name}"),
                description=_text(v, "description", f"variants.{name}"),
            )
        )

    features: list[RecipeFeature] = []
    for name, f in features_raw.items():
        if not isinstance(name, str) or not name:
            raise RecipeSchemaError("features keys must be non-empty strings")
        if not isinstance(f, dict):
            raise RecipeSchemaError(f"features.{name} must be a mapping")
        if "modes" in f:
            raise RecipeSchemaError(f"features.{name}.modes is not supported by llmux")
        features.append(
            RecipeFeature(
                name=name,
                description=_text(f, "description", f"features.{name}"),
                args=_args(f, "args", f"features.{name}"),
            )
        )

    ctx = model.get("context_length")
    if "context_length" in model and (
        isinstance(ctx, bool) or not isinstance(ctx, int) or ctx <= 0
    ):
        raise RecipeSchemaError("model.context_length must be a positive integer")
    return Recipe(
        model_id=_text(model, "model_id", "model", required=True),
        title=_text(meta, "title", "meta"),
        description=_text(meta, "description", "meta"),
        min_vllm_version=_text(model, "min_vllm_version", "model"),
        context_length=ctx,
        base_args=_args(model, "base_args", "model"),
        variants=variants,
        features=features,
    )


class RecipeUnavailable(RuntimeError):
    pass


async def fetch_recipe(model_id: str, timeout: int = 15) -> Recipe | None:
    if not model_id or "/" not in model_id:
        return None
    url = recipe_url(model_id)
    loop = asyncio.get_running_loop()

    def _do() -> Recipe | None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llmux"})
            with open_url(req, timeout=timeout) as r:
                raw = yaml.safe_load(r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise RecipeUnavailable(f"{url} → HTTP {exc.code}") from exc
        except Exception as exc:
            raise RecipeUnavailable(f"{url} → {exc}") from exc
        if not isinstance(raw, dict):
            raise RecipeUnavailable(f"{url} → malformed recipe file")
        try:
            return _parse(raw, model_id)
        except RecipeSchemaError as exc:
            raise RecipeUnavailable(f"{url} → invalid recipe schema: {exc}") from exc

    return await loop.run_in_executor(None, _do)


def args_to_config(args: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    i = 0
    n = len(args)
    while i < n:
        tok = args[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key, equals, inline = tok[2:].partition("=")
        i += 1
        values = [inline] if equals else []
        while not equals and i < n and not args[i].startswith("--"):
            values.append(args[i])
            i += 1
        parsed = []
        for raw in values:
            try:
                parsed.append(yaml.safe_load(raw) if raw else "")
            except yaml.YAMLError:
                parsed.append(raw)
        out[key] = parsed[0] if len(parsed) == 1 else parsed if parsed else True
    return out


def validate_feature_names(recipe: Recipe, feature_names: list[str]) -> None:
    if isinstance(feature_names, str):
        raise ValueError("recipe feature names must be a list of strings")
    available = {feature.name for feature in recipe.features}
    unknown = set(feature_names) - available
    if unknown:
        choices = ", ".join(sorted(available)) or "none"
        raise ValueError(
            f"unknown recipe feature(s): {', '.join(sorted(unknown))}; available: {choices}"
        )


def build_config(
    recipe: Recipe,
    variant: RecipeVariant | None,
    feature_names: list[str],
) -> tuple[str, dict[str, Any]]:
    validate_feature_names(recipe, feature_names)
    model_id = (variant.model_id if variant and variant.model_id else recipe.model_id)
    args = list(recipe.base_args)
    if variant:
        args += variant.extra_args
    enabled = {f.name for f in recipe.features if f.name in feature_names}
    for f in recipe.features:
        if f.name in enabled:
            args += f.args
    params = args_to_config(args)
    if recipe.context_length and "max-model-len" not in params:
        params["max-model-len"] = recipe.context_length
    return model_id, params
