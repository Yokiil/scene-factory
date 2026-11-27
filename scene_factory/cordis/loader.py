"""Declarative loader for Cordis plugin admission and exit (TOML or JSON)."""

from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path
from typing import Any

from .kernel import CordisContext, PluginMeta


def _resolve_factory(entrypoint: str) -> Any:
    if ":" not in entrypoint:
        raise ValueError(f"plugin entrypoint must have format 'module.path:callable': {entrypoint!r}")
    module_name, attr_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name, None)
    if not callable(factory):
        raise ValueError(f"plugin entrypoint is not callable: {entrypoint!r}")
    return factory


def load_cordis_config(path: str | Path) -> list[dict[str, Any]]:
    """Read plugin declarations from a .toml or .json file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"cordis config not found: {file_path}")
    if file_path.suffix.lower() == ".toml":
        payload = tomllib.loads(file_path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    plugins = payload.get("plugins", [])
    if not isinstance(plugins, list):
        raise ValueError("cordis config 'plugins' must be a list")
    return [dict(item) for item in plugins]


def apply_cordis_config(
    context: CordisContext,
    path: str | Path | None = None,
    *,
    disable_all: bool = False,
) -> list[dict[str, Any]]:
    """Mount enabled plugins from config and cleanly unmount disabled ones.

    Passing disable_all=True unmounts every plugin so callers can benchmark a
    pure, unconstrained LLM + tools baseline on stronger models.
    """
    if disable_all or path is None:
        context.dispose_all()
        return []

    declarations = load_cordis_config(path)
    enabled_names: set[str] = set()
    statuses: list[dict[str, Any]] = []

    for item in declarations:
        name = str(item["name"])
        enabled = bool(item.get("enabled", True))
        if not enabled:
            context.unmount(name)
            statuses.append({"name": name, "status": "retired_or_disabled"})
            continue

        factory = _resolve_factory(str(item["entrypoint"]))
        meta = PluginMeta(
            name=name,
            kind=str(item.get("kind", "prompt_injection")),
            purpose=str(item.get("purpose", "")),
            added_for_model=str(item.get("added_for_model", "general")),
            exit_criteria=str(
                item.get(
                    "exit_criteria",
                    "Retire once base model passes benchmark without this plugin.",
                )
            ),
            version=str(item.get("version", "1.0")),
        )
        context.mount(factory, meta, dict(item.get("config", {})))
        enabled_names.add(name)
        statuses.append({"name": name, "status": "mounted", "kind": meta.kind})

    for active in list(context.mounted_names()):
        if active not in enabled_names:
            context.unmount(active)

    return statuses
