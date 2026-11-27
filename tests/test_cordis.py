"""Tests for the Cordis reversible plugin microkernel and admission/exit governance."""

from __future__ import annotations

from pathlib import Path

from scene_factory.cordis import (
    CordisContext,
    PluginMeta,
    apply_cordis_config,
    load_cordis_config,
)


def test_reversible_mount_hot_reload_and_unmount() -> None:
    ctx = CordisContext()
    log: list[str] = []

    def p1(scope, cfg):
        tag = cfg.get("tag", "v1")
        scope.on_prompt(lambda _state: f"prompt:{tag}")
        scope.effect(lambda: log.append(f"disposed:{tag}"))

    meta = PluginMeta(
        name="demo",
        kind="prompt_injection",
        purpose="guide prompt",
        added_for_model="test-model",
        exit_criteria="remove when model improves",
    )
    ctx.mount(p1, meta, {"tag": "v1"})
    assert ctx.collect_prompt_sections({}) == ["prompt:v1"]
    assert ctx.mounted_names() == ["demo"]

    # Hot-reload disposes previous effects cleanly before mounting new ones.
    ctx.mount(p1, meta, {"tag": "v2"})
    assert log == ["disposed:v1"]
    assert ctx.collect_prompt_sections({}) == ["prompt:v2"]

    # Unmounting completely retires the plugin with zero leftover hooks.
    assert ctx.unmount("demo") is True
    assert ctx.unmount("demo") is False
    assert log == ["disposed:v1", "disposed:v2"]
    assert ctx.collect_prompt_sections({}) == []
    assert ctx.snapshot() == []


def test_config_admission_exit_and_naked_disable_all() -> None:
    config_path = Path("config/cordis.toml")
    decls = load_cordis_config(config_path)
    assert len(decls) >= 2
    assert all("exit_criteria" in item and "added_for_model" in item for item in decls)

    ctx = CordisContext()
    statuses = apply_cordis_config(ctx, config_path)
    assert any(s["name"] == "mjcf_prompt_hints" and s["status"] == "mounted" for s in statuses)
    assert ctx.collect_prompt_sections({"has_images": True})

    # Passing disable_all=True cleanly strips every plugin for naked LLM benchmarking.
    apply_cordis_config(ctx, config_path, disable_all=True)
    assert ctx.mounted_names() == []
    assert ctx.collect_prompt_sections({"has_images": True}) == []


def test_xml_guard_advise_vs_traversal_block() -> None:
    ctx = CordisContext()
    apply_cordis_config(ctx, Path("config/cordis.toml"))

    # Malformed XML in advise mode emits guidance hint instead of welding a hard block.
    _, hints, blocked = ctx.run_before_tool("write_xml", {"xml": "<mujoco><broken>"}, {})
    assert blocked is None
    assert hints and "XML syntax error" in hints[0]

    # Path traversal in external file attributes is blocked.
    bad_xml = '<mujoco><asset><mesh name="m" file="../../etc/passwd"/></asset></mujoco>'
    _, _, blocked = ctx.run_before_tool("write_xml", {"xml": bad_xml}, {})
    assert blocked is not None and "disallowed" in blocked
