"""Integration tests for real MuJoCo verification, offscreen PNG rendering, and XML randomization."""

from __future__ import annotations

from pathlib import Path

import pytest

from scene_factory.core import (
    AssetCatalog,
    FragmentCache,
    create_base_xml,
    merge_fragment,
    randomize_xml,
    verify_mujoco_xml,
)


def _table_and_mug_xml() -> str:
    catalog = AssetCatalog()
    cache = FragmentCache(catalog)
    xml = create_base_xml("table_mug")

    table_entry = catalog.get("kd_table_01")
    table_h = table_entry.bbox_m[2]
    a_t, b_t, d_t, _ = cache.get_or_build(
        "kd_table_01", name="dining_table", pos=(0.0, 0.0, table_h / 2.0), dynamic=False
    )
    xml = merge_fragment(xml, a_t, b_t, enable_multiccd=d_t)

    mug_entry = catalog.get("kd_mug_01")
    mug_h = mug_entry.bbox_m[2]
    a_m, b_m, d_m, _ = cache.get_or_build(
        "kd_mug_01", name="mug_1", pos=(0.0, 0.0, table_h + mug_h / 2.0 + 0.002), dynamic=True
    )
    xml = merge_fragment(xml, a_m, b_m, enable_multiccd=d_m)
    return xml


def test_verify_rejects_malformed_xml_without_mujoco() -> None:
    report = verify_mujoco_xml("<mujoco><unclosed>", render=False)
    assert report.compiled is False
    assert report.error and "XML parse error" in report.error
    assert report.score == 0.0


def test_verify_real_mujoco_settle_and_png_render(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    xml = _table_and_mug_xml()
    render_dir = tmp_path / "renders"
    report = verify_mujoco_xml(xml, settle_steps=120, render=True, render_dir=render_dir)
    assert report.compiled is True
    assert report.finite_state is True
    assert report.settled_stable is True
    assert report.fallen_bodies == []
    assert report.score == 1.0

    # Verify real PNG bytes were written when offscreen GL is available.
    if report.render_error is None:
        assert len(report.renders) >= 2
        for item in report.renders:
            p = Path(item["path"])
            assert p.is_file()
            assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_randomize_xml_uses_single_compile_and_is_deterministic() -> None:
    pytest.importorskip("mujoco")
    xml = _table_and_mug_xml()
    v1, stats1 = randomize_xml(
        xml,
        count=4,
        seed=123,
        xy_jitter_m=0.12,
        yaw_jitter_deg=45.0,
        mass_scale_range=(0.8, 1.2),
        friction_scale_range=(0.7, 1.3),
    )
    v2, _ = randomize_xml(
        xml,
        count=4,
        seed=123,
        xy_jitter_m=0.12,
        yaw_jitter_deg=45.0,
        mass_scale_range=(0.8, 1.2),
        friction_scale_range=(0.7, 1.3),
    )
    assert len(v1) == 4
    assert stats1["native_compiles"] == 1
    assert stats1["patched_checks"] >= 4
    assert stats1["randomized_bodies"] == ["mug_1"]
    assert [v.xml_sha256 for v in v1] == [v.xml_sha256 for v in v2]
    assert len({v.xml_sha256 for v in v1}) == 4
    assert all(v.max_penetration_m <= 0.01 for v in v1)
