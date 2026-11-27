"""Tests for AssetCatalog, multi-part convex meshes, FragmentCache, and MJCF assembly."""

from __future__ import annotations

import pytest

from scene_factory.core import (
    AssetCatalog,
    FragmentCache,
    create_base_xml,
    inspect_xml_summary,
    merge_fragment,
)
from scene_factory.core.mjcf import patch_xml


def test_asset_catalog_bilingual_search_and_detailed_meshes() -> None:
    catalog = AssetCatalog()
    assert len(catalog) >= 40

    zh_mugs = catalog.search("马克杯")
    assert any("mug" in item["category"] for item in zh_mugs)

    detailed = catalog.search("table", detailed_only=True)
    assert any(item["asset_id"] == "kd_table_01" and item["has_detailed_mesh"] for item in detailed)

    # Both kitchen_dining_pack and dessert assets load multi-part geometry cleanly.
    for aid in ("kd_table_01", "kd_mug_01", "dessert_tiered_stand_01"):
        parts = catalog.load_mesh_parts(aid)
        assert parts and len(parts) >= 1


def test_fragment_cache_and_merge_replace() -> None:
    catalog = AssetCatalog()
    cache = FragmentCache(catalog)

    xml = create_base_xml("demo_kitchen")
    a1, b1, d1, hit1 = cache.get_or_build(
        "kd_table_01", name="table_1", pos=(0.0, 0.0, 0.375), dynamic=False
    )
    assert d1 is True and hit1 is False
    xml = merge_fragment(xml, a1, b1, enable_multiccd=d1)

    # Identical call hits the content-addressed fragment cache.
    a2, b2, d2, hit2 = cache.get_or_build(
        "kd_table_01", name="table_1", pos=(0.0, 0.0, 0.375), dynamic=False
    )
    assert hit2 is True and d2 is True
    xml = merge_fragment(xml, a2, b2, enable_multiccd=d2, replace_existing=True)

    summary = inspect_xml_summary(xml)
    assert summary["body_count"] == 1
    assert summary["mesh_count"] == len(a1)
    assert 'multiccd="enable"' in xml
    assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 1


def test_patch_xml_exact_and_ambiguity_checks() -> None:
    xml = create_base_xml("patch_test")
    updated = patch_xml(xml, 'model="patch_test"', 'model="renamed_scene"')
    assert 'model="renamed_scene"' in updated

    with pytest.raises(ValueError, match="not found"):
        patch_xml(xml, "nonexistent_token", "x")
