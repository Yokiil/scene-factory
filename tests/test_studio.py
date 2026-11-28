"""Tests for SceneFactory v2 Precision 3D Studio backend and HTTP server."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

from scene_factory.studio.server import StudioBackend, create_studio_server, extract_render_geoms


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict) -> dict:
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_studio_backend_presets_and_cordis_toggle(tmp_path: Path) -> None:
    backend = StudioBackend(work_dir=tmp_path)
    status = backend.status()
    assert status["asset_count"] >= 48
    assert status["detailed_asset_count"] >= 10
    assert len(status["presets"]) == 4

    # Level 2 nested collision preset must compile cleanly in MuJoCo and detect analytical collision.
    col_scene = backend.build_preset("level2_nested_collision")
    assert col_scene["verification"]["compiled"] is True
    assert col_scene["analysis"]["assembly_valid"] is False
    assert len(col_scene["analysis"]["colliding_pairs"]) >= 1
    assert len(col_scene["scene_3d"]["bodies"]) >= 4

    # Three-tier dessert display preset must compile cleanly with zero collisions and zero floating.
    dessert_scene = backend.build_preset("dessert_display")
    assert dessert_scene["verification"]["compiled"] is True
    assert dessert_scene["analysis"]["assembly_valid"] is True
    assert len(dessert_scene["analysis"]["colliding_pairs"]) == 0
    assert len(dessert_scene["analysis"]["floating_bodies"]) == 0

    # Toggle Cordis plugin cleanly without breaking backend state.
    updated = backend.toggle_cordis_plugin("xml_spatial_hint", True)
    by_name = {p["name"]: p["enabled"] for p in updated}
    assert by_name["xml_spatial_hint"] is True
    reverted = backend.toggle_cordis_plugin("xml_spatial_hint", False)
    assert {p["name"]: p["enabled"] for p in reverted}["xml_spatial_hint"] is False


def test_studio_http_server_routes(tmp_path: Path) -> None:
    backend = StudioBackend(work_dir=tmp_path)
    server = create_studio_server("127.0.0.1", 0, backend=backend)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        with urlopen(f"{base}/", timeout=10) as resp:
            html = resp.read().decode("utf-8")
            assert "SceneFactory Studio" in html
            assert "/styles.css" in html

        with urlopen(f"{base}/vendor/three.module.js", timeout=10) as resp:
            assert resp.status == 200

        status = _get_json(f"{base}/api/status")
        assert status["version"] == "0.2.0"

        demo = _get_json(f"{base}/api/preset/kitchen_high_fidelity")
        assert demo["ok"] is True
        assert demo["verification"]["compiled"] is True
        assert demo["verification"]["score"] >= 0.8
        assert demo["trace"] is not None
        assert len(demo["variants"]) >= 1
        assert len(demo["scene_3d"]["meshes"]) >= 5

        geoms = extract_render_geoms(demo["xml"])
        assert len(geoms["bodies"]) == 5

        toggled = _post_json(f"{base}/api/cordis/toggle", {"name": "xml_spatial_hint", "enabled": True})
        assert toggled["ok"] is True
        assert any(p["name"] == "xml_spatial_hint" and p["enabled"] for p in toggled["cordis_plugins"])
    finally:
        server.shutdown()
        server.server_close()
