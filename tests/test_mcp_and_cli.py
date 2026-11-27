"""Tests for the zero-dependency MCP stdio plugin server and v2 CLI."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from scene_factory.cli import main as cli_main
from scene_factory.core import create_base_xml
from scene_factory.plugin_api import run_stdio_server


def test_mcp_stdio_plugin_roundtrip() -> None:
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search_assets", "arguments": {"query": "mug", "limit": 5}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "compose_assets_xml",
                "arguments": {
                    "model_name": "mcp_demo",
                    "objects": [
                        {"asset_id": "kd_mug_01", "name": "mug_1", "pos": [0.0, 0.0, 0.055], "dynamic": True}
                    ],
                },
            },
        },
    ]
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    stdout = io.StringIO()
    assert run_stdio_server(stdin=stdin, stdout=stdout) == 0

    replies = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert len(replies) == 4
    assert replies[0]["result"]["serverInfo"]["name"] == "scene-factory"
    tool_names = {t["name"] for t in replies[1]["result"]["tools"]}
    assert {"search_assets", "compose_assets_xml", "verify_mujoco_xml", "randomize_mujoco_xml"} <= tool_names

    composed = json.loads(replies[3]["result"]["content"][0]["text"])
    assert "<mujoco" in composed["xml"] and "mug_1" in composed["xml"]


def test_cli_assets_plugins_verify_and_randomize(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["assets", "--query", "table", "--limit", "5"]) == 0
    assets_out = json.loads(capsys.readouterr().out)
    assert assets_out["matches"]

    assert cli_main(["plugins"]) == 0
    plugins_out = json.loads(capsys.readouterr().out)
    assert len(plugins_out["plugins"]) >= 2

    pytest.importorskip("mujoco")
    xml_path = tmp_path / "base.xml"
    xml_path.write_text(
        create_base_xml("cli_demo").replace(
            "</worldbody>",
            '  <body name="box_1" pos="0 0 0.051"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="0.2"/></body>\n  </worldbody>',
        ),
        encoding="utf-8",
    )
    assert cli_main(["verify", str(xml_path), "--settle-steps", "100", "--no-render"]) == 0
    ver_out = json.loads(capsys.readouterr().out)
    assert ver_out["compiled"] is True and ver_out["settled_stable"] is True

    rnd_dir = tmp_path / "variants"
    assert cli_main(["randomize", str(xml_path), "--count", "3", "--seed", "9", "--output", str(rnd_dir)]) == 0
    rnd_out = json.loads(capsys.readouterr().out)
    assert rnd_out["stats"]["generated_count"] == 3
    assert len(list(rnd_dir.glob("variant_*.xml"))) == 3
