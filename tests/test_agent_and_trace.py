"""End-to-end tests for the multimodal SceneAgent, self-correction loop, and SFT trace export."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from scene_factory.cordis import CordisContext, apply_cordis_config
from scene_factory.core import (
    AssetCatalog,
    LLMResponse,
    SceneAgent,
    ToolCall,
    export_sft_jsonl,
)


def _tiny_png_bytes() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    raw = b"\x00\xff\x00\x00\x00\xff\x00\x00\x00\x00\xff\xff\xff\xff"
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class ScriptedSelfCorrectingVLM:
    """Simulates a VLM that builds, sees MuJoCo failure/renders, self-corrects, and verifies."""

    model = "mock-vlm-2026"

    def __init__(self, table_h: float, mug_h: float) -> None:
        self.table_h = table_h
        self.mug_h = mug_h
        self.calls = 0
        self.seen_messages: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.calls += 1
        self.seen_messages.append(list(messages))

        if self.calls == 1:
            return LLMResponse(
                content="Let me search for a table and mug in the asset library first.",
                tool_calls=(ToolCall("c1", "search_assets", {"query": "table mug"}),),
                usage={"prompt_tokens": 120, "completion_tokens": 30},
            )
        if self.calls == 2:
            # Deliberately place the mug intersecting the table to trigger physical self-correction.
            return LLMResponse(
                content="Inserting table and mug, then running MuJoCo verification.",
                tool_calls=(
                    ToolCall(
                        "c2",
                        "insert_asset",
                        {"asset_id": "kd_table_01", "name": "table_1", "pos": [0.0, 0.0, self.table_h / 2.0], "dynamic": False},
                    ),
                    ToolCall(
                        "c3",
                        "insert_asset",
                        {"asset_id": "kd_mug_01", "name": "mug_1", "pos": [0.0, 0.0, self.table_h - 0.04], "dynamic": True},
                    ),
                    ToolCall("c4", "verify_xml", {"settle_steps": 80, "render": True}),
                ),
                usage={"prompt_tokens": 220, "completion_tokens": 60},
            )
        if self.calls == 3:
            # Self-correct mug z height after reading MuJoCo diagnostics and rendered views.
            return LLMResponse(
                content="The mug penetrated the tabletop and moved. Raising it cleanly onto the top surface.",
                tool_calls=(
                    ToolCall(
                        "c5",
                        "insert_asset",
                        {
                            "asset_id": "kd_mug_01",
                            "name": "mug_1",
                            "pos": [0.0, 0.0, self.table_h + self.mug_h / 2.0 + 0.002],
                            "dynamic": True,
                        },
                    ),
                    ToolCall("c6", "verify_xml", {"settle_steps": 80, "render": True}),
                ),
                usage={"prompt_tokens": 350, "completion_tokens": 55},
            )
        return LLMResponse(
            content="The high-fidelity MuJoCo XML scene is verified and stable.",
            tool_calls=(),
            usage={"prompt_tokens": 410, "completion_tokens": 20},
        )


def test_multimodal_self_correcting_agent_and_sft_export(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    ref_img = tmp_path / "reference.png"
    ref_img.write_bytes(_tiny_png_bytes())

    catalog = AssetCatalog()
    table_h = catalog.get("kd_table_01").bbox_m[2]
    mug_h = catalog.get("kd_mug_01").bbox_m[2]

    cordis = CordisContext()
    apply_cordis_config(cordis, Path("config/cordis.toml"))

    vlm = ScriptedSelfCorrectingVLM(table_h, mug_h)
    agent = SceneAgent(vlm, catalog=catalog, cordis=cordis, max_turns=8)
    out_dir = tmp_path / "run_01"
    result = agent.run(
        "根据参考图生成一个餐桌和马克杯的 MuJoCo 场景",
        images=[ref_img],
        output_dir=out_dir,
        randomize_count=3,
        seed=7,
    )

    assert result.verification.compiled is True
    assert result.verification.settled_stable is True
    assert result.verification.score == 1.0
    assert result.trace.turns == 4
    assert len(result.variants) == 3
    assert (out_dir / "scene.xml").is_file()
    assert (out_dir / "trace.json").is_file()
    assert len(list((out_dir / "variants").glob("variant_*.xml"))) == 3

    # Verify input image reached turn 1 and rendered MuJoCo views reached turn 3.
    first_user = vlm.seen_messages[0][1]["content"]
    assert isinstance(first_user, list) and any(item.get("type") == "image_url" for item in first_user)
    if result.verification.render_error is None:
        turn3_roles = [m["role"] for m in vlm.seen_messages[2]]
        assert turn3_roles.count("user") >= 2

    # Export SFT JSONL directly from saved trace.json.
    sft_out = tmp_path / "sft.jsonl"
    export_info = export_sft_jsonl([out_dir / "trace.json"], sft_out, min_score=0.9)
    assert export_info["exported"] == 1
    row = json.loads(sft_out.read_text(encoding="utf-8").strip())
    assert row["reward"] == 1.0
    assert "mjcf_prompt_hints" in row["cordis_plugin_names"]
    assert any(m["role"] == "tool" for m in row["messages"])


def test_naked_agent_runs_with_zero_cordis_plugins(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    catalog = AssetCatalog()
    table_h = catalog.get("kd_table_01").bbox_m[2]
    mug_h = catalog.get("kd_mug_01").bbox_m[2]

    # Empty CordisContext proves no constraints are welded into SceneAgent.
    agent = SceneAgent(
        ScriptedSelfCorrectingVLM(table_h, mug_h),
        catalog=catalog,
        cordis=CordisContext(),
    )
    result = agent.run("table and mug", output_dir=tmp_path / "naked")
    assert result.verification.settled_stable is True
    assert result.trace.cordis_plugins == []
