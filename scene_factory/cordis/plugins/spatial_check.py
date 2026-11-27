"""Optional Cordis plugin: zero-render analytical XML collision & floating hint.

Runs purely on the MuJoCo XML geometry tree (no renderer, no physics engine)
after `write_xml` / `edit_xml` / `insert_asset` and appends concise factual
hints when interpenetration or floating bodies are detected. Toggle via
`config/cordis.toml` (`enabled = true/false`).
"""

from __future__ import annotations

from typing import Any

from ...spatial.geometry import analyze_xml_spatial
from ..kernel import PluginScope


def plugin(scope: PluginScope, config: dict[str, Any]) -> None:
    collision_tol = float(config.get("collision_tol_m", 0.004))
    floating_tol = float(config.get("floating_tol_m", 0.02))

    def _after_tool(
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        _state: dict[str, Any],
    ) -> dict[str, Any] | None:
        if tool_name != "write_xml" or not result.get("ok"):
            return None
        xml = str(arguments.get("xml", ""))
        if not xml:
            return None
        try:
            analysis = analyze_xml_spatial(
                xml, collision_tol_m=collision_tol, floating_tol_m=floating_tol
            )
        except Exception:
            return None
        if analysis.assembly_valid:
            return None
        notes: list[str] = []
        for pair in analysis.colliding_pairs[:4]:
            notes.append(
                f"analytical collision: {pair.body_a} <-> {pair.body_b} "
                f"(penetration={pair.penetration_m:.4f}m, sep_axis={[round(v, 2) for v in pair.separation_axis]})"
            )
        for flt in analysis.floating_bodies[:4]:
            notes.append(
                f"analytical floating: {flt['body']} (clearance={flt['clearance_m']:.4f}m above {flt['supporter']})"
            )
        updated = dict(result)
        existing = list(updated.get("advisory_hints", []))
        existing.extend(f"[xml_spatial_hint] {note}" for note in notes)
        updated["advisory_hints"] = existing
        return updated

    scope.on_after_tool(_after_tool)
