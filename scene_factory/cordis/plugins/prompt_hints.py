"""Prompt-injection plugin guiding high-fidelity MuJoCo MJCF construction.

Preferred Cordis pattern: guide the LLM's reasoning trace via prompt injection
rather than welding Python interception functions into the agent loop.
"""

from __future__ import annotations

from typing import Any

from ..kernel import PluginScope


_DEFAULT_GUIDANCE = """\
MuJoCo MJCF authoring guidelines (advisory):
- Coordinate frame: Z-up, meters, radians/degrees per <compiler angle="degree"/>.
- Box geom `size` uses HALF-extents (hx, hy, hz), not full dimensions; cylinder `size` uses (radius, half_height).
- Place resting objects cleanly on top of support surfaces: body_z = support_top_z + object_half_height + 0.001 to avoid initial interpenetration.
- Add `<freejoint/>` to movable tabletop/floor items; keep large static furniture without `<freejoint/>`.
- Prefer `search_assets` + `insert_asset` when a high-fidelity multi-part library asset matches the request; use `write_xml` or `edit_xml` freely for custom geometry, lighting, walls, and spatial composition.
- Always call `verify_xml` after authoring or editing XML, inspect both the physical settling report AND the rendered views against the user request/reference image, and self-correct any collisions, falls, or layout discrepancies."""


def plugin(scope: PluginScope, config: dict[str, Any]) -> None:
    extra = str(config.get("extra_prompt", "")).strip()
    include_defaults = bool(config.get("include_defaults", True))

    def _inject(state: dict[str, Any]) -> list[str]:
        parts: list[str] = []
        if include_defaults:
            parts.append(_DEFAULT_GUIDANCE)
        if state.get("has_images"):
            parts.append(
                "Reference image(s) were provided: match object count, relative spatial layout, "
                "support surfaces, approximate scale, and color palette by comparing the MuJoCo "
                "rendered views against the input image(s)."
            )
        if extra:
            parts.append(extra)
        return parts

    scope.on_prompt(_inject)
