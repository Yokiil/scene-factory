"""High-fidelity MuJoCo MJCF base templates, fragment merging, and content-addressed caching."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .assets import AssetCatalog, sanitize_mjcf_name


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unindent(elem: ET.Element) -> None:
    for node in elem.iter():
        if node.text is not None and not node.text.strip():
            node.text = None
        if node.tail is not None and not node.tail.strip():
            node.tail = None


def format_xml(root: ET.Element) -> str:
    _unindent(root)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False) + "\n"


def create_base_xml(
    model_name: str = "scene_factory_scene",
    *,
    floor_half_size: tuple[float, float] = (3.0, 3.0),
    offwidth: int = 640,
    offheight: int = 480,
) -> str:
    """Create a clean, physically sensible MuJoCo MJCF starting skeleton with cameras."""
    root = ET.Element("mujoco", {"model": sanitize_mjcf_name(model_name)})
    ET.SubElement(
        root,
        "compiler",
        {"angle": "degree", "autolimits": "true", "balanceinertia": "true"},
    )
    ET.SubElement(
        root,
        "option",
        {"timestep": "0.002", "gravity": "0 0 -9.81", "integrator": "implicitfast"},
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        {"condim": "4", "solref": "0.02 1", "solimp": "0.9 0.95 0.001"},
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        {
            "azimuth": "135",
            "elevation": "-25",
            "offwidth": str(offwidth),
            "offheight": str(offheight),
        },
    )
    ET.SubElement(visual, "headlight", {"ambient": "0.35 0.35 0.35", "diffuse": "0.6 0.6 0.6"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "grid_tex",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.86 0.87 0.85",
            "rgb2": "0.78 0.80 0.78",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "floor_mat", "texture": "grid_tex", "texrepeat": "6 6", "reflectance": "0.05"},
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        {
            "name": "key_light",
            "directional": "true",
            "pos": "0 0 4.0",
            "dir": "0.2 0.2 -1",
            "diffuse": "0.85 0.85 0.85",
        },
    )
    ET.SubElement(
        world,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": "0 0 0",
            "size": f"{floor_half_size[0]:.6g} {floor_half_size[1]:.6g} 0.05",
            "material": "floor_mat",
            "friction": "0.8 0.01 0.001",
        },
    )
    ET.SubElement(
        world,
        "camera",
        {"name": "cam_overview", "pos": "2.4 -2.4 2.1", "xyaxes": "0.707 0.707 0 -0.408 0.408 0.816"},
    )
    ET.SubElement(
        world,
        "camera",
        {"name": "cam_front", "pos": "0 -2.6 1.5", "xyaxes": "1 0 0 0 0.38 0.925"},
    )
    return format_xml(root)


def merge_fragment(
    scene_xml: str,
    asset_elements: list[ET.Element],
    body_element: ET.Element,
    *,
    enable_multiccd: bool = False,
    replace_existing: bool = True,
) -> str:
    """Insert or replace one `<body/>` and its `<asset>` meshes inside a MuJoCo XML."""
    root = ET.fromstring(scene_xml)
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    asset_node = root.find("asset")
    if asset_node is None and asset_elements:
        asset_node = ET.Element("asset")
        world_idx = list(root).index(world)
        root.insert(world_idx, asset_node)

    target_name = body_element.attrib.get("name", "")
    if replace_existing and target_name:
        existing_bodies = [b for b in world.findall("body") if b.attrib.get("name") == target_name]
        removed_meshes: set[str] = set()
        for existing in existing_bodies:
            for geom in existing.iter("geom"):
                mesh_ref = geom.attrib.get("mesh")
                if mesh_ref:
                    removed_meshes.add(mesh_ref)
            world.remove(existing)
        if asset_node is not None and removed_meshes:
            for mesh in list(asset_node.findall("mesh")):
                if mesh.attrib.get("name") in removed_meshes:
                    asset_node.remove(mesh)

    if asset_node is not None:
        existing_mesh_names = {m.attrib.get("name") for m in asset_node.findall("mesh")}
        for elem in asset_elements:
            if elem.attrib.get("name") not in existing_mesh_names:
                asset_node.append(elem)

    world.append(body_element)

    if enable_multiccd:
        option = root.find("option")
        if option is None:
            option = ET.Element("option")
            root.insert(0, option)
        flag = option.find("flag")
        if flag is None:
            ET.SubElement(option, "flag", {"multiccd": "enable"})
        else:
            flag.set("multiccd", "enable")

    return format_xml(root)


def patch_xml(scene_xml: str, old_text: str, new_text: str) -> str:
    """Apply one exact substring replacement to a MuJoCo XML document."""
    if not old_text:
        raise ValueError("old_text must be non-empty")
    count = scene_xml.count(old_text)
    if count == 0:
        raise ValueError("old_text was not found in the current XML")
    if count > 1:
        raise ValueError(f"old_text matched {count} times; provide more surrounding context")
    return scene_xml.replace(old_text, new_text, 1)


def inspect_xml_summary(scene_xml: str) -> dict[str, Any]:
    """Return a compact, factual structural overview of a MuJoCo XML string."""
    root = ET.fromstring(scene_xml)
    world = root.find("worldbody")
    bodies: list[dict[str, Any]] = []
    cameras: list[str] = []
    if world is not None:
        for body in world.findall("body"):
            bodies.append(
                {
                    "name": body.attrib.get("name", ""),
                    "pos": body.attrib.get("pos", "0 0 0"),
                    "quat": body.attrib.get("quat", "1 0 0 0"),
                    "dynamic": body.find("freejoint") is not None,
                    "geoms": len(body.findall("geom")),
                }
            )
        cameras = [c.attrib.get("name", "") for c in world.findall("camera") if c.attrib.get("name")]
    asset_node = root.find("asset")
    mesh_count = len(asset_node.findall("mesh")) if asset_node is not None else 0
    return {
        "model": root.attrib.get("model", ""),
        "body_count": len(bodies),
        "dynamic_count": sum(1 for b in bodies if b["dynamic"]),
        "mesh_count": mesh_count,
        "cameras": cameras,
        "bodies": bodies,
        "xml_bytes": len(scene_xml.encode("utf-8")),
        "xml_sha256": _sha256_text(scene_xml),
    }


class FragmentCache:
    """Content-addressed LRU cache for library asset MJCF fragments.

    Enables fast, low-cost incremental assembly when multiple objects or layout
    variants reuse unchanged high-fidelity mesh assets.
    """

    def __init__(self, catalog: AssetCatalog, *, max_entries: int = 2048) -> None:
        self.catalog = catalog
        self.max_entries = max(1, int(max_entries))
        self._cache: OrderedDict[str, tuple[list[str], str, bool]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get_or_build(
        self,
        asset_id: str,
        *,
        name: str,
        pos: tuple[float, float, float],
        yaw_deg: float = 0.0,
        dynamic: bool = True,
        scale: float = 1.0,
        mass_kg: float | None = None,
        friction: float | None = None,
        rgba: tuple[float, float, float, float] | None = None,
    ) -> tuple[list[ET.Element], ET.Element, bool, bool]:
        """Return `(asset_elems, body_elem, uses_detailed_mesh, cache_hit)`."""
        entry = self.catalog.get(asset_id)
        key = hashlib.sha256(
            _canonical_bytes(
                {
                    "asset_id": asset_id,
                    "geometry_sha256": entry.geometry_sha256,
                    "name": name,
                    "pos": [float(v) for v in pos],
                    "yaw_deg": float(yaw_deg),
                    "dynamic": bool(dynamic),
                    "scale": float(scale),
                    "mass_kg": None if mass_kg is None else float(mass_kg),
                    "friction": None if friction is None else float(friction),
                    "rgba": None if rgba is None else [float(v) for v in rgba],
                }
            )
        ).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            self._cache.move_to_end(key)
            asset_strs, body_str, detailed = cached
            return (
                [ET.fromstring(s) for s in asset_strs],
                ET.fromstring(body_str),
                detailed,
                True,
            )

        self.misses += 1
        asset_elems, body_elem, detailed = self.catalog.build_mjcf_fragment(
            asset_id,
            name=name,
            pos=pos,
            yaw_deg=yaw_deg,
            dynamic=dynamic,
            scale=scale,
            mass_kg=mass_kg,
            friction=friction,
            rgba=rgba,
        )
        self._cache[key] = (
            [ET.tostring(e, encoding="unicode") for e in asset_elems],
            ET.tostring(body_elem, encoding="unicode"),
            detailed,
        )
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return asset_elems, body_elem, detailed, False

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._cache)}


def write_xml_atomic(path: str | Path, xml: str) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(xml, encoding="utf-8")
    tmp.replace(dest)
    return dest
