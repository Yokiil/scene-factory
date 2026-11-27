from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..models import CompiledScene, Pose
from ..asset_geometry import load_asset_parts
from ..registry import AssetRegistry


_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _value(item: Any, name: str) -> Any:
    return item[name] if isinstance(item, Mapping) else getattr(item, name)


def _numbers(values: Any) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _physics_override(item: Any, *, dynamic: bool, primitive: bool) -> dict[str, Any] | None:
    raw = item.get("physics") if isinstance(item, Mapping) else getattr(item, "physics", None)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("object.physics must be an object")
    if any(not isinstance(key, str) for key in raw):
        raise ValueError("object.physics keys must be strings")
    allowed = {"mass_kg", "inertia_kg_m2", "friction", "source"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unsupported object.physics fields: {sorted(unknown)}")
    if not dynamic:
        raise ValueError("object.physics overrides require a dynamic object")
    if not primitive:
        raise ValueError("object.physics overrides currently require primitive collision assets")
    if not isinstance(raw.get("source"), str) or not raw["source"].strip():
        raise ValueError("object.physics.source is required")
    result = dict(raw)
    def number(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("object.physics values must be finite numbers")
        try:
            converted = float(value)
        except OverflowError as exc:
            raise ValueError("object.physics value exceeds floating-point range") from exc
        if not math.isfinite(converted):
            raise ValueError("object.physics values must be finite numbers")
        return converted
    mass = number(result.get("mass_kg"))
    if mass <= 0:
        raise ValueError("object.physics.mass_kg must be finite and positive")
    inertia = result.get("inertia_kg_m2")
    if not isinstance(inertia, (list, tuple)) or len(inertia) != 3:
        raise ValueError("object.physics.inertia_kg_m2 must contain three values")
    values = [number(value) for value in inertia]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("object.physics.inertia_kg_m2 must be finite and positive")
    if any(values[index] >= values[(index + 1) % 3] + values[(index + 2) % 3] for index in range(3)):
        raise ValueError("object.physics.inertia_kg_m2 violates the triangle inequalities")
    friction = number(result.get("friction"))
    if friction < 0:
        raise ValueError("object.physics.friction must be finite and nonnegative")
    result["mass_kg"] = float(mass)
    result["inertia_kg_m2"] = values
    result["friction"] = float(friction)
    return result


class MujocoMjcfExporter:
    """Export simulator-neutral scene layouts as lightweight MuJoCo MJCF."""

    def __init__(
        self,
        registry: AssetRegistry,
        *,
        offscreen_width: int = 960,
        offscreen_height: int = 720,
    ) -> None:
        self.registry = registry
        self.offscreen_width = offscreen_width
        self.offscreen_height = offscreen_height

    @staticmethod
    def body_name(object_id: str) -> str:
        normalized = _NAME_RE.sub("_", object_id).strip("_.-") or "object"
        return f"object__{normalized}"

    @classmethod
    def object_body_names(cls, scene: CompiledScene | Mapping[str, Any]) -> dict[str, str]:
        names: dict[str, str] = {}
        used: set[str] = set()
        for index, item in enumerate(_value(scene, "objects")):
            object_id = str(_value(item, "object_id"))
            if object_id in names:
                raise ValueError(f"duplicate object_id: {object_id}")
            base = name = cls.body_name(object_id)
            suffix = index
            while name in used:
                name = f"{base}_{suffix}"
                suffix += 1
            names[object_id] = name
            used.add(name)
        return names

    def to_string(self, scene: CompiledScene | Mapping[str, Any]) -> str:
        scene_id = str(_value(scene, "scene_id"))
        room_x, room_y, room_z = _value(scene, "room_dimensions_m")
        root = ET.Element("mujoco", {"model": self.body_name(scene_id)})
        ET.SubElement(
            root,
            "compiler",
            {"angle": "degree", "autolimits": "true", "balanceinertia": "true"},
        )
        option = ET.SubElement(
            root,
            "option",
            {"timestep": "0.002", "gravity": "0 0 -9.81", "integrator": "implicitfast"},
        )
        if any("kitchen_dining_pack" in self.registry.get(str(_value(item,"asset_id"))).tags
               for item in _value(scene,"objects")):
            # Multi-point convex contacts prevent a flat mesh base balancing on
            # one moving GJK contact, which otherwise causes spurious drift.
            ET.SubElement(option,"flag",{"multiccd":"enable"})
        default = ET.SubElement(root, "default")
        ET.SubElement(
            default,
            "geom",
            {
                "condim": "4",
                "solref": "0.02 1",
                "solimp": "0.9 0.95 0.001",
            },
        )
        visual = ET.SubElement(root, "visual")
        ET.SubElement(
            visual,
            "global",
            {
                "azimuth": "135",
                "elevation": "-24",
                "offwidth": str(self.offscreen_width),
                "offheight": str(self.offscreen_height),
            },
        )
        ET.SubElement(visual, "headlight", {"ambient": "0.35 0.35 0.35"})

        mesh_assets = ET.SubElement(root, "asset")
        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": "key_light",
                "directional": "true",
                "pos": _numbers((0.0, 0.0, float(room_z) + 2.0)),
                "dir": "0 0 -1",
                "diffuse": "0.8 0.8 0.8",
            },
        )
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "room_floor",
                "type": "plane",
                "pos": "0 0 0",
                "size": _numbers((float(room_x) / 2.0, float(room_y) / 2.0, 0.05)),
                "rgba": "0.82 0.83 0.8 1",
                "friction": "0.8 0.01 0.001",
            },
        )

        names = self.object_body_names(scene)
        for item in _value(scene, "objects"):
            object_id = str(_value(item, "object_id"))
            body_name = names[object_id]
            asset = self.registry.get(str(_value(item, "asset_id")))
            pose = _value(item, "pose")
            position = _value(pose, "position")
            normalized_pose = Pose.from_dict(dict(pose)) if isinstance(pose, Mapping) else pose
            quaternion = normalized_pose.effective_orientation_wxyz
            body = ET.SubElement(
                worldbody,
                "body",
                {
                    "name": body_name,
                    "pos": _numbers(position),
                    "quat": _numbers(quaternion),
                },
            )
            dynamic = bool(_value(item, "dynamic"))
            if dynamic:
                ET.SubElement(body, "freejoint", {"name": f"joint__{body_name}"})

            bbox = tuple(float(value) for value in _value(item, "bbox_m"))
            parts=load_asset_parts(self.registry,asset)
            primitive = (asset.primitive or "cube").lower()
            physics = _physics_override(item, dynamic=dynamic, primitive=not bool(parts))
            if parts:
                if dynamic:
                    mass=asset.mass_kg
                    inertia=(mass*(bbox[1]**2+bbox[2]**2)/12,
                             mass*(bbox[0]**2+bbox[2]**2)/12,
                             mass*(bbox[0]**2+bbox[1]**2)/12)
                    ET.SubElement(body,"inertial",{"pos":"0 0 0","mass":_numbers((mass,)),
                                                   "diaginertia":_numbers(inertia)})
                for part_index,part in enumerate(parts):
                    name=f"mesh__{body_name}_{part_index}"
                    triangles=[(face[0],face[i],face[i+1]) for face in part["faces"] for i in range(1,len(face)-1)]
                    ET.SubElement(mesh_assets,"mesh",{"name":name,
                        "vertex":_numbers(x for v in part["vertices"] for x in v),
                        "face":" ".join(str(i) for triangle in triangles for i in triangle)})
                    ET.SubElement(body,"geom",{"name":f"geom__{body_name}_{part_index}","type":"mesh","mesh":name,
                        "mass":"0","rgba":_numbers((*part["color"],1)),
                        "friction":_numbers((asset.friction,.008,.001)),
                        "contype":"1" if asset.collision_enabled else "0",
                        "conaffinity":"1" if asset.collision_enabled else "0"})
                continue
            geom_type, size = self._geom_shape(primitive, bbox)
            static_friction = physics["friction"] if physics else (asset.static_friction or asset.friction or 0.5)
            dynamic_friction = physics["friction"] if physics else (asset.dynamic_friction or asset.friction or 0.5)
            geom_attributes = {
                "name": f"geom__{body_name}",
                "type": geom_type,
                "size": _numbers(size),
                "rgba": _numbers((*asset.color, 1.0)),
                "friction": _numbers((static_friction, dynamic_friction * 0.02, 0.001)),
                "contype": "1" if asset.collision_enabled else "0",
                "conaffinity": "1" if asset.collision_enabled else "0",
            }
            if dynamic:
                geom_attributes["mass"] = f"{max(float(physics["mass_kg"] if physics else asset.mass_kg), 1e-6):.9g}"
            ET.SubElement(body, "geom", geom_attributes)
            if physics:
                ET.SubElement(body, "inertial", {
                    "pos": "0 0 0", "mass": _numbers((physics["mass_kg"],)),
                    "diaginertia": _numbers(physics["inertia_kg_m2"]),
                })

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode", xml_declaration=False) + "\n"

    def export(
        self,
        scene: CompiledScene | Mapping[str, Any],
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_string(scene), encoding="utf-8")
        return output_path

    @staticmethod
    def _geom_shape(
        primitive: str,
        bbox: tuple[float, float, float],
    ) -> tuple[str, tuple[float, ...]]:
        if primitive == "sphere":
            return "sphere", (max(bbox) / 2.0,)
        if primitive in {"cylinder", "capsule"}:
            return primitive, (max(bbox[0], bbox[1]) / 2.0, bbox[2] / 2.0)
        return "box", tuple(value / 2.0 for value in bbox)
