"""Upgrade a compiled layout without changing its world transforms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.dataset import write_json_atomic
from scene_factory.pose_migration import upgrade_scene_poses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.scene.resolve():
        raise ValueError("use a new output path to preserve the source layout")
    scene = json.loads(args.scene.read_text(encoding="utf-8-sig"))
    write_json_atomic(args.output, upgrade_scene_poses(scene))
    print(json.dumps({"output": str(args.output), "schema_version": "scene_factory.compiled_scene.v2",
                      "quaternion_order": "wxyz", "length_unit": "m", "object_count": len(scene["objects"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
