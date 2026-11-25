"""Check that the assembled dessert display settles in Isaac Sim/PhysX."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args()
    source = args.usd.resolve()
    report_path = args.report.resolve()
    if not str(source).isascii() or not str(report_path).isascii():
        raise ValueError("Isaac scene and report paths must be ASCII-only")
    if args.steps < 60:
        raise ValueError("steps must be at least 60")

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    sys.argv = [sys.argv[0]]
    from isaac_kit_compat import load_simulation_app

    app = load_simulation_app()({
        "headless": True, "hide_ui": True, "renderer": "Minimal",
        "multi_gpu": False, "max_gpu_count": 1,
        "disable_viewport_updates": True, "fast_shutdown": True,
        "open_usd": source.as_posix(),
    })
    exit_code = 1
    try:
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        parent = stage.GetPrimAtPath("/World/Objects")
        cakes = [prim for prim in parent.GetChildren() if prim.GetName() != "stand_1"]
        if len(cakes) != 42:
            raise RuntimeError(f"expected 42 cakes, found {len(cakes)}")

        def position(prim):
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return tuple(float(value) for value in matrix.ExtractTranslation())

        initial = {prim.GetName(): position(prim) for prim in cakes}
        simulation = SimulationContext(
            physics_dt=1 / 60, rendering_dt=1 / 60,
            stage_units_in_meters=1, physics_prim_path="/World/PhysicsScene", stage=stage,
        )
        simulation.get_physics_context().set_physx_update_transformations_settings(
            update_to_usd=True, update_velocities_to_usd=True,
        )
        simulation.initialize_physics()
        simulation.play()
        for _ in range(args.steps):
            simulation.step(render=False)

        objects = []
        for prim in cakes:
            name = prim.GetName()
            final = position(prim)
            drift_xy = math.dist(initial[name][:2], final[:2])
            drift_z = abs(initial[name][2] - final[2])
            velocity = prim.GetAttribute("physics:velocity").Get()
            speed = math.sqrt(sum(float(value) ** 2 for value in velocity)) if velocity else None
            passed = (all(math.isfinite(value) for value in final)
                      and drift_xy < 0.02 and drift_z < 0.02
                      and speed is not None and speed < 0.05)
            objects.append({"object_id": name, "drift_xy_m": drift_xy,
                            "drift_z_m": drift_z, "speed_mps": speed, "passed": passed})
        simulation.stop()
        report = {
            "usd": str(source), "steps": args.steps, "dt_s": 1 / 60,
            "valid": all(item["passed"] for item in objects),
            "max_drift_xy_m": max(item["drift_xy_m"] for item in objects),
            "max_drift_z_m": max(item["drift_z_m"] for item in objects),
            "failed_objects": [item for item in objects if not item["passed"]],
            "objects": objects,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        exit_code = 0 if report["valid"] else 2
    finally:
        app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
