"""Read live Franka articulation masses/inertias without controlling the robot.

Use Isaac Python. No standalone pxr import occurs before SimulationApp starts.
The report is persisted before Kit fast-shutdown can exit this process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.backends.isaac import IsaacSimBackend
from scene_factory.dataset import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("usd", type=Path)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"schema_version": "scene_factory.franka_inertia_audit.v1", "status": "failed",
              "scope": "live_PhysX_tensor_mass_and_inertia_readback_no_control_trial",
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
              "usd_sha256": hashlib.sha256(args.usd.read_bytes()).hexdigest()}
    backend = None
    try:
        import numpy as np

        scene = json.loads(args.scene.read_text(encoding="utf-8"))
        backend = IsaacSimBackend(args.usd, headless=True, robot_asset_root=args.asset_root,
                                  enable_general_observation=True, physics_dt=1 / 240)
        backend.reset(scene)
        from pxr import UsdGeom, UsdPhysics

        view = backend._robot._articulation_view
        names = list(view.body_names)
        masses = np.asarray(view.get_body_masses(), dtype=float).reshape(-1)
        inertias = np.asarray(view.get_body_inertias(), dtype=float).reshape(-1, 3, 3)
        if len(names) != len(masses) or len(names) != len(inertias):
            raise RuntimeError("articulation body names do not match tensor dimensions")
        rows = []
        for index, name in enumerate(names):
            prim = backend._stage.GetPrimAtPath("/World/Robot/" + name)
            authored = UsdPhysics.MassAPI(prim) if prim else None
            rows.append({"name": name, "mass_kg": float(masses[index]),
                         "inertia_kg_m2": inertias[index].tolist(),
                         "authored_mass_kg": authored.GetMassAttr().Get() if authored else None,
                         "authored_diagonal_inertia_kg_m2": list(authored.GetDiagonalInertiaAttr().Get()) if authored else None})
        report.update(status="measured", rows=rows, runtime=backend.runtime_summary,
                      meters_per_unit=UsdGeom.GetStageMetersPerUnit(backend._stage),
                      kilograms_per_unit=UsdPhysics.GetStageKilogramsPerUnit(backend._stage))
    except Exception as exc:
        report.update(error={"type": type(exc).__name__, "message": str(exc)}, traceback=traceback.format_exc())
        # Verification can fail after PhysX created the articulation. Preserve
        # its unmodified tensors before Kit shutdown, including the offending
        # entries; a failed reset is never promoted to a successful audit.
        try:
            view = backend._robot._articulation_view
            centers, axes = view.get_body_coms()
            values = {"masses_kg": view.get_body_masses(), "inertias_kg_m2": view.get_body_inertias(),
                      "centers_of_mass_m": centers, "principal_axes_wxyz": axes}
            def serial(value):
                array = np.asarray(value, dtype=float)
                return np.where(np.isfinite(array), array.astype(object), np.char.mod("%s", array)).tolist()
            report["raw_live_tensors_on_failure"] = {"body_names": list(view.body_names),
                                                      **{key: serial(value) for key, value in values.items()}}
        except Exception as readback_error:
            report["raw_readback_error"] = {"type": type(readback_error).__name__, "message": str(readback_error)}
    finally:
        report["cleanup"] = "pending_Kit_process_shutdown"
        write_json_atomic(args.output, report)
        print(json.dumps({"status": report["status"], "report": str(args.output)}), flush=True)
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
