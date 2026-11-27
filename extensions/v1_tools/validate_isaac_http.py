"""Measure a real Isaac HTTP session from frozen USD assets through RGB-D download.

Launch using Isaac Python. This starts a loopback service and one spawned Kit
worker, never a second in-process SimulationApp. Build uses OpenUSD only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.dataset import write_json_atomic
from scene_factory.joint_control import ARM_JOINT_NAMES
from scene_factory.runtime_client import RuntimeClient
from scene_factory.runtime_service import RuntimeService, serve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("usd", type=Path)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True,
                        help="ASCII absolute directory for Kit paths on Windows")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        raise FileExistsError("use a new output directory to retain the previous result")
    if args.runtime_root.exists():
        raise FileExistsError("runtime-root must be a new directory")
    report = {"schema_version": "scene_factory.isaac_http_acceptance.v1", "status": "failed",
              "scope": "loopback_HTTP_real_Isaac_joint_control_and_RGBD_not_assembly_or_Linux_acceptance",
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checks": {}, "timings_s": {}}
    manager = server = client = session_id = None
    thread = None
    try:
        import numpy as np

        # Trusted local inputs can live on the F: drive; HTTP still cannot read
        # outside that explicitly selected root.
        manager = RuntimeService(args.runtime_root, allowed_scene_root=Path(args.runtime_root.anchor),
                                 max_sessions=1, startup_timeout=180, request_timeout=180,
                                 lease_seconds=900, memory_budget_mb=16384)
        server = serve(manager, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = RuntimeClient(f"http://127.0.0.1:{server.server_address[1]}", timeout=240)
        started = time.perf_counter()
        build = client.build({"idempotency_key": "isaac-http-build", "backend": "isaac",
                              "scene_path": str(args.scene.resolve()), "usd_path": str(args.usd.resolve()),
                              "robot_asset_root": str(args.asset_root.resolve()),
                              "backend_options": {"enable_rgbd": True, "enable_general_observation": True,
                                                  "max_steps": args.steps + 20}})
        deadline = time.monotonic() + 300
        while build["status"] in {"queued", "running"}:
            if time.monotonic() > deadline:
                raise TimeoutError("HTTP environment build timed out")
            time.sleep(0.1)
            build = client.build_status(build["build_id"])
        if build["status"] != "completed":
            raise RuntimeError(f"HTTP build failed: {build}")
        report["timings_s"]["asset_build"] = time.perf_counter() - started
        report["build"] = build
        create_request = {"idempotency_key": "isaac-http-create", "backend": "isaac",
                          "environment_path": build["result"]["package_dir"],
                          "resources": {"memory_mb": 8192, "cpu_threads": 4}}
        created_at = time.perf_counter()
        created = client.create(create_request)
        session_id = created["session_id"]
        report["checks"]["verified_package"] = created["inputs"]["asset_closure"] == "verified_package"
        report["checks"]["create_retry_same_session"] = client.create(create_request)["session_id"] == session_id
        reset_request = {"seed": 0, "options": {}, "idempotency_key": "isaac-http-reset"}
        initial = client.reset(session_id, reset_request)
        report["checks"]["full_object_motion_available"] = all(
            all(field in state for field in ("orientation_wxyz", "linear_velocity_mps", "angular_velocity_rps"))
            for state in initial["observation"]["objects"].values())
        coverage = initial["observation"].get("contact_coverage", {})
        report["checks"]["general_contact_coverage_available"] = (
            coverage.get("status") == "measured" and coverage.get("complete") is True
            and coverage.get("includes_instance_proxies") is True
            and coverage.get("force_aggregation") == "sum_per_point_normal_magnitudes"
            and coverage.get("buffer_saturated") is False
        )
        report["timings_s"]["new_worker_to_initial_observation"] = time.perf_counter() - created_at
        report["checks"]["reset_retry_identical"] = client.reset(session_id, reset_request) == initial
        robot = initial["observation"]["robot"]
        names = robot.get("joint_names", robot.get("dof_names"))
        positions = robot["joint_positions"]
        arm = [float(positions[names.index(name)]) for name in ARM_JOINT_NAMES] if names else [float(v) for v in positions[:7]]
        arm[0] += 0.01
        step_request = {"episode_id": initial["episode_id"], "sequence_id": 0,
                        "idempotency_key": "isaac-http-action", "steps": args.steps,
                        "action": {"mode": "joint_position", "payload": {
                            "arm_positions_rad": arm, "gripper_open_fraction": 1.0}}}
        started = time.perf_counter()
        stepped = client.step(session_id, step_request)
        report["timings_s"]["action_response"] = time.perf_counter() - started
        report["checks"]["step_retry_identical"] = client.step(session_id, step_request) == stepped
        observed = client.observe(session_id, modalities=["state", "rgb", "depth"])
        report["checks"]["image_state_alignment"] = (
            observed["step_index"] == args.steps
            and observed["state_version"] == stepped["state_version"]
            and {item["sim_time"] for item in observed["images"]} == {stepped["sim_time"]})
        images = []
        for descriptor in observed["images"]:
            name = Path(descriptor["path"]).name
            data = client.artifact(session_id, name, sha256=descriptor["sha256"])
            target = args.output / "artifacts" / name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(data)
            array = np.load(io.BytesIO(data), allow_pickle=False)
            if list(array.shape) != descriptor["shape"] or str(array.dtype) != descriptor["dtype"]:
                raise ValueError("downloaded sensor array does not match descriptor")
            images.append({**descriptor, "downloaded_path": str(target.resolve())})
        report["checks"]["artifact_downloads_verified"] = len(images) == 2
        report.update(created=created, initial=initial, action=step_request, final=stepped,
                      observation=observed, downloaded_artifacts=images)
        closed = client.close(session_id)
        report["checks"]["worker_reclaimed"] = closed["closed"] and closed["status"] == "closed"
        report["closed"] = closed
        if not all(report["checks"].values()):
            raise RuntimeError("one or more real HTTP checks failed")
        report["status"] = "passed"
    except Exception as exc:
        report.update(error={"type": type(exc).__name__, "message": str(exc)}, traceback=traceback.format_exc())
    finally:
        if client is not None and session_id is not None:
            try:
                report["cleanup_session"] = client.close(session_id)
            except Exception as exc:
                report["cleanup_error"] = str(exc)
                report["status"] = "failed"
        if server is not None:
            server.shutdown()
            server.server_close()
        if manager is not None:
            manager.close()
        if thread is not None:
            thread.join(5)
        report["cleanup"] = "service_stopped"
        write_json_atomic(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
