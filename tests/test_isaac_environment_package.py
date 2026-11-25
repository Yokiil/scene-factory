from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scene_factory.environment_package import build_environment, verify_environment


def _parser_free_package(root):
    from scene_factory.environment_package import LOCK_SCHEMA, PACKAGE_SCHEMA, canonical_bytes
    from scene_factory.isaac_environment_package import ISAAC_CLOSURE

    contents = {
        "scene/layout.json": json.dumps({"scene_id": "prekit", "objects": [], "task": {}}).encode(),
        "scene/scene.usda": b'#usda 1.0\ndef Xform "fixture" {}\n',
        "assets/robot.usda": b'#usda 1.0\ndef Xform "robot" {}\n',
        "assets/registry.jsonl": b"",
    }
    for name, data in contents.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    runtime = {"engine_version": "test", "python_version": "test", "distributions": {},
               "source_sha256": {"test.py": "test-hash"}, "extension_config_sha256": {},
               "runtime_builtins": {}, "mdl_runtime_files_sha256": {}}
    lock = {"schema_version": LOCK_SCHEMA, "backend": "isaac", "dependency_closure": ISAAC_CLOSURE,
            "runtime": runtime, "dependencies": [
                {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in contents.items()],
            "scene": {"layout_path": "scene/layout.json", "registry_path": "assets/registry.jsonl",
                      "compiled_path": "scene/scene.usda", "robot_usd_path": "assets/robot.usda"}}
    digest = hashlib.sha256(canonical_bytes(lock)).hexdigest()
    lock["environment_sha256"] = digest
    (root / "runtime").mkdir()
    (root / "runtime/lock.json").write_bytes(canonical_bytes(lock))
    (root / "manifest.json").write_bytes(canonical_bytes({"schema_version": PACKAGE_SCHEMA,
        "backend": "isaac", "environment_sha256": digest, "lock_path": "runtime/lock.json"}))
    return runtime


def test_parser_free_verification_still_checks_bytes_required_files_and_current_runtime(tmp_path, monkeypatch):
    from scene_factory import isaac_environment_package as package_module

    runtime = _parser_free_package(tmp_path)
    monkeypatch.setattr(package_module, "_runtime_lock", lambda: runtime)
    monkeypatch.setattr(package_module, "_collect", lambda *_: pytest.fail("USD resolver must not run before Kit"))
    result = verify_environment(tmp_path, resolve_assets=False)
    assert result["valid"] and result["runtime_checked"]
    assert not result["usd_dependency_resolution_checked"]
    runtime["source_sha256"]["test.py"] = "changed-current-runtime"
    result = verify_environment(tmp_path, resolve_assets=False)
    assert not result["valid"] and "runtime_mismatch:source_sha256" in result["errors"]
    runtime["source_sha256"]["test.py"] = "test-hash"
    (tmp_path / "scene/scene.usda").write_bytes(b"tampered")
    result = verify_environment(tmp_path, resolve_assets=False)
    assert not result["valid"] and "dependency_mismatch:scene/scene.usda" in result["errors"]


@pytest.mark.parametrize("tamper", [False, True])
def test_isaac_worker_prekit_verification_does_not_import_pxr(tmp_path, tamper):
    import subprocess
    import sys
    import textwrap

    package = tmp_path / "package"
    _parser_free_package(package)
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    if tamper:
        (package / "assets/robot.usda").write_bytes(b"tampered")
    # A fresh interpreter proves import order; no SimulationApp or GPU is used.
    # Runtime metadata is a fixture here, while every frozen byte is verified.
    script = textwrap.dedent('''
        import json, sys
        from pathlib import Path
        from scene_factory import isaac_environment_package as package_module
        from scene_factory.sandbox_worker import worker_main
        root, session_dir = map(Path, sys.argv[1:3])
        lock = json.loads((root / "runtime/lock.json").read_text())
        package_module._runtime_lock = lambda: lock["runtime"]
        def forbidden(*args):
            raise AssertionError("USD parser was called before Kit")
        package_module._collect = forbidden
        class Connection:
            messages = []
            def send(self, value): self.messages.append(value)
            def recv(self): raise EOFError
            def close(self): pass
        connection = Connection()
        worker_main(connection, {
            "resources": {"cpu_threads": 1}, "backend": "isaac", "backend_options": {},
            "environment_path": str(root), "usd_path": str(root / "scene/scene.usda"),
            "scene_path": str(root / "scene/layout.json"), "session_id": "prekit-test",
            "session_dir": str(session_dir),
        })
        assert not any(name == "pxr" or name.startswith("pxr.") for name in sys.modules)
        print(json.dumps(connection.messages))
    ''')
    result = subprocess.run([sys.executable, "-c", script, str(package), str(session_dir)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    first = json.loads(result.stdout.strip().splitlines()[-1])[0]
    assert first["ok"] is not tamper
    if tamper:
        assert "dependency_mismatch:assets/robot.usda" in first["error"]["message"]


def test_isaac_builder_requires_explicit_asset_inputs(tmp_path):
    with pytest.raises(ValueError, match="usd_path and robot_asset_root"):
        build_environment(tmp_path / "layout.json", tmp_path / "package", backend="isaac")


def test_isaac_portable_usd_dependencies_and_tamper_detection(tmp_path, monkeypatch):
    Sdf = pytest.importorskip("pxr.Sdf")
    pytest.importorskip("pxr.Usd")  # registers USD file-format plugins before CreateNew
    from scene_factory import isaac_environment_package as package_module

    # This verifies real OpenUSD dependency traversal/rewriting. The engine lock
    # is replaced only to allow this asset-unit test outside a Kit installation.
    runtime = {"engine": "isaac", "engine_version": "test", "python_version": "test",
               "distributions": {}, "source_sha256": {}, "extension_config_sha256": {},
               "runtime_builtins": {}, "mdl_runtime_files_sha256": {}}
    monkeypatch.setattr(package_module, "_runtime_lock", lambda: runtime)
    scene = tmp_path / "source"
    scene.mkdir()
    child = Sdf.Layer.CreateNew((scene / "child.usda").as_posix())
    child.ImportFromString('#usda 1.0\ndef Xform "fixture" {}\n')
    child.Save()
    root = Sdf.Layer.CreateNew((scene / "scene.usda").as_posix())
    root.subLayerPaths.append((scene / "child.usda").as_posix())
    root.Save()
    robot_path = scene / "robot" / package_module.FRANKA_RELATIVE
    robot_path.parent.mkdir(parents=True)
    robot = Sdf.Layer.CreateNew(robot_path.as_posix())
    robot.ImportFromString('#usda 1.0\ndef Xform "robot" {}\n')
    robot.Save()
    layout = scene / "layout.json"
    layout.write_text(json.dumps({"scene_id": "unit", "objects": [{"object_id": "part"}]}), encoding="utf-8")
    registry = scene / "registry.jsonl"
    registry.write_text("", encoding="utf-8")
    output = tmp_path / "package"
    result = build_environment(layout, output, backend="isaac", usd_path=root.realPath,
                               robot_asset_root=scene / "robot", registry_path=registry)
    assert result["verification"]["valid"]
    # A package remains self-contained after its source layer is removed.
    Path(child.realPath).unlink()
    Path(root.realPath).unlink()
    assert verify_environment(output)["valid"]
    lock = json.loads((output / "runtime/lock.json").read_text())
    copied = next(row for row in lock["dependencies"] if row["path"].endswith("child.usda"))
    assert hashlib.sha256((output / copied["path"]).read_bytes()).hexdigest() == copied["sha256"]
    (output / copied["path"]).write_text("tampered", encoding="utf-8")
    assert not verify_environment(output, check_runtime=False)["valid"]


def test_runtime_client_artifact_hash_check(tmp_path):
    import threading
    from scene_factory.runtime_client import RuntimeClient
    from scene_factory.runtime_service import RuntimeService, serve

    manager = RuntimeService(tmp_path / "service", allowed_scene_root=tmp_path)
    server = serve(manager, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = RuntimeClient(f"http://127.0.0.1:{server.server_port}")
    try:
        created = client.create({"scene": {"scene_id": "artifact", "objects": [], "task": {}}, "backend": "dry_run"})
        sid = created["session_id"]
        artifacts = manager.sessions_root / sid / "artifacts"
        (artifacts / "sample.npy").write_bytes(b"sensor")
        assert client.artifact(sid, "sample.npy", sha256=hashlib.sha256(b"sensor").hexdigest()) == b"sensor"
        with pytest.raises(ValueError, match="digest"):
            client.artifact(sid, "sample.npy", sha256="0" * 64)
        with pytest.raises(ValueError):
            client.artifact(sid, "../secret")
    finally:
        server.shutdown()
        server.server_close()
        manager.close()
        thread.join(5)


def test_optional_isaac_observations_do_not_advertise_snapshots(tmp_path):
    from scene_factory.backends.isaac import IsaacSimBackend
    from scene_factory.session_contracts import backend_capabilities

    backend = IsaacSimBackend(tmp_path / "not_loaded.usd", enable_general_observation=True)
    capabilities = backend.observation_capabilities
    assert capabilities["general_observation_status"] == "pending_initialization"
    assert not capabilities["full_object_motion"]
    assert capabilities["snapshot_levels"] == backend_capabilities("isaac")["snapshot_levels"] == []
    backend._general_observation_error = "stage has no supported sensor"
    assert backend.observation_capabilities["general_observation_status"] == "failed"


def test_contact_adapter_rejects_missing_measurements_and_avoids_reverse_double_counting():
    from scene_factory.isaac_observation import IsaacObservationAdapter

    class Contact:
        def __init__(self, matrix):
            self.matrix = matrix

        def get_contact_force_matrix(self, dt):
            return self.matrix

        def get_contact_force_data(self, dt):
            import numpy as np

            matrix = np.asarray(self.matrix)
            if matrix.ndim != 3:
                return None
            counts = (np.linalg.norm(matrix, axis=2) > 0).astype(np.int32)
            starts = np.zeros_like(counts)
            force = np.zeros((8, 1))
            index = 0
            for row, column in zip(*np.nonzero(counts), strict=True):
                starts[row, column] = index
                force[index, 0] = np.linalg.norm(matrix[row, column])
                index += 1
            return force, np.zeros((8, 3)), np.zeros((8, 3)), np.zeros((8, 1)), counts, starts

    adapter = IsaacObservationAdapter.__new__(IsaacObservationAdapter)
    adapter.dt = 0.01
    adapter.colliders = ["/a", "/b"]
    adapter.owner_by_collider = {"/a": "a", "/b": "b"}
    adapter.contact_views = {"/a": ("a", Contact([[[0, 0, 0], [0, 0, 2]]])),
                             "/b": ("b", Contact([[[0, 0, -2], [0, 0, 0]]]))}
    contacts, coverage = adapter.contacts()
    assert contacts[0]["pair"] == ["a", "b"]
    assert contacts[0]["normal_force_resultant_N"] == 2
    assert contacts[0]["normal_force_N"] == 2
    assert coverage["status"] == "measured"
    adapter.contact_views = {"/a": ("a", Contact(None))}
    with pytest.raises(RuntimeError, match="invalid general contact"):
        adapter.contacts()


def test_contact_point_absolute_sum_preserves_opposing_forces_and_rejects_saturated_buffer():
    import numpy as np
    from scene_factory.isaac_observation import IsaacObservationAdapter

    class Opposing:
        capacity = 8

        def get_contact_force_matrix(self, dt):
            return np.zeros((1, 2, 3))  # Equal opposite forces cancel in the resultant.

        def get_contact_force_data(self, dt):
            forces = np.zeros((self.capacity, 1))
            forces[:2, 0] = [-3, 3]
            return (forces, np.zeros((self.capacity, 3)), np.zeros((self.capacity, 3)),
                    np.zeros((self.capacity, 1)), np.asarray([[0, 2]], dtype=np.int32),
                    np.asarray([[0, 0]], dtype=np.int32))

    view = Opposing()
    adapter = IsaacObservationAdapter.__new__(IsaacObservationAdapter)
    adapter.dt = 0.01
    adapter.colliders = ["/a", "/b"]
    adapter.owner_by_collider = {"/a": "a", "/b": "b"}
    adapter.contact_views = {"/a": ("a", view)}
    contacts, coverage = adapter.contacts()
    assert contacts[0]["normal_force_N"] == 6
    assert contacts[0]["normal_force_resultant_N"] == 0
    assert coverage["complete"] and not coverage["buffer_saturated"]
    view.capacity = 2
    with pytest.raises(RuntimeError, match="buffer saturated"):
        adapter.contacts()


def test_service_can_load_its_completed_build_outside_input_directory(tmp_path):
    import time
    from scene_factory.factory import SceneFactory
    from scene_factory.runtime_service import RuntimeService

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    scene = inputs / "layout.json"
    scene.write_text(json.dumps(SceneFactory().build_from_recipe("living_room_returned_home", 19).scene.to_dict()), encoding="utf-8")
    manager = RuntimeService(tmp_path / "service", allowed_scene_root=inputs)
    try:
        build = manager.build({"idempotency_key": "outside-build", "scene_path": str(scene), "backend": "mujoco"})
        deadline = time.monotonic() + 20
        while build["status"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.05)
            build = manager.build_status(build["build_id"])
        assert build["status"] == "completed", build
        created = manager.create({"idempotency_key": "outside-session", "backend": "mujoco", "environment_path": build["result"]["package_dir"]})
        assert created["inputs"]["asset_closure"] == "verified_package"
    finally:
        manager.close()
