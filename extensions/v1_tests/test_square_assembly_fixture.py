from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.build_square_assembly_fixture import square_physics


def test_square_inertia_and_nominal_clearance_change_without_changing_mass():
    physics = square_physics()
    assert physics["dimensions_m"] == [.02, .02, .1]
    assert physics["mass_kg"] == .08
    assert physics["inertia_kg_m2"] == pytest.approx([6.933333333333334e-5, 6.933333333333334e-5, 5.3333333333333345e-6])
    assert physics["density_kg_m3"] == pytest.approx(2000)
    assert physics["nominal_face_clearance_m"] == pytest.approx(.00075)
    assert 0 < physics["yaw_only_clearance_at_frozen_angular_tolerance_m"] < physics["nominal_face_clearance_m"]


@pytest.fixture
def usd_python():
    configured = os.environ.get("SCENE_FACTORY_USD_PYTHON")
    if configured:
        return configured
    if importlib.util.find_spec("pxr"):
        return sys.executable
    pytest.skip("Set SCENE_FACTORY_USD_PYTHON to run actual OpenUSD build/inspection tests.")


def test_actual_usd_variant_preserves_other_prims_and_spec_and_authors_new_inertia(tmp_path, usd_python):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/"square"
    subprocess.run([usd_python, str(root/"tools/build_square_assembly_fixture.py"), str(output)], check=True, capture_output=True, text=True)
    report = json.loads((output/"fixture_report.json").read_text(encoding="utf-8"))
    source = json.loads((output/"SOURCE.json").read_text(encoding="utf-8"))
    assert report["acceptance"] == "not_verified"
    assert report["primitive_type"] == "Cube"
    assert report["spec_unchanged"] and report["non_peg_prims_unchanged"]
    assert (output/"assembly_spec.json").read_bytes() == (root/"examples/assembly/peg_socket/assembly_spec.json").read_bytes()
    assert source["usd_sha256"] == hashlib.sha256((output/"scene.usda").read_bytes()).hexdigest()
    assert source["generator_sha256"] == hashlib.sha256((root/"tools/build_square_assembly_fixture.py").read_bytes()).hexdigest()
    inspect = """import json,sys
from pxr import Usd,UsdGeom,UsdPhysics
s=Usd.Stage.Open(sys.argv[1]); p=s.GetPrimAtPath('/World/Objects/peg_1'); m=UsdPhysics.MassAPI(p)
print(json.dumps({'type':p.GetTypeName(),'mass':m.GetMassAttr().Get(),'inertia':list(m.GetDiagonalInertiaAttr().Get()),
 'ops':[(str(o.GetOpName()),list(o.Get())) for o in UsdGeom.Xformable(p).GetOrderedXformOps()],
 'contact':p.GetAttribute('physxCollision:contactOffset').Get(), 'collision':p.HasAPI(UsdPhysics.CollisionAPI),
 'dynamic':p.HasAPI(UsdPhysics.RigidBodyAPI),'external_refs':any(p.HasAuthoredReferences() for p in s.Traverse())}))
"""
    result = subprocess.run([usd_python, "-c", inspect, str(output/"scene.usda")], check=True, capture_output=True, text=True)
    actual = json.loads(result.stdout)
    assert actual["type"] == "Cube"
    assert actual["mass"] == pytest.approx(.08)
    assert actual["inertia"] == pytest.approx(square_physics()["inertia_kg_m2"])
    assert actual["ops"][0][1] == pytest.approx([.55, -.13, .97])
    assert actual["ops"][1][1] == pytest.approx([.02, .02, .1])
    assert actual["contact"] == pytest.approx(.0001)
    assert actual["collision"] and actual["dynamic"] and not actual["external_refs"]
    # Re-running into an occupied destination must preserve its existing evidence.
    before = (output/"scene.usda").read_bytes()
    rerun = subprocess.run([usd_python, str(root/"tools/build_square_assembly_fixture.py"), str(output)], capture_output=True, text=True)
    assert rerun.returncode != 0
    assert (output/"scene.usda").read_bytes() == before


def test_square_room_extension_keeps_backgrounds_and_default_cylinder_geometry(tmp_path, usd_python):
    root = Path(__file__).resolve().parents[1]
    builder = root/"tools/build_assembly_room_fixture.py"
    outputs = {}
    for shape in ("cylinder", "square"):
        output = tmp_path/shape
        command = [usd_python, str(builder), str(output), "--objects", "20", "--seed", "19"]
        if shape == "square":
            command += ["--peg-shape", "square"]
        subprocess.run(command, check=True, capture_output=True, text=True)
        outputs[shape] = output
    cylinder_source = json.loads((outputs["cylinder"]/"SOURCE.json").read_text(encoding="utf-8"))
    square_source = json.loads((outputs["square"]/"SOURCE.json").read_text(encoding="utf-8"))
    frozen_cylinder = json.loads((root/"examples/assembly/room_assembly/room20/SOURCE.json").read_text(encoding="utf-8"))
    assert cylinder_source["usd_sha256"] == frozen_cylinder["usd_sha256"]
    assert square_source["backgrounds"] == cylinder_source["backgrounds"] == frozen_cylinder["backgrounds"]
    assert square_source["geometry"]["peg_shape"] == "square_prism"
    assert square_source["fixture_id"] == "original_square_peg_socket_room20_seed19"
    assert square_source["original_workspace"]["unchanged_after_extension"]
    assert (outputs["square"]/"assembly_spec.json").read_bytes() == (outputs["cylinder"]/"assembly_spec.json").read_bytes()
