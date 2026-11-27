"""Tests for pure-XML SE(3)+OBB collision analysis, 3-level S-Space benchmark, and readout math."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scene_factory.cli import main as cli_main
from scene_factory.cordis import CordisContext, PluginMeta
from scene_factory.cordis.plugins.spatial_check import plugin as spatial_check_plugin
from scene_factory.spatial import (
    analyze_xml_spatial,
    cross_modal_subspace_alignment,
    evaluate_collision_methods,
    fit_spatial_box_probe,
    generate_xml_spatial_benchmark,
    predict_box_collision,
    steering_delta_for_separation,
)


def test_nested_se3_transforms_and_rotated_obb_collision() -> None:
    # Parent rotated 90 deg around Z at (1, 0, 0); child local pos=(0, 0.5, 0.2) -> world (0.5, 0, 0.2).
    xml = """<mujoco model="nested_rot">
      <compiler angle="degree"/>
      <worldbody>
        <geom type="plane" pos="0 0 0" size="3 3 0.05"/>
        <body name="parent_frame" pos="1 0 0" euler="0 0 90">
          <body name="child_box" pos="0 0.5 0.2">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1" mass="0.5"/>
          </body>
        </body>
        <body name="sibling_box" pos="0.62 0 0.2">
          <freejoint/>
          <geom type="box" size="0.1 0.1 0.1" mass="0.5"/>
        </body>
      </worldbody>
    </mujoco>"""
    analysis = analyze_xml_spatial(xml, crosscheck_mujoco=True)
    child = analysis.bodies["child_box"]
    assert child.depth == 1 and child.parent == "parent_frame"
    assert child.world_pos == pytest.approx((0.5, 0.0, 0.2), abs=1e-5)

    # Distance between centers is 0.12m, sum of half-extents is 0.20m -> 0.08m penetration!
    assert len(analysis.colliding_pairs) == 1
    pair = analysis.colliding_pairs[0]
    assert {pair.body_a, pair.body_b} == {"child_box", "sibling_box"}
    assert pair.penetration_m == pytest.approx(0.08, abs=1e-4)
    if analysis.mujoco_crosschecked and pair.mujoco_signed_distance_m is not None:
        assert pair.mujoco_signed_distance_m == pytest.approx(-0.08, abs=2e-3)


def test_generate_three_level_benchmark_with_exact_spans(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    out_dir = tmp_path / "xml_spatial_bench"
    manifest = generate_xml_spatial_benchmark(
        out_dir,
        per_level=3,
        seed=101,
        render_images=False,
        crosscheck_mujoco=True,
    )
    assert manifest["total_samples"] == 3 * 3 * 3  # 3 levels x 3 groups x 3 variants
    rows = [json.loads(line) for line in (out_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 27

    for row in rows:
        # Verify S-Space token alignment spans match exact substrings inside prompts & XML!
        q, r = row["query_object"], row["reference_object"]
        for axis in ("horizontal", "vertical", "distance", "collision"):
            p = row["prompts"][axis]
            qs, qe = p["query_span"]
            rs, re = p["reference_span"]
            assert p["text"][qs:qe] == q
            assert p["text"][rs:re] == r
        sup = row["prompts"]["support"]
        assert sup["text"][sup["query_span"][0] : sup["query_span"][1]] == q
        for name in (q, r):
            s, e = row["xml_name_spans"][name]
            assert row["xml"][s:e] == name

        # Minimal-pair labels match constructed physical intent!
        if row["variant_kind"] == "colliding":
            assert row["ground_truth"]["collision_label"] == "colliding"
            assert row["ground_truth"]["pair"]["penetration_m"] > 0.005
        elif row["variant_kind"] == "floating":
            assert row["ground_truth"]["support_label"] == "floating"
        elif row["variant_kind"] == "valid":
            assert row["ground_truth"]["collision_label"] == "separated"
            assert row["ground_truth"]["support_label"] == "supported"
            assert row["ground_truth"]["assembly_valid"] is True

        # Level 2 nested_peg must actually live at nesting depth >= 2!
        if row["level"] == "level2_nested":
            assert row["ground_truth"]["query_body"]["depth"] >= 2


def test_sspace_inner_programming_box_collision_and_steering() -> None:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    d_model = 64
    # Construct synthetic linear 6D workspace inside a 64-D activation space (like S-Space).
    q_mat, _ = np.linalg.qr(rng.normal(size=(d_model, 6)))  # [64, 6] orthonormal
    axes_3d = q_mat[:, :3].T  # [3, 64]

    def sample_pairs(n: int):
        ca = rng.uniform(-1.0, 1.0, size=(n, 3))
        ea = rng.uniform(0.05, 0.20, size=(n, 3))
        eb = rng.uniform(0.05, 0.20, size=(n, 3))
        # Half colliding, half separated with varied sizes (so center-distance alone cannot classify!).
        colliding = np.arange(n) % 2 == 0
        cb = np.empty_like(ca)
        for i in range(n):
            if colliding[i]:
                cb[i] = ca[i] + rng.uniform(-0.7, 0.7, size=3) * (ea[i] + eb[i])
            else:
                cb[i] = ca[i] + np.array([ea[i, 0] + eb[i, 0] + rng.uniform(0.05, 0.35), 0.0, 0.0])
        noise_a = rng.normal(scale=0.005, size=(n, d_model))
        noise_b = rng.normal(scale=0.005, size=(n, d_model))
        ha = np.concatenate([ca, ea], axis=-1) @ q_mat.T + noise_a
        hb = np.concatenate([cb, eb], axis=-1) @ q_mat.T + noise_b
        return ha, hb, ca, ea, cb, eb, colliding

    tr = sample_pairs(300)
    ev = sample_pairs(120)
    metrics = evaluate_collision_methods(*tr, *ev, ridge_lambda=1e-4)
    # Explicit 6D S-Space box inner-programming recovers exact box collision even across varied sizes!
    assert metrics["explicit_6d_box_accuracy"] >= 0.98
    assert metrics["center_mae_m"] < 0.01
    assert metrics["extent_mae_m"] < 0.01

    # Verify cross-modal Procrustes alignment detects rotated identical 3D subspace!
    rot3 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_xml_axes = rot3 @ axes_3d
    align = cross_modal_subspace_alignment(axes_3d, rotated_xml_axes)
    assert align["subspace_overlap_r2"] == pytest.approx(1.0, abs=1e-5)
    assert all(angle < 0.1 for angle in align["principal_angles_deg"])

    # Verify right-inverse steering moves predicted 3D coordinate by exact Delta_p!
    probe = fit_spatial_box_probe(
        np.concatenate([tr[0], tr[1]]),
        np.concatenate([tr[2], tr[4]]),
        np.concatenate([tr[3], tr[5]]),
        ridge_lambda=1e-4,
    )
    h0 = tr[0][0]
    delta_p = np.array([0.15, -0.05, 0.08])
    dh = steering_delta_for_separation(axes_3d, delta_p)
    assert axes_3d @ dh == pytest.approx(delta_p, abs=1e-6)
    pred0, _ = probe.predict(h0)
    pred1, _ = probe.predict(h0 + dh)
    assert math.dist((pred1 - pred0).tolist(), delta_p.tolist()) < 0.02
    assert predict_box_collision(h0, tr[1][0], probe)["signed_distance_m"].ndim == 0


def test_cordis_spatial_hint_plugin_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    colliding_xml = """<mujoco><worldbody>
      <geom type="plane" pos="0 0 0" size="2 2 0.05"/>
      <body name="a" pos="0 0 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>
      <body name="b" pos="0.12 0 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>
    </worldbody></mujoco>"""

    ctx = CordisContext()
    ctx.mount(
        spatial_check_plugin,
        PluginMeta(name="xml_spatial_hint", kind="tool_hook", purpose="test"),
        {"collision_tol_m": 0.004},
    )
    out = ctx.run_after_tool("write_xml", {"xml": colliding_xml}, {"ok": True}, {})
    assert any("analytical collision: a <-> b" in h for h in out.get("advisory_hints", []))
    ctx.unmount("xml_spatial_hint")
    assert ctx.run_after_tool("write_xml", {"xml": colliding_xml}, {"ok": True}, {}) == {"ok": True}

    xml_path = tmp_path / "collide.xml"
    xml_path.write_text(colliding_xml, encoding="utf-8")
    assert cli_main(["spatial-check", str(xml_path)]) == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["assembly_valid"] is False and len(parsed["colliding_pairs"]) == 1
