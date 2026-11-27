from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

from scene_factory import SceneFactory
from scene_factory.webapp import SceneWebApplication


class DessertDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"SCENE_FACTORY_LLM_MODE": "off"}):
            cls.factory = SceneFactory()

    def test_assets_are_bound_to_accepted_geometry(self):
        records = [asset for asset in self.factory.registry.list()
                   if "dessert_display_pack" in asset.tags]
        self.assertEqual(len(records), 4)
        for asset in records:
            with self.subTest(asset=asset.asset_id):
                self.assertEqual(asset.status, "ready")
                report = json.loads((self.factory.registry.base_dir / asset.qa_report).read_text())
                self.assertTrue(report["valid"])
                self.assertTrue(all(case["passed"] for case in report["physics_cases"]))
                for key, path in (
                    ("usd_sha256", self.factory.registry.base_dir / asset.source_path),
                    ("collision_sha256", self.factory.registry.base_dir / asset.collision_path),
                    ("glb_sha256", self.factory.registry.base_dir / "source" / asset.asset_id / "visual.glb"),
                ):
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), report[key])

    def test_fixed_three_tier_arrangement_and_web_models(self):
        result = self.factory.build_from_recipe("three_tier_dessert_display", 42)
        self.assertTrue(result.valid, result.validation.to_dict())
        self.assertEqual(len(result.scene.objects), 43)
        self.assertEqual(Counter(item.support for item in result.scene.objects), {
            "floor": 1, "stand_1:lower": 24,
            "stand_1:middle": 12, "stand_1:upper": 6,
        })
        stand = next(item for item in result.scene.objects if item.object_id == "stand_1")
        geometry = json.loads((self.factory.registry.base_dir / "source" / stand.asset_id
                               / "geometry.json").read_text())
        posts = [part for part in geometry["parts"]
                 if part["name"].startswith("FrameUpright_")]
        self.assertEqual(len(posts), 2)
        for part, expected_x in zip(posts, (-0.095, 0.095)):
            vertices = part["vertices"]
            mins = [min(vertex[i] for vertex in vertices) + stand.pose.position[i]
                    for i in range(3)]
            maxs = [max(vertex[i] for vertex in vertices) + stand.pose.position[i]
                    for i in range(3)]
            self.assertAlmostEqual((mins[0] + maxs[0]) / 2, expected_x, places=6)
            self.assertAlmostEqual((mins[1] + maxs[1]) / 2, 0, places=6)
            for cake in result.scene.objects[1:]:
                self.assertFalse(all(
                    mins[i] < cake.pose.position[i] + cake.bbox_m[i] / 2
                    and maxs[i] > cake.pose.position[i] - cake.bbox_m[i] / 2
                    for i in range(3)
                ), f"{part['name']} intersects {cake.object_id}")
        for asset_id, expected_names in (
            ("dessert_strawberry_square_01", ("CreamRosette_0", "Berry_Seed_0_0")),
            ("dessert_cream_cupcake_01", ("PleatedGoldWrapper", "VanillaSwirl")),
            ("dessert_pink_mini_cake_01", ("WhitePaperCup", "RosePetal_1_0")),
        ):
            parts = json.loads((self.factory.registry.base_dir / "source" / asset_id
                                / "geometry.json").read_text())["parts"]
            self.assertTrue(set(expected_names).issubset({part["name"] for part in parts}))
        for item in result.scene.objects:
            if item is stand:
                continue
            surface_name = item.support.split(":")[1]
            surface = next(s for s in self.factory.registry.get(stand.asset_id).support_surfaces
                           if s.name == surface_name)
            self.assertAlmostEqual(item.pose.position[2] - item.bbox_m[2] / 2,
                                   stand.pose.position[2] + surface.center[2])
        with tempfile.TemporaryDirectory() as directory:
            app = SceneWebApplication(directory, factory=self.factory)
            self.assertTrue(all(app._asset_payload(item.asset_id)["visual_url"]
                                and app._asset_payload(item.asset_id)["preserve_visual_materials"]
                                for item in result.scene.objects))
        self.assertEqual(result.scene.objects, self.factory.build_from_recipe(
            "three_tier_dessert_display", 43).scene.objects)

    def test_explicit_web_recipe_does_not_use_prompt_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            app = SceneWebApplication(directory, factory=self.factory)
            with patch.object(self.factory, "build_from_prompt", side_effect=AssertionError):
                payload = app.generate({
                    "prompt": "三层蛋糕架", "recipe": "three_tier_dessert_display",
                    "seed": 42, "export_mjcf": False,
                })
            self.assertEqual(payload["items"][0]["scene"]["recipe_name"],
                             "three_tier_dessert_display")
            self.assertEqual(payload["items"][0]["appearance"]["proxy_count"], 0)


if __name__ == "__main__":
    unittest.main()
