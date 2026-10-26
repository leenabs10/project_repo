"""Training evidence and load supports constrain the new cargo coverage planner."""

import copy
import json
from pathlib import Path
import unittest

from src.synthesis.warehouse_balanced import build_balanced_job
from tests.test_warehouse_quality import planner_inputs


class BalancedSceneTests(unittest.TestCase):
    def inputs(self):
        profile, profile_sha, camera, catalog, catalog_sha, base = planner_inputs()
        for variant in ("front", "back"):
            identifier = f"polyhaven.old_tyre.{variant}.4k"
            entry = copy.deepcopy(
                catalog["assets"][f"polyhaven.cardboard_box_01.{variant}.4k"]
            )
            entry.update(
                asset_id=identifier,
                conversion=dict(original_extent_metres=[0.6, 0.6, 0.165]),
            )
            entry["mesh"]["asset_id"] = identifier
            catalog["assets"][identifier] = entry
        config = json.loads(
            (
                Path(__file__).parents[1] / "configs/warehouse_goods_balanced.json"
            ).read_text()
        )
        evidence = dict(training_support=[dict(sample_id="Aisle_CCW_Run_1/000223")])
        return [
            profile,
            profile_sha,
            camera,
            catalog,
            catalog_sha,
            base,
            config,
            evidence,
            config["training_evidence_sha256"],
        ]

    def test_cargo_is_repeatable_and_empty_equipment_stays_background(self):
        arguments = self.inputs()
        first = build_balanced_job(*arguments)
        self.assertEqual(first, build_balanced_job(*arguments))
        self.assertEqual(len(first["scenes"]), 8)
        tyre_objects = [
            obj
            for scene in first["scenes"]
            for obj in scene["objects"]
            if "old_tyre" in obj.get("mesh", {}).get("asset_id", "")
        ]
        self.assertTrue(tyre_objects)
        self.assertTrue(all(obj["class_id"] == 1 for obj in tyre_objects))
        self.assertTrue(
            all(
                obj["class_id"] == 0
                for scene in first["scenes"]
                for obj in scene["objects"]
                if obj["role"] == "equipment"
            )
        )

    def test_wrong_split_evidence_overhang_and_overlapping_stacks_fail(self):
        for kind in ("split", "hash", "overhang", "overlap"):
            with self.subTest(kind=kind):
                args = self.inputs()
                if kind == "split":
                    args[-2]["training_support"][0][
                        "sample_id"
                    ] = "Aisle_CW_Run_1/000223"
                elif kind == "hash":
                    args[-1] = "0" * 64
                elif kind == "overhang":
                    args[6]["cases"][0]["stacks"][0]["xy"] = [-0.25, -1.6]
                else:
                    args[6]["cases"][0]["stacks"][1]["xy"] = args[6]["cases"][0][
                        "stacks"
                    ][0]["xy"].copy()
                with self.assertRaises(ValueError):
                    build_balanced_job(*args)
