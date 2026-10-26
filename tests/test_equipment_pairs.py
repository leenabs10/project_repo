"""Paired equipment has identical context and explicit, source-bound cargo."""

import copy
import unittest

from src.synthesis.equipment_pairs import build_equipment_pairs
from src.synthesis.warehouse_balanced import build_balanced_job
from tests import test_balanced_scene


class EquipmentPairTests(unittest.TestCase):
    def inputs(self):
        args = test_balanced_scene.BalancedSceneTests().inputs()
        source = build_balanced_job(*args)
        evidence = args[-2]
        evidence["training_support"][0]["split"] = "train"
        evidence["observations"] = [
            dict(family="stacked_tyres"),
            dict(family="stacked_cartons"),
        ]
        scene = source["scenes"][0]
        ids = [
            o["instance_id"]
            for o in scene["objects"]
            if o["role"] == "equipment"
            and o["material"] == "wood"
            and o["instance_id"] in range(830, 838)
        ]
        self.assertEqual(len(ids), 8)
        config = dict(
            schema_version="dubaidata.equipment-pairs/1.0",
            job_id="pair_fixture",
            source_job_sha256="a" * 64,
            source_sample_id=scene["sample_id"],
            training_evidence_sha256=args[-1],
            pallet_template_ids=ids,
            remove_equipment_ids=ids,
            pairs=[
                dict(
                    pair_id="close_pallet",
                    camera=[0, -3.8, 0.7, 0],
                    xy=[-0.85, -1.6],
                    pallet_layers=2,
                    cargo="old_tyre",
                    cargo_scale=1.1,
                    cargo_count=3,
                )
            ],
        )
        return [source, "a" * 64, args[3], args[4], config, evidence, args[-1]]

    def test_only_cargo_differs_between_pair_members(self):
        args = self.inputs()
        original = copy.deepcopy(args)
        job, pairs = build_equipment_pairs(*args)
        self.assertEqual(args, original)
        self.assertEqual(job["required_visible_class_ids"], [0])
        empty, loaded = job["scenes"]
        self.assertTrue(
            all(o["role"] != "goods" and o["class_id"] == 0 for o in empty["objects"])
        )
        cargo = [o for o in loaded["objects"] if o["role"] == "goods"]
        self.assertEqual(len(cargo), 3)
        loaded_without_cargo = copy.deepcopy(loaded)
        loaded_without_cargo["objects"] = [
            o for o in loaded["objects"] if o["role"] != "goods"
        ]
        loaded_without_cargo.update(
            sample_id=empty["sample_id"], scenario=empty["scenario"]
        )
        self.assertEqual(empty, loaded_without_cargo)
        self.assertEqual(
            pairs[0]["cargo_instance_ids"], [o["instance_id"] for o in cargo]
        )

    def test_wrong_sources_unsupported_semantics_and_placement_fail(self):
        for kind in (
            "hash",
            "split",
            "family",
            "count",
            "overhang",
            "support",
            "camera",
        ):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                args = self.inputs()
                if kind == "hash":
                    args[1] = "0" * 64
                elif kind == "split":
                    args[5]["training_support"][0]["split"] = "validation"
                elif kind == "family":
                    args[5]["observations"] = []
                elif kind == "count":
                    args[4]["pairs"][0]["pallet_layers"] = True
                elif kind == "overhang":
                    args[2]["assets"]["polyhaven.old_tyre.back.4k"]["conversion"][
                        "original_extent_metres"
                    ][0] = 3.0
                elif kind == "support":
                    args[4]["remove_equipment_ids"] = args[4]["remove_equipment_ids"][
                        1:
                    ]
                else:
                    args[4]["pairs"][0]["camera"] = [0, -2, 0.7, 0]
                build_equipment_pairs(*args)


if __name__ == "__main__":
    unittest.main()
