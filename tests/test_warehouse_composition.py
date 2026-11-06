"""Composition fixtures protect source isolation, metric support and load semantics."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from src.synthesis.warehouse_balanced import build_balanced_job
from src.synthesis.warehouse_composition import build_composition_job
from tests import test_balanced_scene


def inputs():
    """Use small registered-asset fixtures with the real portable layout policy."""
    arguments = test_balanced_scene.BalancedSceneTests().inputs()
    source = build_balanced_job(*arguments)
    configuration = json.loads(
        (
            Path(__file__).parents[1]
            / "archive/configs/synthesis/warehouse_rack_composition.json"
        ).read_text()
    )
    source_sha = "2" * 64
    configuration["source_job_sha256"] = source_sha
    evidence = dict(
        training_support=[dict(sample_id="training/example", split="train")],
        observations=[dict(family="stacked_tyres"), dict(family="stacked_cartons")],
    )
    return [
        source,
        source_sha,
        arguments[3],
        arguments[4],
        configuration,
        evidence,
        configuration["training_evidence_sha256"],
    ]


class WarehouseCompositionTests(unittest.TestCase):
    def test_preserves_sources_and_load_aspect_ratios(self):
        arguments = inputs()
        before = copy.deepcopy(arguments)
        result, lineage = build_composition_job(*arguments)
        self.assertEqual(arguments, before)
        self.assertEqual((result, lineage), build_composition_job(*arguments))
        self.assertEqual(len(result["scenes"]), 4)
        self.assertEqual(result["materials"], arguments[0]["materials"])
        self.assertEqual(result["render"], arguments[0]["render"])
        self.assertEqual(result["camera"], arguments[0]["camera"])
        source_scene = next(
            scene
            for scene in arguments[0]["scenes"]
            if scene["sample_id"] == arguments[4]["source_sample_id"]
        )
        for scene in result["scenes"]:
            self.assertEqual(scene["lights"], source_scene["lights"])
            objects = {obj["instance_id"]: obj for obj in scene["objects"]}
            for original in source_scene["objects"]:
                identifier = original["instance_id"]
                if identifier not in arguments[4]["remove_instance_ids"]:
                    self.assertEqual(objects[identifier], original)
                else:
                    self.assertNotIn(identifier, objects)
            self.assertTrue(
                all(
                    obj["class_id"] == int(obj["role"] == "goods")
                    for obj in scene["objects"]
                )
            )
            for binding in (
                row for row in lineage if row["sample_id"] == scene["sample_id"]
            ):
                obj = objects[binding["instance_id"]]
                extent = arguments[2]["assets"][binding["asset_id"]]["conversion"][
                    "original_extent_metres"
                ]
                np.testing.assert_allclose(
                    np.array(obj["dimensions_metres"]) / extent,
                    binding["uniform_scale"],
                    rtol=0,
                    atol=1e-12,
                )

    def test_rejects_cross_split_or_mismatched_sources(self):
        for failure in ("source", "catalog", "evidence", "split", "unsupported_family"):
            with self.subTest(failure=failure):
                arguments = inputs()
                if failure == "source":
                    arguments[1] = "3" * 64
                elif failure == "catalog":
                    arguments[3] = "3" * 64
                elif failure == "evidence":
                    arguments[-1] = "3" * 64
                elif failure == "split":
                    arguments[-2]["training_support"][0]["split"] = "validation"
                else:
                    arguments[-2]["observations"] = []
                with self.assertRaises(ValueError):
                    build_composition_job(*arguments)

    def test_rejects_invalid_support_geometry_and_identity(self):
        for failure in (
            "overhang",
            "pallet_overlap",
            "pallet_class",
            "missing_id",
            "duplicate_id",
            "unsupported_field",
            "nonfinite",
            "duplicate_case",
            "camera",
            "count",
        ):
            with self.subTest(failure=failure):
                arguments = inputs()
                configuration = arguments[4]
                if failure == "overhang":
                    configuration["load_templates"]["tyres"]["scale"] = 1.7
                    # Restrict the support width independently of load dimensions.
                    scene = next(
                        scene
                        for scene in arguments[0]["scenes"]
                        if scene["sample_id"] == configuration["source_sample_id"]
                    )
                    for obj in scene["objects"]:
                        if obj["instance_id"] in configuration["pallet_template_ids"]:
                            obj["dimensions_metres"][0] *= 0.2
                            obj["center_metres"][0] = -0.85
                elif failure == "pallet_overlap":
                    configuration["pallet_centres_xy_metres"][1] = configuration[
                        "pallet_centres_xy_metres"
                    ][0].copy()
                elif failure == "pallet_class":
                    configuration["pallet_template_ids"][0] = 454
                elif failure == "missing_id":
                    configuration["remove_instance_ids"].append(60000)
                elif failure == "duplicate_id":
                    configuration["remove_instance_ids"].append(
                        configuration["remove_instance_ids"][0]
                    )
                elif failure == "unsupported_field":
                    configuration["ignored_typo"] = True
                elif failure == "nonfinite":
                    configuration["pallet_centres_xy_metres"][0][0] = float("nan")
                elif failure == "duplicate_case":
                    configuration["cases"][1]["sample_id"] = configuration["cases"][0][
                        "sample_id"
                    ]
                elif failure == "camera":
                    configuration["cases"][0]["camera"][0] = 4
                else:
                    configuration["load_templates"]["tyres"]["count"] = True
                with self.assertRaises(ValueError):
                    build_composition_job(*arguments)

    def test_real_support_collision_fails_before_rendering(self):
        arguments = inputs()
        # Leaving the old lower shelf in place intersects the proposed tyre load.
        arguments[4]["remove_instance_ids"].remove(454)
        with self.assertRaisesRegex(ValueError, "penetrates"):
            build_composition_job(*arguments)


if __name__ == "__main__":
    unittest.main()
