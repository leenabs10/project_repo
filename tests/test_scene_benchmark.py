"""Run the complete three-seed comparison on known fixtures, not customer data."""

import copy
import json
from pathlib import Path
import unittest

from src.adapters.storage.local import file_hash
from src.pipeline.scene_benchmark import compare_arms, run_arm
from src.validation.artifacts import write_json
import tests.test_scene_training as fixture_support


class SceneComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = fixture_support.SceneTrainingTests
        cls.fixtures.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.fixtures.tearDownClass()

    def test_complete_three_seed_fixture_and_budget_mismatch_rejection(self):
        data = self.fixtures
        configs = []
        for kind in ("real_only", "real_plus_scene"):
            config = copy.deepcopy(data.config)
            config.update(experiment_kind=kind, model_seeds=[17, 29, 43])
            if kind == "real_only":
                config.pop("synthetic_mixing")
                config["selection_metric"] = "validation_goods_iou"
            else:
                config["selection_metric"] = None
            path = data.base / f"{kind}.json"
            write_json(path, config)
            configs.append(path)
            run_arm(
                data.references,
                data.reference_sha,
                data.task,
                path,
                None,
                data.base / kind,
                synthetic=data.admission if kind == "real_plus_scene" else None,
                synthetic_sha256=(
                    file_hash(data.admission / "profile.json")
                    if kind == "real_plus_scene"
                    else None
                ),
                device="cpu",
                fixture=True,
            )
        control, scene = data.base / "real_only", data.base / "real_plus_scene"
        report = compare_arms(
            control,
            file_hash(control / "profile.json"),
            scene,
            file_hash(scene / "profile.json"),
            data.references,
            data.reference_sha,
            data.base / "comparison",
            fixture=True,
        )
        self.assertTrue(report["matched_comparison_complete"])
        self.assertFalse(report["scene_utility_measured"])
        self.assertFalse(report["held_out_test_evaluated"])
        self.assertEqual(report["model_seeds"], [17, 29, 43])
        results = json.loads((scene / "seed_results.json").read_text())
        self.assertTrue(
            all(
                row["optimizer_updates"] == 2 and row["presentations"] == 8
                for row in results
            )
        )
        wrong_config = json.loads(configs[1].read_text())
        wrong_config["epochs"] += 1
        write_json(configs[1], wrong_config)
        # A changed external config cannot mutate the frozen arm copy. Altering
        # the saved copy is caught by its snapshot checksum before comparison.
        write_json(scene / "configuration.json", wrong_config)
        output = data.base / "mismatched_comparison"
        with self.assertRaises(ValueError):
            compare_arms(
                control,
                file_hash(control / "profile.json"),
                scene,
                file_hash(scene / "profile.json"),
                data.references,
                data.reference_sha,
                output,
                fixture=True,
            )
        self.assertFalse(output.exists())
