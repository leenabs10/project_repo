"""Train/evaluate the actual loop on known labels; exercise real-data gate failures."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash
from src.training.goods_benchmark import (
    fixture_run,
    load_references,
    evaluate_run,
    input_tensor,
    target_tensor,
)
from src.validation.artifacts import artifact_hashes, verify_snapshot, write_json


class GoodsBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.output = cls.base / "fixture"
        cls.result = fixture_run(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_actual_train_reload_and_native_metrics(self):
        profile = verify_snapshot(self.output / "training")
        self.assertEqual(profile["optimizer_updates"], 1)
        self.assertEqual(
            profile["optimizer_updates"], profile["expected_optimizer_updates"]
        )
        self.assertEqual(profile["synthetic_training_frames"], 0)
        self.assertFalse(profile["benchmark_complete"])
        self.assertEqual(profile["strict_reload"], "passed")
        progress = json.loads((self.output / "training/progress.json").read_text())
        self.assertEqual(progress["epoch"], progress["epochs"])
        self.assertEqual(progress["optimizer_updates"], profile["optimizer_updates"])
        self.assertEqual(progress["best_epoch"], profile["best_epoch"])
        trained = json.loads(
            (self.output / "training/validation_metrics.json").read_text()
        )
        evaluated = json.loads((self.output / "evaluation/metrics.json").read_text())
        self.assertEqual(trained, evaluated)
        self.assertEqual(trained["score_resolution"], "native")
        self.assertEqual(trained["valid_pixels"], 31 * 32)
        self.assertEqual(trained["reference_coverage"], 31 / 32)

    def test_fixture_and_ai_targets_cannot_become_real_benchmarks(self):
        root = self.output / "references"
        with self.assertRaisesRegex(ValueError, "Fixture"):
            load_references(
                root, file_hash(root / "profile.json"), self.output / "task.json"
            )
        with self.assertRaisesRegex(ValueError, "purpose"):
            evaluate_run(
                self.output / "training",
                file_hash(self.output / "training/profile.json"),
                root,
                file_hash(root / "profile.json"),
                self.base / "invalid_evaluation",
                device="cpu",
            )

    def test_native_mask_resampling_and_normalization(self):
        config = json.loads((self.output / "config.json").read_text())
        row = json.loads(
            (self.output / "references/references.jsonl").read_text().splitlines()[0]
        )
        image = input_tensor(self.output / "references", row, config)
        target = target_tensor(self.output / "references", row, config)
        self.assertEqual(tuple(image.shape), (3, 16, 16))
        self.assertEqual(tuple(target.shape), (16, 16))
        self.assertTrue(set(target.unique().tolist()) <= {0, 1, 255})
        self.assertAlmostEqual(
            float(image[0, 10, 0]), (20 / 255 - 0.485) / 0.229, places=5
        )

    def changed_references(self, mutation, name):
        import shutil

        destination = self.base / name
        shutil.copytree(self.output / "references", destination)
        rows = [
            json.loads(line)
            for line in (destination / "references.jsonl").read_text().splitlines()
        ]
        mutation(rows)
        (destination / "references.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        profile = json.loads((destination / "profile.json").read_text())
        profile["artifact_sha256"] = artifact_hashes(destination)
        write_json(destination / "profile.json", profile)
        return destination

    def test_whole_capture_group_leakage_and_test_membership_are_rejected(self):
        root = self.changed_references(
            lambda rows: rows[-1].update(group_id=rows[0]["group_id"]), "leak"
        )
        with self.assertRaisesRegex(ValueError, "group"):
            load_references(
                root,
                file_hash(root / "profile.json"),
                self.output / "task.json",
                fixture=True,
            )
        root = self.changed_references(
            lambda rows: rows[-1].update(split="test"), "test"
        )
        with self.assertRaisesRegex(ValueError, "Held-out"):
            load_references(
                root,
                file_hash(root / "profile.json"),
                self.output / "task.json",
                fixture=True,
            )

    def test_recomputed_snapshot_cannot_hide_wrong_target_hash_or_ai_origin(self):
        root = self.changed_references(
            lambda rows: rows[0].update(target_sha256="0" * 64), "hash"
        )
        with self.assertRaisesRegex(ValueError, "hash"):
            load_references(
                root,
                file_hash(root / "profile.json"),
                self.output / "task.json",
                fixture=True,
            )

        root = self.changed_references(
            lambda rows: rows[0].update(reference_origin="ai_reviewed_model_assisted"),
            "ai",
        )
        with self.assertRaisesRegex(ValueError, "AI/predicted"):
            load_references(
                root,
                file_hash(root / "profile.json"),
                self.output / "task.json",
                fixture=True,
            )

    def test_human_reference_count_metadata_cannot_disagree_with_rows(self):
        def human_fixture(rows):
            for row in rows:
                row.update(
                    reference_origin="human_reviewed_model_assisted",
                    reviewer_kind="human",
                )

        root = self.changed_references(human_fixture, "human_count_disagreement")
        profile = json.loads((root / "profile.json").read_text())
        profile.pop("purpose")
        profile["qualified_reference_counts"] = {"train": 0, "validation": 0}
        write_json(root / "profile.json", profile)
        with self.assertRaisesRegex(ValueError, "Qualified reference counts"):
            load_references(
                root, file_hash(root / "profile.json"), self.output / "task.json"
            )


if __name__ == "__main__":
    unittest.main()
