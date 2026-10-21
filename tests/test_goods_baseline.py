"""Matched-seed gates and native prediction displays with known pixel outcomes."""

from copy import deepcopy
import unittest

import numpy as np

from src.training.goods_baseline import metric_text, summarize_seeds
from src.validation.goods_predictions import error_colours, label_overlay


class BaselineSummaryTests(unittest.TestCase):
    def setUp(self):
        self.config = {"model_seeds": [17, 29, 43]}
        self.runs = [
            {
                "seed": seed,
                "status": "passed",
                "reference_profile_sha256": "cohort",
                "synthetic_training_frames": 0,
                "optimizer_updates": 120,
                "expected_optimizer_updates": 120,
                "strict_reload": "passed",
                "real_training_frames": 24,
                "development_frames": 12,
                "initial_checkpoint_sha256": "initial",
                "configuration_sha256": "config",
                "protocol_sha256": "task",
                "selected_validation_goods_iou": score,
            }
            for seed, score in [(17, 0.2), (29, 0.4), (43, 0.6)]
        ]

    def test_all_configured_seeds_and_sample_variability(self):
        result = summarize_seeds(self.runs, self.config, "cohort")
        self.assertAlmostEqual(result["development_goods_iou_mean"], 0.4)
        self.assertAlmostEqual(result["development_goods_iou_sample_std"], 0.2)
        for rows in [self.runs[:2], self.runs + [self.runs[0]]]:
            with self.assertRaisesRegex(ValueError, "exactly once"):
                summarize_seeds(rows, self.config, "cohort")

    def test_unmatched_budget_provenance_or_nonfinite_scores_fail(self):
        for change in [
            {"optimizer_updates": 119},
            {"reference_profile_sha256": "other"},
            {"synthetic_training_frames": 1},
            {"initial_checkpoint_sha256": "different"},
            {"selected_validation_goods_iou": float("nan")},
        ]:
            with self.subTest(change=change):
                rows = deepcopy(self.runs)
                rows[0].update(change)
                with self.assertRaises(ValueError):
                    summarize_seeds(rows, self.config, "cohort")

    def test_undefined_metrics_are_not_silently_zero(self):
        self.assertEqual(metric_text(None), "undefined")
        self.assertEqual(metric_text(0), "0.0000")


class PredictionDisplayTests(unittest.TestCase):
    def test_error_palette_excludes_ignored_predictions(self):
        target = np.array([[1, 0, 1, 255, 0]], dtype=np.uint8)
        prediction = np.array([[1, 1, 0, 1, 0]], dtype=np.uint8)
        actual = error_colours(prediction, target)
        expected = [
            [[40, 200, 90], [235, 70, 210], [255, 225, 0], [255, 160, 0], [35, 35, 35]]
        ]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(prediction, [[1, 1, 0, 1, 0]])

    def test_display_is_a_copy_and_rejects_unknown_labels(self):
        rgb = np.full((1, 3, 3), 100, dtype=np.uint8)
        target = np.array([[0, 1, 255]], dtype=np.uint8)
        result = label_overlay(rgb, target)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(rgb, np.full_like(rgb, 100))
        np.testing.assert_array_equal(target, [[0, 1, 255]])
        target[0, 0] = 7
        with self.assertRaisesRegex(ValueError, "class-ID"):
            label_overlay(rgb, target)


if __name__ == "__main__":
    unittest.main()
