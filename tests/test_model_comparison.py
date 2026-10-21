"""Prevent metric tradeoffs or unmatched cohorts from masquerading as fixes."""

import copy
import unittest

import numpy as np

from src.validation.model_comparison import (
    MATCHED_CONTRACT_FIELDS,
    assess_relative_metrics,
    background_false_positive_rate,
)


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.contract = dict.fromkeys(MATCHED_CONTRACT_FIELDS, "fixture")
        self.contract.update(model_seeds=[17, 29], test_access="disabled")
        self.rows = [
            {
                "seed": seed,
                "metrics": {"goods_iou": 0.6, "precision": 0.7, "recall": 0.8},
            }
            for seed in (17, 29)
        ]

    def test_recall_tradeoff_blocks_relative_acceptance(self):
        candidate = copy.deepcopy(self.rows)
        candidate[0]["metrics"].update(goods_iou=0.7, recall=0.7)
        result = assess_relative_metrics(
            self.contract, self.contract, self.rows, candidate
        )
        self.assertFalse(result["relative_metrics_passed"])
        self.assertFalse(result["relative_metric_checks"]["recall"]["passed"])

    def test_passing_comparison_never_certifies_labels(self):
        result = assess_relative_metrics(
            self.contract, self.contract, self.rows, self.rows
        )
        self.assertTrue(result["relative_metrics_passed"])
        self.assertFalse(result["absolute_quality_certified"])
        self.assertFalse(result["model_predictions_may_be_used_as_reference_labels"])

    def test_mismatched_reference_budget_and_missing_seeds_fail(self):
        for field in (
            "reference_profile_sha256",
            "epochs_per_seed",
            "evaluation_resolution",
        ):
            with self.subTest(field=field):
                candidate = {**self.contract, field: "different"}
                with self.assertRaisesRegex(ValueError, "Unmatched"):
                    assess_relative_metrics(
                        self.contract, candidate, self.rows, self.rows
                    )
        with self.assertRaisesRegex(ValueError, "seeds"):
            assess_relative_metrics(
                self.contract, self.contract, self.rows, self.rows[:1]
            )

    def test_missing_or_nonfinite_metrics_fail_closed(self):
        for value in (None, float("nan"), True, -0.1):
            with self.subTest(value=value):
                candidate = copy.deepcopy(self.rows)
                candidate[0]["metrics"]["precision"] = value
                with self.assertRaises(ValueError):
                    assess_relative_metrics(
                        self.contract, self.contract, self.rows, candidate
                    )

    def test_roi_excludes_ignored_and_goods_pixels(self):
        target = np.array([[0, 0, 255, 1]], dtype=np.uint8)
        prediction = np.array([[0, 1, 1, 1]], dtype=np.uint8)
        self.assertEqual(background_false_positive_rate(prediction, target), 0.5)
        self.assertIsNone(
            background_false_positive_rate(prediction[:, 2:], target[:, 2:])
        )
        with self.assertRaises(ValueError):
            background_false_positive_rate(prediction.astype(float), target)
