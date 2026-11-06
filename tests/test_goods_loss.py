"""Verify recall-oriented loss options without admitting unreviewed annotations."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional

from src.adapters.storage.local import file_hash
from src.training.goods_benchmark import (
    evaluate_run,
    fixture_run,
    segmentation_loss,
    train_run,
    validate_loss_configuration,
)


class GoodsLossTests(unittest.TestCase):
    def test_explicit_reduction_preserves_weighted_objective_and_gradients(self):
        targets = torch.tensor([[[0, 1, 255], [1, 0, 1]]])
        for weights in (None, [0.7, 2.3]):
            with self.subTest(weights=weights):
                base = torch.tensor(
                    [[[[0.2, 1.4, 30.0], [0.8, -0.4, 0.2]],
                      [[0.7, 0.3, -30.0], [-0.1, 0.5, 0.9]]]],
                    dtype=torch.float64,
                    requires_grad=True,
                )
                explicit = base.detach().clone().requires_grad_()
                settings = {"name": "cross_entropy", "ignore_index": 255}
                if weights is not None:
                    settings["class_weights"] = weights
                expected, expected_mass = segmentation_loss(base, targets, settings)
                actual, actual_mass = segmentation_loss(
                    explicit, targets,
                    {**settings, "reduction_policy": "explicit_valid_mean"},
                )
                expected.backward()
                actual.backward()
                self.assertEqual(actual_mass, expected_mass)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(explicit.grad, base.grad, rtol=1e-12, atol=1e-12)
                self.assertTrue(torch.equal(explicit.grad[:, :, 0, 2], torch.zeros((1, 2))))

    def test_explicit_reduction_rejects_empty_supervision_and_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "no supervised pixels"):
            segmentation_loss(
                torch.zeros((1, 2, 1, 1)), torch.full((1, 1, 1), 255),
                {"name": "cross_entropy", "ignore_index": 255,
                 "reduction_policy": "explicit_valid_mean"},
            )
        for policy in ("mean", None, True, 3):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                validate_loss_configuration(
                    {"name": "cross_entropy", "ignore_index": 255,
                     "reduction_policy": policy}
                )

    def test_default_is_exact_existing_cross_entropy(self):
        logits = torch.tensor([[[[0.2, 1.4]], [[0.7, 0.3]]]])
        targets = torch.tensor([[[0, 1]]])
        loss, mass = segmentation_loss(
            logits, targets, {"name": "cross_entropy", "ignore_index": 255}
        )
        self.assertTrue(
            torch.equal(
                loss, functional.cross_entropy(logits, targets, ignore_index=255)
            )
        )
        self.assertEqual(mass, 2)

    def test_goods_weight_changes_relative_gradient_and_denominator(self):
        logits = torch.zeros((1, 2, 1, 2), requires_grad=True)
        targets = torch.tensor([[[0, 1]]])
        settings = {
            "name": "cross_entropy",
            "ignore_index": 255,
            "class_weights": [1, 2],
        }
        loss, mass = segmentation_loss(logits, targets, settings)
        loss.backward()
        self.assertEqual(mass, 3)
        self.assertAlmostEqual(
            float(abs(logits.grad[0, 1, 0, 1] / logits.grad[0, 1, 0, 0])), 2
        )

    def test_ignored_logits_never_change_loss_or_receive_gradients(self):
        targets = torch.tensor([[[0, 1, 255]]])
        settings = {
            "name": "cross_entropy",
            "ignore_index": 255,
            "class_weights": [1, 2],
        }
        first = torch.zeros((1, 2, 1, 3), requires_grad=True)
        second = first.detach().clone()
        second[0, :, 0, 2] = torch.tensor([100.0, -100.0])
        second.requires_grad_()
        first_loss, _ = segmentation_loss(first, targets, settings)
        second_loss, _ = segmentation_loss(second, targets, settings)
        self.assertTrue(torch.equal(first_loss, second_loss))
        second_loss.backward()
        self.assertTrue(torch.equal(second.grad[:, :, :, 2], torch.zeros((1, 2, 1))))

    def test_invalid_weights_and_unknown_options_fail_closed(self):
        for weights in (None, [1], [1, 0], [1, True], [1, float("nan")], [1, 2, 3]):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    validate_loss_configuration(
                        {
                            "name": "cross_entropy",
                            "ignore_index": 255,
                            "class_weights": weights,
                        }
                    )
        with self.assertRaises(ValueError):
            validate_loss_configuration(
                {"name": "cross_entropy", "ignore_index": 255, "unrecorded_option": 1}
            )

    def test_all_ignored_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no supervised"):
            segmentation_loss(
                torch.zeros((1, 2, 1, 2)),
                torch.full((1, 1, 2), 255),
                {"name": "cross_entropy", "ignore_index": 255},
            )

    def test_weighted_fixture_trains_reloads_and_evaluates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            fixture_run(original)
            configuration = json.loads((original / "config.json").read_text())
            configuration["loss"]["class_weights"] = [1, 2]
            settings = root / "weighted.json"
            settings.write_text(json.dumps(configuration))
            references = original / "references"
            reference_sha = file_hash(references / "profile.json")
            run = root / "weighted_run"
            result = train_run(
                references,
                reference_sha,
                original / "task.json",
                settings,
                None,
                run,
                17,
                "cpu",
                fixture=True,
            )
            self.assertEqual(result["optimizer_updates"], 1)
            self.assertEqual(result["strict_reload"], "passed")
            evaluated = evaluate_run(
                run,
                file_hash(run / "profile.json"),
                references,
                reference_sha,
                root / "evaluation",
                "cpu",
                fixture=True,
            )
            self.assertEqual(evaluated["status"], "passed")
            self.assertEqual(
                json.loads((run / "validation_metrics.json").read_text()),
                json.loads((root / "evaluation/metrics.json").read_text()),
            )
            # The proposed real-data correction cannot run against older labels.
            configuration["required_reference_packet_sha256"] = "a" * 64
            settings.write_text(json.dumps(configuration))
            with self.assertRaisesRegex(
                ValueError, "required reviewed annotation revision"
            ):
                train_run(
                    references,
                    reference_sha,
                    original / "task.json",
                    settings,
                    None,
                    root / "stale_run",
                    17,
                    "cpu",
                    fixture=True,
                )
            self.assertFalse((root / "stale_run").exists())
