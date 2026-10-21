"""Verify corrective optimizer rates without changing references or update budgets."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import torch

from src.training.goods_benchmark import build_optimizer, validate_configuration


class DifferentialRateTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (
                Path(__file__).parents[1]
                / "configs/benchmarks/warehouse_goods_real_only.json"
            ).read_text()
        )
        self.model = torch.nn.Module()
        self.model.backbone = torch.nn.Linear(1, 1, bias=False)
        self.model.classifier = torch.nn.Linear(1, 1, bias=False)

    def test_original_configuration_preserves_one_flat_rate_group(self):
        optimizer = build_optimizer(self.model, self.config)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(
            optimizer.param_groups[0]["lr"], self.config["optimizer"]["learning_rate"]
        )

    def test_parameter_groups_cover_each_parameter_once(self):
        self.config["optimizer"]["backbone_learning_rate_multiplier"] = 0.1
        validate_configuration(self.config)
        optimizer = build_optimizer(self.model, self.config)
        groups = {group["name"]: group for group in optimizer.param_groups}
        identifiers = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(
            set(identifiers), {id(parameter) for parameter in self.model.parameters()}
        )
        self.assertAlmostEqual(groups["backbone"]["lr"], 0.00003)
        self.assertAlmostEqual(groups["head"]["lr"], 0.0003)

    def test_actual_update_retains_lower_backbone_step(self):
        self.config["optimizer"].update(
            backbone_learning_rate_multiplier=0.1, weight_decay=0
        )
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.fill_(1)
        optimizer = build_optimizer(self.model, self.config)
        loss = sum(parameter.sum() for parameter in self.model.parameters())
        loss.backward()
        optimizer.step()
        backbone_step = 1 - self.model.backbone.weight.item()
        head_step = 1 - self.model.classifier.weight.item()
        self.assertAlmostEqual(head_step / backbone_step, 10, delta=0.03)

    def test_invalid_rate_multipliers_are_rejected(self):
        for value in [0, -1, 2, True, float("nan"), "0.1"]:
            with self.subTest(value=value):
                config = deepcopy(self.config)
                config["optimizer"]["backbone_learning_rate_multiplier"] = value
                with self.assertRaisesRegex(ValueError, "multiplier"):
                    validate_configuration(config)


if __name__ == "__main__":
    unittest.main()
