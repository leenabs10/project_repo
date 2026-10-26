"""Verify real exposure, balanced gradients and matched replay/scene execution."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

from src.adapters.storage.local import file_hash
from src.pipeline.scene_benchmark import compare_arms, run_arm
from src.training.auxiliary_inputs import (
    balanced_auxiliary_loss,
    prepare_auxiliary_donors,
    select_auxiliary_donor,
    validate_auxiliary_settings,
)
from src.validation.artifacts import write_json
import tests.test_scene_training as fixture_support


class AuxiliaryLossTests(unittest.TestCase):
    def test_background_replication_does_not_dilute_foreground_loss(self):
        first = torch.tensor([[[[2.0, 1.0]], [[0.0, 0.0]]]], requires_grad=True)
        target = torch.tensor([[[1, 0]]])
        second = torch.cat((first, first[:, :, :, 1:].expand(-1, -1, -1, 9)), dim=3)
        second_target = torch.cat(
            (target, torch.zeros((1, 1, 9), dtype=torch.long)), dim=2
        )
        self.assertAlmostEqual(
            float(balanced_auxiliary_loss(first, target, 0.15).detach()),
            float(balanced_auxiliary_loss(second, second_target, 0.15).detach()),
            places=6,
        )
        loss = balanced_auxiliary_loss(first, target, 0.15)
        loss.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertNotAlmostEqual(
            float(torch.nn.functional.cross_entropy(first.detach(), target)),
            float(torch.nn.functional.cross_entropy(second.detach(), second_target)),
            places=4,
        )

    def test_invalid_policies_and_missing_classes_are_rejected(self):
        settings = dict(
            strategy="preserve_real_exposure",
            interval_updates=2,
            loss_weight=0.2,
            foreground_fraction=0.15,
        )
        for field, value in (
            ("loss_weight", 0.9),
            ("interval_updates", True),
            ("foreground_fraction", float("nan")),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_auxiliary_settings(dict(settings, **{field: value}))
        with self.assertRaises(ValueError):
            balanced_auxiliary_loss(
                torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long), 0.15
            )


class PreservedExposureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_support.SceneTrainingTests.setUpClass()
        cls.data = fixture_support.SceneTrainingTests

    @classmethod
    def tearDownClass(cls):
        fixture_support.SceneTrainingTests.tearDownClass()

    def test_three_seed_replay_and_scene_arms_preserve_every_real_slot(self):
        data = self.data
        foreground = background = 0
        for row in data.real_rows:
            if row["split"] == "train":
                with Image.open(data.references / row["target_path"]) as image:
                    values = np.array(
                        image.resize(
                            tuple(data.config["training_resolution"]),
                            Image.Resampling.NEAREST,
                        )
                    )
                foreground += int((values == 1).sum())
                background += int((values == 0).sum())
        configs = []
        for kind in ("real_only", "real_plus_scene"):
            config = copy.deepcopy(data.config)
            config.update(
                experiment_kind=kind,
                model_seeds=[17, 29, 43],
                selection_metric=None,
                auxiliary_training=dict(
                    strategy="preserve_real_exposure",
                    interval_updates=2,
                    loss_weight=0.2,
                    foreground_fraction=foreground / (foreground + background),
                ),
            )
            if kind == "real_only":
                config.pop("synthetic_mixing")
            else:
                config["synthetic_mixing"] = {"strategy": "auxiliary_only"}
            path = data.base / (kind + "_auxiliary.json")
            write_json(path, config)
            configs.append(config)
            root = data.base / (kind + "_auxiliary")
            run_arm(
                data.references,
                data.reference_sha,
                data.task,
                path,
                None,
                root,
                synthetic=data.admission if kind != "real_only" else None,
                synthetic_sha256=(
                    file_hash(data.admission / "profile.json")
                    if kind != "real_only"
                    else None
                ),
                device="cpu",
                fixture=True,
            )
            for seed in (17, 29, 43):
                profile = json.loads((root / f"seed_{seed}/profile.json").read_text())
                sampling = json.loads((root / f"seed_{seed}/sampling.json").read_text())
                self.assertEqual(profile["primary_real_presentations"], 8)
                self.assertEqual(profile["auxiliary_presentations"], 1)
                self.assertEqual(profile["total_training_presentations"], 9)
                self.assertEqual(profile["optimizer_updates"], 2)
                self.assertEqual(set(sampling.values()), {2})
                self.assertTrue(
                    all(identity.startswith("fixture_train/") for identity in sampling)
                )
        control, scene = (
            data.base / (kind + "_auxiliary")
            for kind in ("real_only", "real_plus_scene")
        )
        report = compare_arms(
            control,
            file_hash(control / "profile.json"),
            scene,
            file_hash(scene / "profile.json"),
            data.references,
            data.reference_sha,
            data.base / "auxiliary_comparison",
            fixture=True,
        )
        self.assertTrue(report["matched_comparison_complete"])
        self.assertFalse(report["candidate_promoted"])
        wrong = copy.deepcopy(configs[0])
        wrong["auxiliary_training"]["foreground_fraction"] = 0.99
        with self.assertRaisesRegex(ValueError, "foreground balance"):
            prepare_auxiliary_donors(data.references, data.real_rows, None, [], wrong)
