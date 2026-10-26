"""Balanced donor cycles preserve cadence, legacy behavior and model randomness."""

from collections import Counter
import copy
import json
from pathlib import Path
import random
import unittest

import numpy as np
from PIL import Image
import torch

from src.adapters.storage.local import file_hash
from src.pipeline.scene_benchmark import compare_arms, run_arm
from src.training.auxiliary_inputs import (
    select_auxiliary_donor,
    validate_auxiliary_sampling,
)
from src.validation.artifacts import write_json
import tests.test_scene_training as fixture_support


def sampling_config():
    return dict(
        generation_seed=7331,
        experiment_kind="real_plus_scene",
        synthetic_mixing={"strategy": "auxiliary_only"},
        auxiliary_sampling={"strategy": "balanced_scene_cycle"},
        auxiliary_training=dict(
            strategy="preserve_real_exposure",
            interval_updates=2,
            loss_weight=0.2,
            foreground_fraction=0.15,
        ),
    )


def donors():
    return [
        (Path("source"), dict(sample_id=f"scene/{index}", split="train"))
        for index in range(8)
    ]


class BalancedSamplingTests(unittest.TestCase):
    def test_every_prefix_is_balanced_and_input_order_does_not_change_selection(self):
        config, inputs = sampling_config(), donors()
        counts = Counter()
        sequence = []
        for update in range(1, 121):
            selected = select_auxiliary_donor(inputs, config, update)
            reversed_selected = select_auxiliary_donor(
                list(reversed(inputs)), config, update
            )
            if update % 2:
                self.assertIsNone(selected)
                self.assertIsNone(reversed_selected)
                continue
            self.assertEqual(selected, reversed_selected)
            identity = selected[1]["sample_id"]
            sequence.append(identity)
            counts[identity] += 1
            all_counts = [counts[row["sample_id"]] for _, row in inputs]
            self.assertLessEqual(max(all_counts) - min(all_counts), 1)
        self.assertEqual(sum(counts.values()), 60)
        self.assertEqual(sorted(counts.values()), [7, 7, 7, 7, 8, 8, 8, 8])
        for index in range(0, 56, 8):
            self.assertEqual(
                set(sequence[index : index + 8]),
                {row["sample_id"] for _, row in inputs},
            )
        different = copy.deepcopy(config)
        different["generation_seed"] += 1
        self.assertNotEqual(
            sequence,
            [
                select_auxiliary_donor(inputs, different, update)[1]["sample_id"]
                for update in range(2, 121, 2)
            ],
        )

    def test_legacy_golden_sequence_and_random_states_are_preserved(self):
        config, inputs = sampling_config(), donors()
        config.pop("auxiliary_sampling")
        actual = [
            int(
                select_auxiliary_donor(inputs, config, update)[1]["sample_id"].split(
                    "/"
                )[-1]
            )
            for update in range(2, 26, 2)
        ]
        self.assertEqual(actual, [2, 5, 4, 1, 2, 5, 5, 0, 4, 7, 6, 6])
        python_state, numpy_state, torch_state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        for policy in (config, sampling_config()):
            for update in range(1, 121):
                select_auxiliary_donor(inputs, policy, update)
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(np.random.get_state()[0], numpy_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_malformed_policies_and_donors_fail_closed(self):
        config = sampling_config()
        for field, value in (
            ("experiment_kind", "real_only"),
            ("auxiliary_training", None),
            ("auxiliary_sampling", {"strategy": "unknown"}),
            ("auxiliary_sampling", {"strategy": "balanced_scene_cycle", "weight": 2}),
            ("generation_seed", True),
            ("synthetic_mixing", None),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_auxiliary_sampling(dict(config, **{field: value}))
        for inputs in (
            donors() + [donors()[0]],
            [(Path("source"), dict(sample_id="scene/0", split="validation"))],
            [(Path("source"), dict(sample_id="", split="train"))],
        ):
            with self.subTest(), self.assertRaisesRegex(
                ValueError, "unique training-only"
            ):
                select_auxiliary_donor(inputs, config, 2)
        for update in (0, -1, True, 2.5):
            with self.subTest(update=update), self.assertRaises(ValueError):
                select_auxiliary_donor(donors(), config, update)


class BalancedSamplingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_support.SceneTrainingTests.setUpClass()
        cls.data = fixture_support.SceneTrainingTests

    @classmethod
    def tearDownClass(cls):
        fixture_support.SceneTrainingTests.tearDownClass()

    def test_fixed_final_fixture_accepts_only_the_declared_sampling_difference(self):
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
        roots = []
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
                config["auxiliary_sampling"] = {"strategy": "balanced_scene_cycle"}
            configuration = data.base / (kind + "_balanced.json")
            write_json(configuration, config)
            output = data.base / (kind + "_balanced")
            run_arm(
                data.references,
                data.reference_sha,
                data.task,
                configuration,
                None,
                output,
                synthetic=data.admission if kind != "real_only" else None,
                synthetic_sha256=(
                    file_hash(data.admission / "profile.json")
                    if kind != "real_only"
                    else None
                ),
                device="cpu",
                fixture=True,
            )
            roots.append(output)
            for seed in (17, 29, 43):
                profile = json.loads((output / f"seed_{seed}/profile.json").read_text())
                self.assertEqual(profile["primary_real_presentations"], 8)
                self.assertEqual(profile["auxiliary_presentations"], 1)
                self.assertEqual(profile["optimizer_updates"], 2)
        report = compare_arms(
            roots[0],
            file_hash(roots[0] / "profile.json"),
            roots[1],
            file_hash(roots[1] / "profile.json"),
            data.references,
            data.reference_sha,
            data.base / "balanced_comparison",
            fixture=True,
        )
        self.assertTrue(report["matched_comparison_complete"])
        self.assertEqual(
            report["donor_sampling"]["control"], {"strategy": "independent_hash"}
        )
        self.assertEqual(
            report["donor_sampling"]["scene"], {"strategy": "balanced_scene_cycle"}
        )
        self.assertFalse(report["candidate_promoted"])


if __name__ == "__main__":
    unittest.main()
