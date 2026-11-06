"""Verify source isolation and truthful exposure for explicit mixed auxiliaries."""

from collections import Counter
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash
from src.training.auxiliary_inputs import (
    prepare_auxiliary_donors,
    select_auxiliary_donor,
    validate_auxiliary_sampling,
)
from src.training.goods_benchmark import input_tensor, target_tensor, train_run
from src.validation.artifacts import write_json


class MixedAuxiliarySourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.real = self.base / "references"
        self.scene = self.base / "admitted"
        self.real.mkdir()
        self.scene.mkdir()
        self.real_rows, self.scene_rows = [], []
        for root, prefix, rows in (
            (self.real, "real", self.real_rows),
            (self.scene, "scene", self.scene_rows),
        ):
            for index in range(2):
                target = np.zeros((4, 4), np.uint8)
                target[0, 0] = 255
                if index:
                    target[2:] = 1
                rgb = np.full((4, 4, 3), 20 if prefix == "real" else 190, np.uint8)
                Image.fromarray(target).save(root / f"target{index}.png")
                Image.fromarray(rgb).save(root / f"rgb{index}.png")
                rows.append(dict(sample_id=f"{prefix}/{index}", split="train",
                    target_path=f"target{index}.png", rgb_path=f"rgb{index}.png",
                    source_rgb_sha256=file_hash(root / f"rgb{index}.png"),
                    target_sha256=file_hash(root / f"target{index}.png")))
        self.config = dict(experiment_kind="real_plus_scene", generation_seed=3,
            synthetic_mixing={"strategy": "auxiliary_only"}, epochs=4, batch_size=2,
            training_resolution=[4, 4],
            normalization=dict(mean=[0, 0, 0], std=[1, 1, 1]),
            auxiliary_training=dict(strategy="preserve_real_exposure", interval_updates=2,
                loss_weight=0.2, foreground_fraction=8 / 30,
                missing_class_policy="background_only_zero_foreground_mass"),
            auxiliary_sampling=dict(strategy="explicit_mixed_source_sequence", events=[
                dict(update=2, sample_id="real/0", target_kind="background_only", input_source="real_training_reference"),
                dict(update=4, sample_id="scene/1", target_kind="mixed", input_source="admitted_scene"),
            ]))

    def prepare(self, *, config=None, real_rows=None, scenes=None, scene_root=None):
        return prepare_auxiliary_donors(self.real,
            self.real_rows if real_rows is None else real_rows,
            self.scene if scene_root is None else scene_root,
            self.scene_rows if scenes is None else scenes,
            self.config if config is None else config)

    def test_rgb_and_target_keep_their_declared_root_and_ignore_pixels(self):
        original = copy.deepcopy((self.config, self.real_rows, self.scene_rows))
        donors = self.prepare()
        real = select_auxiliary_donor(donors, self.config, 2)
        scene = select_auxiliary_donor(donors, self.config, 4)
        self.assertEqual(real[0], self.real)
        self.assertEqual(scene[0], self.scene)
        self.assertEqual(real[1]["auxiliary_input_source"], "real_training_reference")
        self.assertEqual(scene[1]["auxiliary_input_source"], "admitted_scene")
        self.assertAlmostEqual(float(input_tensor(*real, self.config)[0, 0, 0]), 20 / 255)
        self.assertAlmostEqual(float(input_tensor(*scene, self.config)[0, 0, 0]), 190 / 255)
        self.assertEqual(set(target_tensor(*real, self.config).unique().tolist()), {0, 255})
        self.assertEqual(set(target_tensor(*scene, self.config).unique().tolist()), {0, 1, 255})
        self.assertEqual((self.config, self.real_rows, self.scene_rows), original)

    def test_missing_unknown_and_mismatched_sources_fail_before_fitting(self):
        for mutation in ("missing", "unknown", "wrong", "extra", "single_source"):
            config = copy.deepcopy(self.config)
            event = config["auxiliary_sampling"]["events"][0]
            if mutation == "missing":
                event.pop("input_source")
            elif mutation == "unknown":
                event["input_source"] = "development_reference"
            elif mutation == "wrong":
                event["input_source"] = "admitted_scene"
                config["auxiliary_sampling"]["events"][1]["input_source"] = "real_training_reference"
            elif mutation == "extra":
                event["root"] = str(self.scene)
            else:
                event["input_source"] = "admitted_scene"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.prepare(config=config)
        config = copy.deepcopy(self.config)
        config["experiment_kind"] = "real_only"
        with self.assertRaises(ValueError):
            validate_auxiliary_sampling(config)

    def test_withheld_references_colliding_ids_and_path_escape_are_rejected(self):
        for mutation in ("withheld", "collision", "escape", "wrong_kind", "same_root"):
            rows, scenes = copy.deepcopy(self.real_rows), copy.deepcopy(self.scene_rows)
            config = copy.deepcopy(self.config)
            root = self.scene
            if mutation == "withheld":
                rows[0]["split"] = "validation"
            elif mutation == "collision":
                scenes[0]["sample_id"] = rows[0]["sample_id"]
            elif mutation == "escape":
                rows[0]["target_path"] = "../admitted/target0.png"
            elif mutation == "wrong_kind":
                config["auxiliary_sampling"]["events"][0]["target_kind"] = "mixed"
            else:
                root = self.real
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.prepare(config=config, real_rows=rows, scenes=scenes, scene_root=root)

    def test_native_invalid_id_cannot_disappear_during_downsampling(self):
        target = np.zeros((8, 8), np.uint8)
        target[0, 0] = 2
        Image.fromarray(target).save(self.scene / "target1.png")
        with self.assertRaisesRegex(ValueError, "uint8 categorical"):
            self.prepare()

    def test_schedule_preserves_budget_and_random_state(self):
        donors = self.prepare()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        selected = [select_auxiliary_donor(donors, self.config, update) for update in range(1, 5)]
        self.assertEqual(sum(value is not None for value in selected), 2)
        self.assertEqual(Counter(value[1]["auxiliary_input_source"] for value in selected if value),
            {"real_training_reference": 1, "admitted_scene": 1})
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        for events in (self.config["auxiliary_sampling"]["events"][:-1],
            self.config["auxiliary_sampling"]["events"] + [dict(self.config["auxiliary_sampling"]["events"][-1], update=6)]):
            invalid = copy.deepcopy(self.config)
            invalid["auxiliary_sampling"]["events"] = events
            with self.assertRaises(ValueError):
                self.prepare(config=invalid)


class MixedAuxiliaryTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_scene_training import SceneTrainingTests

        cls.data = type("MixedInputFixture", (SceneTrainingTests,), {})
        cls.data.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.data.tearDownClass()

    def test_saved_exposure_counts_real_replay_as_real_and_strictly_reloads(self):
        data = self.data
        config = copy.deepcopy(data.config)
        training = [row for row in data.real_rows if row["split"] == "train"]
        fg = bg = 0
        for row in training:
            with Image.open(data.references / row["target_path"]) as image:
                target = np.array(image.resize(tuple(config["training_resolution"]), Image.Resampling.NEAREST))
            fg += int((target == 1).sum())
            bg += int((target == 0).sum())
        rows = [json.loads(line) for line in (data.admission / "eligible_manifest.jsonl").read_text().splitlines()]
        config.update(epochs=4, model_seeds=[17], selection_metric=None,
            synthetic_mixing={"strategy": "auxiliary_only"},
            auxiliary_training=dict(strategy="preserve_real_exposure", interval_updates=2,
                loss_weight=0.2, foreground_fraction=fg / (fg + bg),
                missing_class_policy="background_only_zero_foreground_mass"),
            auxiliary_sampling=dict(strategy="explicit_mixed_source_sequence", events=[
                dict(update=2, sample_id=training[0]["sample_id"], target_kind="mixed", input_source="real_training_reference"),
                dict(update=4, sample_id=rows[0]["sample_id"], target_kind="mixed", input_source="admitted_scene"),
            ]))
        path = data.base / "mixed_auxiliary.json"
        write_json(path, config)
        output = data.base / "mixed_auxiliary_run"
        result = train_run(data.references, data.reference_sha, data.task, path, None,
            output, 17, "cpu", fixture=True, synthetic=data.admission,
            synthetic_sha=file_hash(data.admission / "profile.json"))
        self.assertEqual(result["optimizer_updates"], 4)
        self.assertEqual(result["primary_real_presentations"], 16)
        self.assertEqual(result["auxiliary_presentations"], 2)
        self.assertEqual(result["synthetic_presentations"], 1)
        self.assertEqual(result["real_presentations"], 17)
        self.assertEqual(result["total_training_presentations"], 18)
        self.assertEqual(result["auxiliary_source_presentations"],
            {"real_training_reference": 1, "admitted_scene": 1})
        self.assertEqual(result["strict_reload"], "passed")
        events = json.loads((output / "auxiliary_events.json").read_text())
        self.assertEqual([event["input_source"] for event in events],
            ["real_training_reference", "admitted_scene"])
        self.assertEqual(events[0]["target_sha256"], training[0]["target_sha256"])
        self.assertEqual(events[1]["rgb_sha256"], rows[0]["source_rgb_sha256"])


if __name__ == "__main__":
    unittest.main()
