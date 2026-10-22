"""Exercise admission isolation, unchanged targets and matched training budgets."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from src.adapters.storage.local import file_hash
from src.training.goods_benchmark import (
    evaluate_run,
    fixture_run,
    load_references,
    train_run,
    validate_configuration,
)
from src.training.hybrid_inputs import (
    load_synthetic_inputs,
    select_training_input,
    validate_mixing,
)
from src.validation.artifacts import artifact_hashes, verify_snapshot, write_json


class GoodsHybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.fixture = cls.base / "fixture"
        fixture_run(cls.fixture)
        cls.references = cls.fixture / "references"
        profile = json.loads((cls.references / "profile.json").read_text())
        profile["manifest_sha256"] = "1" * 64
        write_json(cls.references / "profile.json", profile)
        cls.reference_sha = file_hash(cls.references / "profile.json")
        cls.task = cls.fixture / "task.json"
        cls.rows = load_references(
            cls.references, cls.reference_sha, cls.task, fixture=True
        )
        cls.config = json.loads((cls.fixture / "config.json").read_text())
        cls.condition = {"condition_id": "dimmer", "method": "rgb_gain", "gain": 0.93}
        cls.config.update(
            experiment_kind="real_plus_synthetic",
            epochs=2,
            synthetic_mixing={
                "strategy": "replace_source_input",
                "substitution_probability": 1,
                "conditions": [cls.condition],
            },
        )
        cls.admission = cls.base / "admission"
        cls.admission.mkdir()
        source = cls.rows[0]
        with Image.open(cls.references / source["rgb_path"]) as image:
            rgb = np.array(image)
        rgb = np.rint(np.clip(rgb.astype(np.float32) * 0.93, 0, 255)).astype(np.uint8)
        Image.fromarray(rgb).save(cls.admission / "condition.png")
        shutil.copyfile(
            cls.references / source["target_path"], cls.admission / "target.png"
        )
        cls.entry = {
            "source_sample_id": source["sample_id"],
            "source_rgb_sha256": source["source_rgb_sha256"],
            "condition": cls.condition,
            "generation_seed": 42,
            "task_id": source["task_id"],
            "target_path": "target.png",
            "target_sha256": source["target_sha256"],
            "target_origin": source["reference_origin"],
            "training_eligible": True,
            "record": {
                "sample_id": "fixture-condition",
                "split": "train",
                "group_id": source["group_id"],
                "image_size": source["image_size"],
                "modalities": {
                    "rgb": {
                        "status": "available",
                        "path": "condition.png",
                        "sha256": file_hash(cls.admission / "condition.png"),
                    }
                },
            },
        }
        (cls.admission / "eligible_manifest.jsonl").write_text(
            json.dumps(cls.entry) + "\n"
        )
        write_json(
            cls.admission / "profile.json",
            {
                "status": "passed",
                "purpose": "known_fixture_only",
                "qualification_status": "accepted",
                "training_eligible": True,
                "reviewer": "Known fixture",
                "reference_profile_sha256": cls.reference_sha,
                "protocol_sha256": file_hash(cls.task),
                "task_id": source["task_id"],
                "manifest_sha256": "1" * 64,
                "eligible_variants": 1,
                "artifact_sha256": artifact_hashes(cls.admission),
            },
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def load(self, root=None, config=None, fixture=True):
        root = root or self.admission
        return load_synthetic_inputs(
            root,
            file_hash(root / "profile.json"),
            self.references,
            self.reference_sha,
            self.rows,
            config or self.config,
            fixture=fixture,
        )

    def changed(self, name, mutate_entry=None, mutate_profile=None, duplicate=False):
        root = self.base / name
        shutil.copytree(self.admission, root)
        entry = deepcopy(self.entry)
        if mutate_entry:
            mutate_entry(entry)
        rows = [entry, entry] if duplicate else [entry]
        (root / "eligible_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        profile = json.loads((root / "profile.json").read_text())
        profile["artifact_sha256"] = artifact_hashes(root)
        if mutate_profile:
            mutate_profile(profile)
        write_json(root / "profile.json", profile)
        return root

    def test_actual_hybrid_train_reload_and_separate_native_evaluation(self):
        config_path = self.base / "hybrid.json"
        write_json(config_path, self.config)
        output = self.base / "hybrid_run"
        result = train_run(
            self.references,
            self.reference_sha,
            self.task,
            config_path,
            None,
            output,
            17,
            "cpu",
            fixture=True,
            synthetic=self.admission,
            synthetic_sha=file_hash(self.admission / "profile.json"),
        )
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertEqual(result["total_training_presentations"], 8)
        self.assertEqual(result["synthetic_training_frames"], 1)
        self.assertEqual(result["synthetic_presentations"], 2)
        self.assertFalse(result["synthetic_data_utility_measured"])
        sampling = json.loads((output / "sampling.json").read_text())
        self.assertEqual(set(sampling.values()), {2})
        inputs = json.loads((output / "input_sampling.json").read_text())
        self.assertEqual(inputs["fixture-condition"], 2)
        self.assertNotIn(self.rows[0]["sample_id"], inputs)
        evaluate_run(
            output,
            file_hash(output / "profile.json"),
            self.references,
            self.reference_sha,
            self.base / "hybrid_eval",
            "cpu",
            fixture=True,
        )
        self.assertEqual(
            json.loads((output / "validation_metrics.json").read_text()),
            json.loads((self.base / "hybrid_eval/metrics.json").read_text()),
        )

    def test_pending_admission_is_rejected_before_output_or_model_allocation(self):
        root = self.changed(
            "pending",
            mutate_profile=lambda row: row.update(
                qualification_status="acceptance_in_progress", training_eligible=False
            ),
        )
        config_path = self.base / "pending_config.json"
        write_json(config_path, self.config)
        output = self.base / "must_not_exist"
        with self.assertRaisesRegex(ValueError, "admission"):
            train_run(
                self.references,
                self.reference_sha,
                self.task,
                config_path,
                None,
                output,
                17,
                "cpu",
                fixture=True,
                synthetic=root,
                synthetic_sha=file_hash(root / "profile.json"),
            )
        self.assertFalse(output.exists())

    def test_fixture_cannot_enter_customer_benchmark(self):
        with self.assertRaisesRegex(ValueError, "admission"):
            self.load(fixture=False)

    def test_stale_reference_version_and_manifest_are_rejected(self):
        for field in ("reference_profile_sha256", "manifest_sha256", "protocol_sha256"):
            with self.subTest(field=field):
                root = self.changed(
                    field, mutate_profile=lambda row: row.update({field: "0" * 64})
                )
                with self.assertRaisesRegex(ValueError, "version mismatch"):
                    self.load(root)

    def test_development_sources_and_group_mismatches_are_rejected(self):
        mutations = [
            lambda row: row.update(source_sample_id=self.rows[-1]["sample_id"]),
            lambda row: row["record"].update(split="validation"),
            lambda row: row["record"].update(group_id="unrelated"),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "training source/group"):
                    self.load(self.changed(f"split_{index}", mutate_entry=mutation))

    def test_wrong_target_and_source_rgb_cannot_rebind_lineage(self):
        for field in ("target_sha256", "source_rgb_sha256"):
            with self.subTest(field=field):
                root = self.changed(
                    field + "_row",
                    mutate_entry=lambda row: row.update({field: "0" * 64}),
                )
                with self.assertRaisesRegex(ValueError, "binding mismatch"):
                    self.load(root)

    def test_duplicate_variants_and_undeclared_conditions_are_rejected(self):
        root = self.changed(
            "duplicate",
            duplicate=True,
            mutate_profile=lambda row: row.update(eligible_variants=2),
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load(root)
        root = self.changed(
            "undeclared",
            mutate_entry=lambda row: row.update(
                condition={"method": "depth_mask_transfer"}
            ),
        )
        with self.assertRaisesRegex(ValueError, "undeclared"):
            self.load(root)

    def test_rehashed_snapshot_cannot_hide_wrong_native_pixels(self):
        root = self.changed("wrong_pixels")
        source = self.references / self.rows[0]["rgb_path"]
        shutil.copyfile(source, root / "condition.png")
        entries = [deepcopy(self.entry)]
        entries[0]["record"]["modalities"]["rgb"]["sha256"] = file_hash(
            root / "condition.png"
        )
        (root / "eligible_manifest.jsonl").write_text(json.dumps(entries[0]) + "\n")
        profile = json.loads((root / "profile.json").read_text())
        profile["artifact_sha256"] = artifact_hashes(root)
        write_json(root / "profile.json", profile)
        with self.assertRaisesRegex(ValueError, "pixels"):
            self.load(root)

    def test_paths_cannot_escape_admission_snapshot(self):
        root = self.changed(
            "escape",
            mutate_entry=lambda row: row["record"]["modalities"]["rgb"].update(
                path="../condition.png"
            ),
        )
        with self.assertRaisesRegex(ValueError, "contained"):
            self.load(root)

    def test_deterministic_source_slots_do_not_advance_torch_rng(self):
        alternatives = self.load()
        config = deepcopy(self.config)
        config["synthetic_mixing"]["substitution_probability"] = 0.5
        state = torch.random.get_rng_state().clone()
        first = [
            select_training_input(
                self.references, self.rows[0], alternatives, config, epoch
            )[1]["sample_id"]
            for epoch in range(100)
        ]
        second = [
            select_training_input(
                self.references, self.rows[0], alternatives, config, epoch
            )[1]["sample_id"]
            for epoch in range(100)
        ]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {self.rows[0]["sample_id"], "fixture-condition"})
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        root, row, changed = select_training_input(
            self.references, self.rows[1], alternatives, config, 0
        )
        self.assertFalse(changed)
        self.assertEqual(row, self.rows[1])
        with self.assertRaisesRegex(ValueError, "training-only"):
            select_training_input(
                self.references, self.rows[-1], alternatives, config, 0
            )

    def test_real_only_rejects_synthetic_settings_and_partial_cli_inputs(self):
        config = deepcopy(self.config)
        config["experiment_kind"] = "real_only"
        with self.assertRaisesRegex(ValueError, "Real-only"):
            validate_configuration(config, fixture=True)
        config.pop("synthetic_mixing")
        path = self.base / "real_only_config.json"
        write_json(path, config)
        with self.assertRaisesRegex(ValueError, "both admitted"):
            train_run(
                self.references,
                self.reference_sha,
                self.task,
                path,
                None,
                self.base / "partial",
                17,
                "cpu",
                fixture=True,
                synthetic=self.admission,
            )

    def test_invalid_mixing_and_noop_conditions_are_rejected(self):
        for value in (0, -1, True, float("nan"), 1.1):
            with self.subTest(value=value):
                settings = deepcopy(self.config["synthetic_mixing"])
                settings["substitution_probability"] = value
                with self.assertRaises(ValueError):
                    validate_mixing(settings)
        settings = deepcopy(self.config["synthetic_mixing"])
        settings["conditions"][0]["gain"] = 1
        with self.assertRaises(ValueError):
            validate_mixing(settings)


if __name__ == "__main__":
    unittest.main()
