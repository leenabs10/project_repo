"""Controlled-generation quality tests using a non-TorWIC customer fixture."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from jsonschema.exceptions import ValidationError

from src.adapters.datasets.paired_images import PairedPngAdapter
from src.adapters.execution.local import LocalExecutor
from src.adapters.generation.sensor_conditions import (
    condition_seed,
    transform_pair,
)
from src.validation.sensor_conditions import check_variant
from src.adapters.storage.local import file_hash
from src.core.contracts import SCHEMA_VERSION, validate_job, validate_sensor_policy
from src.pipeline.sensor_conditions import generate_conditions


class SensorConditionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.raw = self.base / "raw"
        self.raw.mkdir()
        rgb = np.full((2, 4, 3), 80, np.uint8)
        depth = np.array([[0, 100, 200, 300], [400, 500, 600, 65535]], np.uint16)
        donor = depth.copy()
        donor[0, 2] = 65535
        semantic = np.array([[0, 1, 1, 2], [2, 2, 3, 0]], np.uint8)
        for name, array in (
            ("rgb", rgb),
            ("depth", depth),
            ("donor", donor),
            ("semantic", semantic),
        ):
            Image.fromarray(array).save(self.raw / (name + ".png"))
        (self.raw / "cloud.bin").write_bytes(
            b"opaque-preserved-fixture-not-simulated-lidar"
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "fixture",
            "sample_id": "A",
            "group_id": "train_capture",
            "split": "train",
            "timestamp_seconds": 1,
            "timestamp_basis": "relative",
            "image_size": [4, 2],
            "camera": {
                "sensor_id": "camera",
                "image_convention": "fixture",
                "intrinsics": [4, 4, 1.5, 0.5],
            },
            "modalities": {
                "rgb": {"status": "available", "path": "rgb.png"},
                "depth": {
                    "status": "available",
                    "path": "depth.png",
                    "units": "centimetres",
                    "scale_to_metres": 0.01,
                    "invalid_raw_value": 65535,
                },
                "semantic": {
                    "status": "available",
                    "path": "semantic.png",
                    "label_origin": "fixture_labels",
                },
                "lidar": {
                    "status": "available",
                    "path": "cloud.bin",
                    "units": "metres",
                    "frame": "fixture_frame",
                },
            },
            "source_metadata": {"pose": [1, 2, 3]},
        }
        other = copy.deepcopy(record)
        other.update(sample_id="B", timestamp_seconds=2)
        other["modalities"]["depth"]["path"] = "donor.png"
        test = copy.deepcopy(record)
        test.update(sample_id="C", group_id="test_capture", split="test")
        for reference in test["modalities"].values():
            original = self.raw / reference["path"]
            reference["path"] = "held_" + reference["path"]
            (self.raw / reference["path"]).write_bytes(original.read_bytes())
        self.records = [record, other, test]
        self.manifest = self.base / "manifest.jsonl"
        self.write_manifest()
        self.evidence = self.base / "profile.json"
        self.evidence.write_text('{"split":"train","status":"fixture"}')
        self.policy = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "fixture",
            "source_manifest_sha256": file_hash(self.manifest),
            "source_split": "train",
            "training_groups": ["train_capture"],
            "sensor_id": "camera",
            "basis": "training_distribution_bounds_not_sensor_calibration",
            "physical_sensor_realism_certified": False,
            "evidence_sha256": {"profile": file_hash(self.evidence)},
            "bounds": {
                "rgb_gain_range": [0.8, 1.2],
                "mean_luma_range": [0, 240],
                "bright_luma_fraction_max": 0.05,
                "valid_depth_fraction_range": [0.3, 0.9],
                "hidden_valid_fraction_min": 0.01,
                "hidden_valid_fraction_max": 0.5,
                "rgb_mean_absolute_change_min": 0.25,
            },
        }
        self.policy_path = self.base / "policy.json"
        self.write_policy()
        self.job = {
            "schema_version": SCHEMA_VERSION,
            "kind": "generate_sensor_conditions",
            "job_id": "fixture_conditions",
            "backend": "real_sensor_conditions",
            "dataset_id": "fixture",
            "source_manifest_sha256": file_hash(self.manifest),
            "quality_policy_sha256": file_hash(self.policy_path),
            "source_split": "train",
            "source_sample_ids": ["A"],
            "seed": 1337,
            "required_modalities": ["rgb", "depth", "semantic", "lidar"],
            "controls": {
                "preserve": [
                    "native_geometry",
                    "labels",
                    "calibration",
                    "pose",
                    "timestamps",
                    "retained_depth",
                    "unvaried_modalities",
                ],
                "prohibit": [
                    "geometry_warp",
                    "depth_noise",
                    "lidar_corruption",
                    "label_rewrite",
                ],
            },
            "conditions": [
                {"condition_id": "darker", "method": "rgb_gain", "gain": 0.9},
                {
                    "condition_id": "missing",
                    "method": "depth_mask_transfer",
                    "donor_sample_ids": ["B"],
                },
            ],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self):
        self.manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in self.records), encoding="utf-8"
        )

    def write_policy(self):
        self.policy_path.write_text(json.dumps(self.policy), encoding="utf-8")

    def adapter(self):
        return PairedPngAdapter(self.raw, self.manifest, "fixture")

    def generate(self, name="output"):
        return generate_conditions(
            self.adapter(),
            self.job,
            self.policy_path,
            self.base / name,
            {"profile": self.evidence},
        )

    def test_native_export_units_masks_labels_and_lidar_preserved(self):
        before = {path.name: file_hash(path) for path in self.raw.iterdir()}
        report = self.generate()
        self.assertEqual(report["accepted_pending_review"], 2)
        self.assertFalse(report["training_eligible"])
        attempts = [
            json.loads(line)
            for line in (self.base / "output/attempts.jsonl").read_text().splitlines()
        ]
        clean = json.loads((self.base / "output/clean_manifest.jsonl").read_text())
        for attempt in attempts:
            generated = attempt["record"]
            self.assertEqual(generated["camera"], self.records[0]["camera"])
            self.assertEqual(generated["timestamp_seconds"], 1)
            self.assertEqual(generated["source_metadata"]["pose"], [1, 2, 3])
            self.assertEqual(generated["modalities"]["depth"]["scale_to_metres"], 0.01)
            for name in ("semantic", "lidar"):
                self.assertEqual(
                    generated["modalities"][name], clean["modalities"][name]
                )
            self.assertFalse(attempt["training_eligible"])
        masked = attempts[1]
        with Image.open(
            self.base / "output" / masked["record"]["modalities"]["depth"]["path"]
        ) as image:
            depth = np.array(image)
        self.assertEqual(int(depth[0, 2]), 65535)
        self.assertEqual(int(depth[1, 0]), 400)
        self.assertAlmostEqual(masked["metrics"]["hidden_valid_fraction"], 1 / 6)
        self.assertTrue((self.base / "output/review/index.html").is_file())
        self.assertEqual(
            before, {path.name: file_hash(path) for path in self.raw.iterdir()}
        )
        for relative, digest in report["artifact_sha256"].items():
            self.assertEqual(file_hash(self.base / "output" / relative), digest)

    def test_held_out_source_and_donor_rejected_before_decoding(self):
        for field in ("source", "donor"):
            job = copy.deepcopy(self.job)
            if field == "source":
                job["source_sample_ids"] = ["C"]
            else:
                job["conditions"][1]["donor_sample_ids"] = ["C"]
            with patch.object(PairedPngAdapter, "read_rgb_depth") as decode:
                with self.assertRaisesRegex(ValueError, "training sample"):
                    generate_conditions(
                        self.adapter(),
                        job,
                        self.policy_path,
                        self.base / field,
                        {"profile": self.evidence},
                    )
                decode.assert_not_called()

    def test_unsupported_changes_and_embedded_machine_paths_rejected(self):
        for change in (
            {"method": "lidar_thinning"},
            {"method": "gaussian_depth_noise"},
            {"method": "geometry_warp"},
        ):
            job = copy.deepcopy(self.job)
            job["conditions"][0].update(change)
            with self.assertRaises(ValidationError):
                validate_job(job)
        job = copy.deepcopy(self.job)
        job["executable"] = "anything.exe"
        with self.assertRaises(ValidationError):
            validate_job(job)

    def test_quality_bound_and_evidence_tampering_rejected(self):
        self.job["conditions"][0]["gain"] = 2
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.generate()
        self.job["conditions"][0]["gain"] = 0.9
        self.evidence.write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence checksum"):
            self.generate()

    def test_policy_hash_and_manifest_changes_rejected(self):
        self.job["quality_policy_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "policy checksum"):
            self.generate()
        self.job["quality_policy_sha256"] = file_hash(self.policy_path)
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "binding"):
            self.generate()

    def test_noop_is_rejected_without_exporting_training_ready_payloads(self):
        self.job["conditions"] = [
            {"condition_id": "noop", "method": "rgb_gain", "gain": 1}
        ]
        report = self.generate()
        self.assertEqual(report["quality_outcome"], "all_rejected")
        self.assertFalse((self.base / "output/variants").exists())
        self.assertFalse(report["training_eligible"])

    def test_requested_hidden_fraction_is_checked(self):
        self.policy["bounds"]["hidden_valid_fraction_max"] = 0.1
        self.write_policy()
        self.job["quality_policy_sha256"] = file_hash(self.policy_path)
        report = self.generate()
        self.assertEqual(report["rejected_variants"], 1)
        row = json.loads(
            (self.base / "output/attempts.jsonl").read_text().splitlines()[1]
        )
        self.assertIn(
            "depth_hidden_fraction_outside_requested_bounds", row["rejection_reasons"]
        )

    def test_identity_seed_is_stable_across_sample_order(self):
        first = condition_seed(1337, "fixture", "A", "missing")
        self.assertEqual(first, condition_seed(1337, "fixture", "A", "missing"))
        self.assertNotEqual(first, condition_seed(1337, "fixture", "B", "missing"))
        self.generate("first")
        self.job["conditions"].reverse()
        self.generate("second")

        def by_variant(folder):
            return {
                row["variant_id"]: row
                for row in (
                    json.loads(line)
                    for line in (self.base / folder / "attempts.jsonl")
                    .read_text()
                    .splitlines()
                )
            }

        self.assertEqual(by_variant("first"), by_variant("second"))

    def test_sources_and_outputs_cannot_overlap_or_overwrite(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            generate_conditions(
                self.adapter(),
                self.job,
                self.policy_path,
                self.raw / "derivatives",
                {"profile": self.evidence},
            )
        self.generate()
        with self.assertRaises(FileExistsError):
            self.generate()

    def test_donor_self_and_camera_mismatch_rejected_before_decode(self):
        self.job["conditions"][1]["donor_sample_ids"] = ["A"]
        with self.assertRaisesRegex(ValueError, "different training donor"):
            self.generate()
        self.job["conditions"][1]["donor_sample_ids"] = ["B"]
        self.records[1]["camera"]["intrinsics"][0] = 5
        self.write_manifest()
        self.policy["source_manifest_sha256"] = file_hash(self.manifest)
        self.write_policy()
        self.job.update(
            source_manifest_sha256=file_hash(self.manifest),
            quality_policy_sha256=file_hash(self.policy_path),
        )
        with self.assertRaisesRegex(ValueError, "same native camera"):
            self.generate()

    def test_invented_or_changed_measurements_and_wrong_modality_rejected(self):
        pair = self.adapter().read_rgb_depth(self.records[0])
        condition = self.job["conditions"][0]
        rgb, depth = transform_pair(pair, condition)
        changed = depth.copy()
        changed[1, 0] += 1
        with self.assertRaisesRegex(ValueError, "Retained"):
            check_variant(pair, rgb, changed, condition, self.policy["bounds"])
        changed = depth.copy()
        changed[1, 3] = 100
        with self.assertRaisesRegex(ValueError, "invented"):
            check_variant(pair, rgb, changed, condition, self.policy["bounds"])
        changed = depth.copy()
        changed[1, 0] = 65535
        with self.assertRaisesRegex(ValueError, "RGB-only"):
            check_variant(pair, rgb, changed, condition, self.policy["bounds"])

    def test_invalid_quality_envelopes_are_rejected(self):
        self.policy["bounds"]["rgb_gain_range"] = [1.2, 0.8]
        with self.assertRaisesRegex(ValueError, "Reversed"):
            validate_sensor_policy(self.policy)
        self.policy["bounds"]["rgb_gain_range"] = [0.8, 1.2]
        self.policy["bounds"]["hidden_valid_fraction_min"] = float("nan")
        with self.assertRaises(ValueError):
            validate_sensor_policy(self.policy)

    def test_failed_export_cannot_leave_a_successful_profile(self):
        with patch(
            "src.pipeline.sensor_conditions._checked_png",
            side_effect=OSError("simulated disk failure"),
        ):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.generate()
        profile = json.loads((self.base / "output/profile.json").read_text())
        self.assertEqual(profile["status"], "failed")
        self.assertFalse(profile["training_eligible"])

    def test_reused_held_out_payload_path_is_rejected_before_decode(self):
        self.records[2]["modalities"]["rgb"]["path"] = "rgb.png"
        self.write_manifest()
        with patch.object(PairedPngAdapter, "read_rgb_depth") as decode:
            with self.assertRaisesRegex(ValueError, "payload path crosses"):
                self.generate()
            decode.assert_not_called()

    def test_partitioned_jobs_keep_variant_identity_and_decisions(self):
        self.job["source_sample_ids"] = ["B", "A"]
        self.job["conditions"][1]["donor_sample_ids"] = ["A", "B"]
        self.generate("combined")
        combined = {
            row["variant_id"]: row
            for row in (
                json.loads(line)
                for line in (self.base / "combined/attempts.jsonl")
                .read_text()
                .splitlines()
            )
        }
        partitioned = {}
        for sample_id in ("A", "B"):
            self.job["source_sample_ids"] = [sample_id]
            self.generate(sample_id)
            partitioned.update(
                {
                    row["variant_id"]: row
                    for row in (
                        json.loads(line)
                        for line in (self.base / sample_id / "attempts.jsonl")
                        .read_text()
                        .splitlines()
                    )
                }
            )
        self.assertEqual(combined, partitioned)

    def test_real_subprocess_runs_generation_and_verifies_artifacts(self):
        runtime = {
            "dataset": {
                "adapter": "paired_png_manifest",
                "root": str(self.raw),
                "manifest": str(self.manifest),
            },
            "quality_policy": str(self.policy_path),
            "evidence_files": {"profile": str(self.evidence)},
        }
        result = LocalExecutor(
            Path(sys.executable), Path(__file__).resolve().parents[1]
        ).run(self.job, runtime, self.base / "attempt")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["accepted_pending_review"], 2)
        self.assertFalse(result["training_eligible"])


if __name__ == "__main__":
    unittest.main()
