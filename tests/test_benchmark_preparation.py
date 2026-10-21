"""Reference integrity, admission and metrics using a small declared customer."""

import copy
import csv
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import tempfile
import shutil
import subprocess
import unittest

import numpy as np
from PIL import Image

from src.adapters.datasets.paired_images import PairedPngAdapter
from src.adapters.storage.local import file_hash
from src.adapters.tasks.goods import confusion_counts, segmentation_metrics
from src.core.contracts import SCHEMA_VERSION
from src.pipeline.sensor_conditions import generate_conditions
from src.validation.artifacts import verify_snapshot
from src.validation.dataset_acceptance import (
    prepare_decisions,
    qualify_dataset,
    read_decisions,
)
from src.validation.reference_labels import (
    decode_mask_runs,
    import_references,
    prepare_references,
)


class BenchmarkPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.raw = self.base / "source"
        self.raw.mkdir()
        self.submissions = self.base / "submissions"
        self.submissions.mkdir()
        self.records = []
        for index, split in enumerate(("train", "validation", "test")):
            rgb = np.full((2, 4, 3), 80 + index, np.uint8)
            depth = np.full((2, 4), 200, np.uint16)
            semantic = np.array([[0, 7, 7, 0], [0, 0, 7, 0]], np.uint8)
            for name, array in (("rgb", rgb), ("depth", depth), ("semantic", semantic)):
                Image.fromarray(array).save(self.raw / f"{split}_{name}.png")
            self.records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": "fixture",
                    "sample_id": split + "/frame",
                    "group_id": split + "_capture",
                    "split": split,
                    "timestamp_seconds": index,
                    "timestamp_basis": "relative",
                    "image_size": [4, 2],
                    "source_metadata": {},
                    "modalities": {
                        "rgb": {"status": "available", "path": f"{split}_rgb.png"},
                        "depth": {
                            "status": "available",
                            "path": f"{split}_depth.png",
                            "units": "centimetres",
                            "scale_to_metres": 0.01,
                            "invalid_raw_value": 65535,
                        },
                        "semantic": {
                            "status": "available",
                            "path": f"{split}_semantic.png",
                            "label_origin": "provided_model_prediction",
                        },
                    },
                }
            )
        self.manifest = self.base / "manifest.jsonl"
        self.manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in self.records)
        )
        self.adapter = PairedPngAdapter(self.raw, self.manifest, "fixture")
        definition = json.loads(
            (
                Path(__file__).parents[1] / "configs/tasks/warehouse_goods.json"
            ).read_text()
        )
        definition["minimum_reference_frames"] = {"train": 1, "validation": 1}
        self.definition_path = self.base / "task.json"
        self.definition_path.write_text(json.dumps(definition))
        self.packet = self.base / "packet"
        prepare_references(self.adapter, self.definition_path, self.packet)
        self.packet_sha = file_hash(self.packet / "profile.json")

    def tearDown(self):
        self.temporary.cleanup()

    def submission(self, split="train"):
        entry = next(
            row
            for row in json.loads((self.packet / "references.json").read_text())
            if row["binding"]["split"] == split
        )
        return {
            **entry["binding"],
            "reviewer": "Fixture annotator",
            "reviewed": True,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "width": 4,
            "height": 2,
            "mask_rle": [[0, 1], [1, 2], [0, 3], [255, 1], [0, 1]],
        }

    def write_submission(self, submission, name="train.json"):
        (self.submissions / name).write_text(json.dumps(submission))

    def import_targets(self, name="references"):
        return import_references(
            self.packet,
            self.submissions,
            self.base / name,
            self.packet_sha,
            source_root=self.raw,
        )

    def batch(self):
        evidence = self.base / "evidence.json"
        evidence.write_text("{}")
        policy = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "fixture",
            "source_manifest_sha256": file_hash(self.manifest),
            "source_split": "train",
            "training_groups": ["train_capture"],
            "basis": "training_distribution_bounds_not_sensor_calibration",
            "physical_sensor_realism_certified": False,
            "evidence_sha256": {"fixture": file_hash(evidence)},
            "bounds": {
                "rgb_gain_range": [0.8, 1.2],
                "mean_luma_range": [0, 240],
                "bright_luma_fraction_max": 0.1,
                "valid_depth_fraction_range": [0.1, 1],
                "hidden_valid_fraction_min": 0.01,
                "hidden_valid_fraction_max": 0.4,
                "rgb_mean_absolute_change_min": 0.25,
            },
        }
        policy_path = self.base / "policy.json"
        policy_path.write_text(json.dumps(policy))
        job = {
            "schema_version": SCHEMA_VERSION,
            "kind": "generate_sensor_conditions",
            "job_id": "fixture",
            "backend": "real_sensor_conditions",
            "dataset_id": "fixture",
            "source_manifest_sha256": file_hash(self.manifest),
            "quality_policy_sha256": file_hash(policy_path),
            "source_split": "train",
            "source_sample_ids": ["train/frame"],
            "seed": 42,
            "required_modalities": ["rgb", "depth", "semantic"],
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
                {"condition_id": "noop", "method": "rgb_gain", "gain": 1},
            ],
        }
        self.batch_root = self.base / "batch"
        generate_conditions(
            self.adapter, job, policy_path, self.batch_root, {"fixture": evidence}
        )
        self.batch_sha = file_hash(self.batch_root / "profile.json")
        prepare_decisions(
            self.batch_root,
            self.batch_sha,
            self.base / "feedback",
            source_root=self.raw,
        )
        self.decisions = self.base / "feedback/decisions.csv"

    def accept_conditions(self):
        with self.decisions.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row.update(appearance_decision="accept", mask_decision="accept")
        with self.decisions.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def qualify(self, name="qualified"):
        return qualify_dataset(
            self.batch_root,
            self.batch_sha,
            self.decisions,
            "Fixture annotator",
            self.base / "references",
            self.definition_path,
            self.base / name,
            source_root=self.raw,
            expected_reference_sha256=file_hash(self.base / "references/profile.json"),
        )

    def test_admitted_manifest_keeps_condition_seed_and_source_binding(self):
        self.batch()
        self.accept_conditions()
        self.write_submission(self.submission())
        self.write_submission(self.submission("validation"), "validation.json")
        self.import_targets()
        result = self.qualify()
        self.assertEqual(result["manifest_sha256"], file_hash(self.manifest))
        admitted = json.loads(
            (self.base / "qualified/eligible_manifest.jsonl").read_text()
        )
        attempts = [
            json.loads(line)
            for line in (self.batch_root / "attempts.jsonl").read_text().splitlines()
        ]
        original = next(
            row
            for row in attempts
            if row["variant_id"] == admitted["record"]["sample_id"]
        )
        self.assertEqual(admitted["condition"], original["condition"])
        self.assertEqual(admitted["generation_seed"], original["seed"])
        self.assertEqual(
            admitted["source_rgb_sha256"], original["source_sha256"]["rgb"]
        )

    def test_packet_excludes_test_and_keeps_candidates_ineligible(self):
        profile = verify_snapshot(self.packet, self.packet_sha)
        self.assertEqual(profile["selected_frames"], {"train": 1, "validation": 1})
        self.assertFalse(profile["training_eligible"])
        self.assertEqual(profile["test_sensor_files_opened"], 0)
        entries = json.loads((self.packet / "references.json").read_text())
        for row in entries:
            editor = (self.packet / row["folder"] / "index.html").read_text()
            self.assertNotIn("__SOURCE_RECORD__", editor)
            self.assertNotIn("__WIDTH__", editor)
        self.assertEqual(
            {row["binding"]["split"] for row in entries}, {"train", "validation"}
        )

    @unittest.skipUnless(
        shutil.which("node"), "Node is required for editor behavior checks"
    )
    def test_editor_brush_undo_confirmation_and_export_round_trip(self):
        entry = json.loads((self.packet / "references.json").read_text())[0]
        completed = subprocess.run(
            [
                shutil.which("node"),
                str(Path(__file__).with_name("mask_editor_smoke.js")),
                str(self.packet / entry["folder"] / "index.html"),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        submission = json.loads(completed.stdout)
        self.write_submission(submission)
        imported = self.import_targets()
        self.assertEqual(sum(imported["reference_counts"].values()), 1)

    def test_duplicate_condition_decisions_are_rejected(self):
        self.batch()
        lines = self.decisions.read_text().splitlines()
        self.decisions.write_text("\n".join([*lines, lines[1]]) + "\n")
        attempts = [
            json.loads(line)
            for line in (self.batch_root / "attempts.jsonl").read_text().splitlines()
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_decisions(self.decisions, attempts, "", self.batch_sha)

    def test_editor_escapes_source_ids_without_rescanning_inserted_text(self):
        from src.validation.reference_labels import _editor

        template = (
            Path(__file__).parents[1] / "src/validation/templates/mask_editor.html"
        ).read_text()
        binding = {"sample_id": "</script>__WIDTH__"}
        html = _editor(template, binding, {}, 4, 2)
        self.assertEqual(html.count("</script>"), 1)
        embedded = re.search(r"const source = (.*), definition =", html).group(1)
        self.assertEqual(json.loads(embedded), binding)

    def test_empty_submissions_are_pending_not_qualified(self):
        result = self.import_targets()
        self.assertFalse(result["benchmark_reference_ready"])
        self.assertEqual(result["reference_counts"], {"train": 0, "validation": 0})

    def test_required_sources_reject_held_out_and_stale_decision_snapshots(self):
        with self.assertRaisesRegex(ValueError, "training samples"):
            prepare_references(
                self.adapter,
                self.definition_path,
                self.base / "bad_packet",
                required_training_ids=("test/frame",),
            )
        self.batch()
        attempts = [
            json.loads(line)
            for line in (self.batch_root / "attempts.jsonl").read_text().splitlines()
        ]
        with self.assertRaisesRegex(ValueError, "different batch snapshot"):
            read_decisions(self.decisions, attempts, "", "0" * 64)

    def test_corrected_labels_without_condition_acceptance_are_not_admitted(self):
        self.batch()
        self.write_submission(self.submission())
        self.write_submission(self.submission("validation"), "validation.json")
        self.import_targets()
        self.assertEqual(self.qualify()["eligible_variants"], 0)

    def test_corrected_labels_keep_ignore_and_source_binding(self):
        self.write_submission(self.submission())
        self.write_submission(self.submission("validation"), "validation.json")
        result = self.import_targets()
        self.assertTrue(result["benchmark_reference_ready"])
        rows = [
            json.loads(line)
            for line in (self.base / "references/references.jsonl")
            .read_text()
            .splitlines()
        ]
        with Image.open(self.base / "references" / rows[0]["target_path"]) as image:
            mask = np.array(image)
        self.assertEqual(int((mask == 255).sum()), 1)
        self.assertEqual(rows[0]["reference_origin"], "human_reviewed_model_assisted")

    def test_ai_corrections_are_importable_but_do_not_become_human_references(self):
        self.batch()
        self.accept_conditions()
        for split in ("train", "validation"):
            submission = self.submission(split)
            submission["reviewer_kind"] = "ai"
            self.write_submission(submission, split + ".json")
        result = self.import_targets()
        self.assertEqual(result["reference_counts"], {"train": 1, "validation": 1})
        self.assertEqual(
            result["qualified_reference_counts"], {"train": 0, "validation": 0}
        )
        self.assertFalse(result["benchmark_reference_ready"])
        rows = [
            json.loads(line)
            for line in (self.base / "references/references.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertTrue(
            all(row["reference_origin"] == "ai_reviewed_model_assisted" for row in rows)
        )
        admission = self.qualify()
        self.assertEqual(admission["eligible_variants"], 0)
        self.assertFalse(admission["training_eligible"])

    def test_unknown_annotation_reviewer_kind_is_rejected(self):
        submission = self.submission()
        submission["reviewer_kind"] = "unverified"
        self.write_submission(submission)
        with self.assertRaisesRegex(ValueError, "Reviewer kind"):
            self.import_targets()

    def test_mixed_annotations_only_count_matching_reference_provenance(self):
        self.write_submission(self.submission())
        validation = self.submission("validation")
        validation["reviewer_kind"] = "ai"
        self.write_submission(validation, "validation.json")
        result = self.import_targets()
        self.assertEqual(result["reference_origin"], "mixed_model_assisted")
        self.assertEqual(
            result["qualified_reference_counts"], {"train": 1, "validation": 0}
        )
        self.assertFalse(result["benchmark_reference_ready"])

    def test_stale_binding_unknown_identity_and_unconfirmed_masks_rejected(self):
        for field, value in (
            ("source_rgb_sha256", "0" * 64),
            ("sample_id", "test/frame"),
            ("protocol_sha256", "0" * 64),
            ("reviewed", False),
            ("reviewer", " "),
        ):
            changed = self.submission()
            changed[field] = value
            self.write_submission(changed)
            with self.assertRaises(ValueError):
                self.import_targets(field)

    def test_rle_overflow_truncation_bad_ids_and_boolean_counts_fail(self):
        for runs in ([[1, 9]], [[1, 7]], [[2, 8]], [[1, True]], [[1, -1]], [[255, 8]]):
            with self.assertRaises(ValueError):
                mask = decode_mask_runs(runs, 4, 2)
                from src.adapters.tasks.goods import validate_target

                validate_target(mask, (2, 4), 0.5)

    def test_duplicate_submissions_and_packet_tampering_fail(self):
        submission = self.submission()
        self.write_submission(submission)
        self.write_submission(submission, "duplicate.json")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.import_targets()
        (self.packet / "references.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.import_targets()

    def test_raw_output_and_changed_originals_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            import_references(
                self.packet,
                self.submissions,
                self.raw / "output",
                self.packet_sha,
                source_root=self.raw,
            )
        self.write_submission(self.submission())
        (self.raw / "train_rgb.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Original reference source"):
            self.import_targets()

    def test_appearance_acceptance_cannot_admit_uncorrected_predictions(self):
        self.batch()
        self.accept_conditions()
        self.import_targets()
        result = self.qualify()
        self.assertFalse(result["training_eligible"])
        self.assertEqual(result["eligible_variants"], 0)
        self.assertFalse(result["delivered_coverage_complete"])

    def test_corrected_target_and_explicit_acceptance_admit_only_valid_variant(self):
        self.batch()
        self.accept_conditions()
        self.write_submission(self.submission())
        self.write_submission(self.submission("validation"), "validation.json")
        self.import_targets()
        result = self.qualify()
        self.assertEqual(result["eligible_variants"], 1)
        self.assertTrue(result["training_eligible"])
        self.assertFalse(result["delivered_coverage_complete"])
        accepted = json.loads(
            (self.base / "qualified/eligible_manifest.jsonl").read_text()
        )
        self.assertEqual(accepted["target_origin"], "human_reviewed_model_assisted")
        self.assertEqual(
            accepted["record"]["modalities"]["semantic"]["label_origin"],
            "provided_model_prediction",
        )
        self.assertNotEqual(
            accepted["target_path"],
            accepted["record"]["modalities"]["semantic"]["path"],
        )

    def test_pending_decisions_and_incomplete_reference_cohort_block_admission(self):
        self.batch()
        self.write_submission(self.submission())
        self.import_targets()
        self.accept_conditions()
        result = self.qualify()
        self.assertEqual(result["eligible_variants"], 0)

    def test_unknown_duplicate_and_unattributed_decisions_fail(self):
        self.batch()
        self.accept_conditions()
        attempts = [
            json.loads(line)
            for line in (self.batch_root / "attempts.jsonl").read_text().splitlines()
        ]
        with self.assertRaisesRegex(ValueError, "reviewer"):
            read_decisions(self.decisions, attempts, "", self.batch_sha)
        with self.decisions.open("a") as stream:
            stream.write(self.batch_sha + ",unknown,train/frame,accept,accept,\n")
        with self.assertRaisesRegex(ValueError, "stale or unknown"):
            read_decisions(self.decisions, attempts, "Fixture", self.batch_sha)

    def test_ignored_pixels_and_absent_foreground_metrics_are_honest(self):
        target = np.array([[1, 0, 255, 1]], np.uint8)
        counts = confusion_counts(np.array([[1, 1, 1, 0]], np.uint8), target)
        self.assertEqual(counts["valid_pixels"], 3)
        self.assertAlmostEqual(segmentation_metrics(counts)["goods_iou"], 1 / 3)
        absent = confusion_counts(
            np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8)
        )
        self.assertIsNone(segmentation_metrics(absent)["goods_iou"])
