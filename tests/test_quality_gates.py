"""Failures that must stay blocked before accepting synthetic or annotated data."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[1] / "src"
for folder in ["validation", "synthesis"]:
    sys.path.insert(0, str(BASE / folder))
from image_geometry import require_projection_policy
from annotation_provenance import audit_pair, semantic_from_instances
from augmentation_policy import (
    build_policy,
    apply_rgb_gain,
    transfer_depth_missingness,
    validate_pair,
    validate_clean_reference,
)


class GeometryTests(unittest.TestCase):
    def test_unknown_convention_and_wrong_calibration_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cal.txt"
            path.write_text("calibration")
            contract = {
                "calibration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "evidence_status": "unverified",
            }
            with self.assertRaises(ValueError):
                require_projection_policy(contract, path)
            contract.update(
                fusion_enabled=True,
                evidence_status="confirmed",
                evidence_reference="capture metadata",
                source_image_convention="raw_distorted",
            )
            self.assertTrue(require_projection_policy(contract, path))
            contract["fusion_enabled"] = False
            with self.assertRaises(ValueError):
                require_projection_policy(contract, path)
            contract["fusion_enabled"] = True
            path.write_text("different calibration")
            with self.assertRaises(ValueError):
                require_projection_policy(contract, path)

    def test_rectified_intrinsics_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cal.txt"
            path.write_text("calibration")
            contract = {
                "calibration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "evidence_status": "confirmed",
                "evidence_reference": "export metadata",
                "fusion_enabled": True,
                "source_image_convention": "rectified_pinhole",
            }
            with self.assertRaises(ValueError):
                require_projection_policy(contract, path)
            contract["rectified_intrinsics_confirmed"] = True
            self.assertFalse(require_projection_policy(contract, path))


class PairTests(unittest.TestCase):
    def test_clean_reference_geometry_and_classes_required(self):
        rgb, depth, semantic, _ = self.arrays()
        validate_clean_reference(rgb, depth, semantic, (4, 4))
        with self.assertRaises(ValueError):
            validate_clean_reference(rgb, depth, semantic, (5, 5))
        semantic[0, 0] = 16
        with self.assertRaises(ValueError):
            validate_clean_reference(rgb, depth, semantic, (4, 4))

    def arrays(self):
        rgb = np.full((4, 4, 3), 100, dtype=np.uint8)
        depth = np.full((4, 4), 1000, dtype=np.uint16)
        depth[0, 0] = 0
        semantic = np.ones((4, 4), dtype=np.uint8)
        bounds = {
            "accepted_mean_luma_range": [0, 255],
            "accepted_bright_luma_fraction_max": 1,
            "accepted_valid_depth_fraction_range": [0.01, 1],
        }
        return rgb, depth, semantic, bounds

    def test_mask_transfer_preserves_labels_values_and_sources(self):
        rgb, depth, semantic, bounds = self.arrays()
        original = depth.copy()
        donor = depth.copy()
        donor[1, 1] = 0
        result = transfer_depth_missingness(depth, donor)
        self.assertEqual(
            validate_pair(rgb, depth, semantic, rgb, result, semantic, bounds), []
        )
        np.testing.assert_array_equal(depth, original)
        self.assertEqual(result[1, 1], 0)

    def test_modified_measurements_invented_depth_and_labels_rejected(self):
        rgb, depth, semantic, bounds = self.arrays()
        corrupt = depth.copy()
        corrupt[0, 0] = 1
        corrupt[1, 1] = 999
        labels = semantic.copy()
        labels[0, 0] = 2
        reasons = validate_pair(rgb, depth, semantic, rgb, corrupt, labels, bounds)
        self.assertIn("Originally invalid depths invented", reasons)
        self.assertIn("Retained depth measurements modified", reasons)
        self.assertIn("Semantic target modified", reasons)

    def test_gain_retains_shape_and_source(self):
        rgb, _, _, _ = self.arrays()
        original = rgb.copy()
        result = apply_rgb_gain(rgb, 1.1)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, rgb.shape)
        np.testing.assert_array_equal(rgb, original)
        with self.assertRaises(ValueError):
            apply_rgb_gain(rgb, float("nan"))

    def test_nontraining_evidence_rejected(self):
        with self.assertRaises(ValueError):
            build_policy({"split": "validation", "status": "passed"})


class AnnotationTests(unittest.TestCase):
    def test_instance_indices_below_16_and_multiple_instances_remap(self):
        mask = np.array([[0, 9, 20]], dtype=np.uint8)
        metadata = {"fork_truck": [{"index": 9}, {"index": 20}]}
        mapped, lookup = semantic_from_instances(mask, metadata)
        np.testing.assert_array_equal(mapped, [[0, 14, 14]])
        np.testing.assert_array_equal(mask, [[0, 9, 20]])

    def test_conflicting_instance_mapping_rejected(self):
        with self.assertRaises(ValueError):
            semantic_from_instances(
                np.array([[9]], dtype=np.uint8),
                {"fork_truck": {"index": 9}, "person": {"index": 9}},
            )

    def test_valid_payloads_do_not_grant_unknown_split_training_permission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "group" / "image"
            folder.mkdir(parents=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                folder / "source_image.png"
            )
            Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(
                folder / "combined_indexedImage.png"
            )
            (folder / "raw_labels_2d.json").write_text(
                json.dumps({"labelMapping": {"driveable_ground": {"index": 1}}})
            )
            record = audit_pair(folder, root, {(1280, 720)})
            self.assertEqual(record["payload_integrity"], "passed")
            self.assertFalse(record["training_eligible"])
            self.assertFalse(record["native_trajectory_size_compatible"])

    def test_unknown_class_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                folder / "source_image.png"
            )
            Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(
                folder / "combined_indexedImage.png"
            )
            (folder / "raw_labels_2d.json").write_text(json.dumps({"labelMapping": {}}))
            self.assertEqual(
                audit_pair(folder, folder, {(4, 4)})["payload_integrity"], "failed"
            )


if __name__ == "__main__":
    unittest.main()
