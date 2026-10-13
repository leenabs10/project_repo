"""Negative controls for cache provenance and baseline reproduction failures."""

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from src.validation.depth_cache_review import (
    compare_metrics,
    validate_cache_arrays,
    reuse_cache_evidence,
    file_hash,
)


class CacheReviewTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["Aisle_CCW_Run_1/000000", "Aisle_CCW_Run_1/000001"]
        self.rgb = np.zeros((2, 2, 4, 3), dtype=np.uint8)
        self.depth = np.full((2, 2, 4), 1000, dtype=np.uint16)

    def validate(self, ids=None, depth=None, rgb=None):
        return validate_cache_arrays(
            self.rgb if rgb is None else rgb,
            self.depth if depth is None else depth,
            self.ids if ids is None else ids,
            self.ids,
            (4, 2),
            "train",
        )

    def test_missing_and_reordered_manifest_members_are_rejected(self):
        for ids in (self.ids[:1], self.ids[::-1]):
            with self.assertRaises(ValueError):
                self.validate(ids=ids)

    def test_float_metres_cannot_masquerade_as_uint16_millimetres(self):
        with self.assertRaises(ValueError):
            self.validate(depth=self.depth.astype(np.float32) * 0.001)

    def test_incorrect_geometry_and_out_of_range_depth_are_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(rgb=self.rgb[:, :, :2])
        corrupted = self.depth.copy()
        corrupted[0, 0, 0] = 20001
        with self.assertRaises(ValueError):
            self.validate(depth=corrupted)

    def test_original_zero_depth_is_accepted_as_invalid(self):
        depth = self.depth.copy()
        depth[0, 0, 0] = 0
        self.assertEqual(self.validate(depth=depth), self.ids)

    def test_metric_count_changes_fail_even_if_average_is_equal(self):
        expected = {"nearest": {"mae_metres": 0.1, "pixels": 100}}
        with self.assertRaises(ValueError):
            compare_metrics({"nearest": {"mae_metres": 0.1, "pixels": 99}}, expected)
        with self.assertRaises(ValueError):
            compare_metrics({"nearest": {"mae_metres": 0.11, "pixels": 100}}, expected)
        self.assertEqual(
            compare_metrics(expected, expected)["nearest"]["mae_metres"], 0
        )

    def test_changed_manifest_cannot_reuse_prior_decoding_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            manifest.write_text("changed manifest")
            (root / "cache_integrity.json").write_text(
                json.dumps({"status": "passed", "manifest_sha256": "old hash"})
            )
            with self.assertRaisesRegex(ValueError, "another manifest"):
                reuse_cache_evidence(root, root, manifest, root, root / "output")

    def test_changed_raw_source_cannot_reuse_prior_decoding_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            (raw / "source.png").write_bytes(b"changed source")
            # A complete synthetic membership fixture isolates the source-hash gate.
            records = [
                {
                    "sample_id": f"Aisle_CCW_Run_1/{i:06d}",
                    "split": "train",
                    "rgb_path": "source.png",
                    "depth_path": "source.png",
                }
                for i in range(2173)
            ]
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in records))
            (root / "cache_integrity.json").write_text(
                json.dumps({"status": "passed", "manifest_sha256": file_hash(manifest)})
            )
            (root / "source_verification.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {"sample_id": row["sample_id"], "rgb_sha256": "old hash"}
                    )
                    for row in records
                )
            )
            with patch("src.validation.depth_cache_review.evaluation_records"):
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    reuse_cache_evidence(root, raw, manifest, root, root / "output")


if __name__ == "__main__":
    unittest.main()
