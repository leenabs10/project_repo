"""Error strata and exposure accounting preserve target validity and source bounds."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash
from src.profiling.goods_failure_audit import (
    image_profile,
    luma_confusion,
    read_pair,
    run_audit,
    scene_family_profile,
    weighted_exposure,
)


class GoodsFailureAuditTests(unittest.TestCase):
    def test_missing_snapshot_identity_fails_before_any_source_is_opened(self):
        for invalid in ("", None, "A" * 64, "0" * 63):
            with self.subTest(identity=invalid):
                with patch(
                    "src.profiling.goods_failure_audit.verify_snapshot"
                ) as verify:
                    with self.assertRaisesRegex(ValueError, "explicit SHA256"):
                        run_audit(
                            Path("."),
                            invalid,
                            Path("task.json"),
                            Path("control"),
                            "0" * 64,
                            Path("scene"),
                            "0" * 64,
                            Path("admission"),
                            "0" * 64,
                            Path("output"),
                        )
                    verify.assert_not_called()

    def test_half_open_luminance_bins_preserve_boundary_and_ignored_counts(self):
        gray = np.array([[0, 49, 50, 99, 100, 149, 150, 199, 200, 255]], dtype=np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        target = np.array([[1, 255, 0, 1, 1, 0, 0, 1, 1, 255]], dtype=np.uint8)
        prediction = np.array([[0, 1, 1, 1, 0, 0, 1, 1, 1, 0]], dtype=np.uint8)
        result = luma_confusion(rgb, target, prediction)
        self.assertEqual(
            {key: result[key] for key in ("tn", "fp", "fn", "tp")},
            dict(tn=1, fp=2, fn=2, tp=3),
        )
        self.assertEqual(result["ignored_pixels"], 2)
        self.assertEqual(
            [
                sum(row[key] for key in ("tn", "fp", "fn", "tp"))
                for row in result["strata"]
            ],
            [1, 2, 2, 2, 1],
        )
        self.assertAlmostEqual(result["false_positive_rate"], 2 / 3)
        self.assertAlmostEqual(result["false_negative_rate"], 2 / 5)
        profile = image_profile(rgb, target)
        self.assertEqual(
            profile["class_pixels"], dict(background=3, goods=5, ignored=2)
        )
        self.assertEqual(profile["luma_histograms"]["goods"], [1, 1, 1, 1, 1])

    def test_missing_classes_and_all_ignored_regions_remain_undefined(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        ignored = np.full((2, 3), 255, dtype=np.uint8)
        prediction = np.ones((2, 3), dtype=np.uint8)
        result = luma_confusion(rgb, ignored, prediction)
        self.assertIsNone(result["false_negative_rate"])
        self.assertIsNone(result["false_positive_rate"])
        self.assertIsNone(image_profile(rgb, ignored)["dark_goods_fraction"])
        self.assertIsNone(
            weighted_exposure({"ignored": image_profile(rgb, ignored)}, {"ignored": 2})[
                "foreground_fraction_of_supervised"
            ]
        )
        background = np.zeros((2, 3), dtype=np.uint8)
        result = luma_confusion(rgb, background, prediction)
        self.assertEqual(result["false_positive_rate"], 1)
        self.assertIsNone(result["false_negative_rate"])

    def test_weighted_exposure_counts_presentations_without_relabelling(self):
        dark_rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        bright_rgb = np.full((1, 2, 3), 200, dtype=np.uint8)
        profiles = {
            "dark": image_profile(dark_rgb, np.array([[1, 255]], dtype=np.uint8)),
            "bright": image_profile(bright_rgb, np.array([[0, 1]], dtype=np.uint8)),
        }
        preserved = copy.deepcopy(profiles)
        result = weighted_exposure(profiles, {"dark": 3, "bright": 1})
        self.assertEqual(result["presentations"], 4)
        self.assertEqual(result["class_pixels"], dict(background=1, goods=4, ignored=3))
        self.assertEqual(result["dark_goods_fraction"], 0.75)
        self.assertEqual(result["foreground_fraction_of_supervised"], 0.8)
        self.assertEqual(profiles, preserved)
        for schedule in (
            {},
            {"unknown": 1},
            {"dark": 0},
            {"dark": -1},
            {"dark": True},
            {"dark": 1.5},
        ):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                weighted_exposure(profiles, schedule)

    def test_malformed_arrays_and_prediction_ids_fail(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        target = np.zeros((2, 3), dtype=np.uint8)
        for bad_rgb, bad_target in (
            (rgb.astype(float), target),
            (rgb, target.astype(np.int64)),
            (rgb, np.full_like(target, 2)),
            (rgb[:, :2], target),
            (rgb[:0], target[:0]),
        ):
            with self.subTest(), self.assertRaises(ValueError):
                image_profile(bad_rgb, bad_target)
        for prediction in (
            target.astype(bool),
            np.full_like(target, 255),
            target[:, :2],
        ):
            with self.subTest(), self.assertRaises(ValueError):
                luma_confusion(rgb, target, prediction)

    def test_scene_family_partition_uses_visible_instances_and_checks_roles(self):
        rgb = np.array([[[10, 10, 10], [100, 100, 100], [0, 0, 0]]], dtype=np.uint8)
        target = np.array([[1, 1, 0]], dtype=np.uint8)
        instances = np.array([[7, 8, 9]], dtype=np.uint16)
        objects = [
            dict(instance_id=7, role="goods", class_id=1, mesh=dict(asset_id="tyre")),
            dict(instance_id=8, role="goods", class_id=1, mesh=dict(asset_id="carton")),
            dict(instance_id=9, role="equipment", class_id=0),
        ]
        result = scene_family_profile(rgb, target, instances, objects)
        self.assertEqual(result["tyre"], dict(goods_pixels=1, dark_goods_pixels=1))
        self.assertEqual(result["carton"], dict(goods_pixels=1, dark_goods_pixels=0))
        with self.assertRaisesRegex(ValueError, "no source object"):
            scene_family_profile(rgb, target, instances, objects[1:])
        bad = copy.deepcopy(objects)
        bad[0]["role"] = "equipment"
        with self.assertRaisesRegex(ValueError, "conflict"):
            scene_family_profile(rgb, target, instances, bad)
        with self.assertRaisesRegex(ValueError, "uint16"):
            scene_family_profile(rgb, target, instances.astype(np.int32), objects)

    def test_bound_pair_resampling_preserves_ids_and_rejects_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
            target = np.array(
                [[255, 255, 1, 1], [255, 255, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                dtype=np.uint8,
            )
            Image.fromarray(rgb).save(root / "rgb.png")
            Image.fromarray(target).save(root / "target.png")
            row = dict(
                rgb_path="rgb.png",
                target_path="target.png",
                source_rgb_sha256=file_hash(root / "rgb.png"),
                target_sha256=file_hash(root / "target.png"),
            )
            resized_rgb, resized_target = read_pair(root, row, (2, 2))
            self.assertEqual(resized_rgb.shape, (2, 2, 3))
            np.testing.assert_array_equal(resized_target, [[255, 1], [0, 1]])
            Image.fromarray(np.zeros_like(target)).save(root / "target.png")
            with self.assertRaisesRegex(ValueError, "changed"):
                read_pair(root, row)


if __name__ == "__main__":
    unittest.main()
