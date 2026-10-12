"""Loader contracts for explicit IMU sorting, coverage and measurement integrity."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ingestion"))
import torwic_loader
from torwic_loader import MODALITIES, SequenceImu, build_manifest, load_sample


class ImuIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        sequence = self.root / "sequence" / "sequence"
        sequence.mkdir(parents=True)
        self.record = {
            "sample_id": "sequence/000000",
            "sequence_id": "sequence",
            "timestamp_seconds": 1.0,
            "pose_xyz_qxyzw": [0, 0, 0, 0, 0, 0, 1],
            "split": "train",
        }
        self.cal = {"Camera1.height": 2, "Camera1.width": 3}
        images = {
            "rgb_path": ("image_left", np.zeros((2, 3, 3), dtype=np.uint8)),
            "depth_path": (
                "depth_left",
                np.array([[0, 1000, 2000], [3000, 4000, 5000]], dtype=np.uint16),
            ),
            "semantic_path": (
                "segmentation_greyscale_left",
                np.ones((2, 3), dtype=np.uint8),
            ),
        }
        for key, (folder, array) in images.items():
            path = sequence / folder / "000000.png"
            path.parent.mkdir()
            Image.fromarray(array).save(path)
            self.record[key] = path.relative_to(self.root).as_posix()
        lidar_path = sequence / "lidar" / "000000.pcd"
        lidar_path.parent.mkdir()
        header = b"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS 1\nDATA binary\n"
        lidar_path.write_bytes(header + np.array([[1, 2, 3]], dtype="<f4").tobytes())
        self.record["lidar_path"] = lidar_path.relative_to(self.root).as_posix()
        self.original_rows = np.column_stack(
            [[0.0, 2.0, 1.0], np.arange(18).reshape(3, 6)]
        )
        self.original_bytes = {}
        for side in ("left", "right"):
            path = sequence / f"imu_{side}.txt"
            np.savetxt(path, self.original_rows)
            self.record[f"imu_{side}_path"] = path.relative_to(self.root).as_posix()
            self.original_bytes[side] = path.read_bytes()

    def reader(self, **kwargs):
        return SequenceImu(
            self.root, self.record, max_gap_seconds=0.4, sort=True, **kwargs
        )

    def test_default_loader_does_not_read_imu(self):
        for side in ("left", "right"):
            (self.root / self.record[f"imu_{side}_path"]).unlink()
        sample = load_sample(self.root, self.record, self.cal)
        self.assertNotIn("imu", sample)
        self.assertTrue(np.isnan(sample["depth_m"][0, 0]))

    def test_strict_ordering_and_explicit_gap_limit(self):
        with self.assertRaisesRegex(ValueError, "Backward"):
            SequenceImu(self.root, self.record, max_gap_seconds=0.4)
        for limit in (None, 0, -1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "gap limit"):
                SequenceImu(self.root, self.record, max_gap_seconds=limit, sort=True)

    def test_loader_preserves_measurements_raw_bytes_and_depth_units(self):
        reader = self.reader()
        sample = load_sample(self.root, self.record, self.cal, imu_sequence=reader)
        for side in ("left", "right"):
            bracket = sample["imu"][side]
            self.assertTrue(bracket["valid"])
            self.assertEqual(bracket["reason"], "exact_match")
            np.testing.assert_array_equal(
                bracket["measurement_rows"], self.original_rows[[2, 2], 1:]
            )
            self.assertFalse(bracket["measurements_interpolated"])
            self.assertTrue(bracket["ordering_report"]["rows_reordered"])
            self.assertEqual(
                (self.root / self.record[f"imu_{side}_path"]).read_bytes(),
                self.original_bytes[side],
            )
        np.testing.assert_allclose(
            sample["depth_m"][sample["depth_valid"]], [1, 2, 3, 4, 5]
        )

    def test_outside_span_and_large_gaps_do_not_return_readings(self):
        reader = self.reader()
        for timestamp, reason in (
            (-0.1, "outside_span"),
            (2.1, "outside_span"),
            (0.5, "gap_exceeds_limit"),
        ):
            record = {**self.record, "timestamp_seconds": timestamp}
            sample = load_sample(self.root, record, self.cal, imu_sequence=reader)
            for bracket in sample["imu"].values():
                self.assertFalse(bracket["valid"])
                self.assertEqual(bracket["reason"], reason)
                self.assertTrue(np.isnan(bracket["measurement_rows"]).all())
                np.testing.assert_array_equal(bracket["bracket_indices"], [-1, -1])

    def test_endpoints_and_returned_arrays_are_isolated_from_cached_rows(self):
        reader = self.reader()
        for timestamp in (0.0, 2.0):
            result = reader.bracket_record(
                {**self.record, "timestamp_seconds": timestamp}
            )
            self.assertTrue(result["left"]["valid"])
        returned = reader.bracket_record(self.record)
        returned["left"]["measurement_rows"][:] = 999
        returned["left"]["ordering_report"]["measurement_columns"].clear()
        fresh = reader.bracket_record(self.record)
        np.testing.assert_array_equal(
            fresh["left"]["measurement_rows"], self.original_rows[[2, 2], 1:]
        )
        self.assertEqual(
            fresh["left"]["ordering_report"]["measurement_columns"], [1, 2, 3, 4, 5, 6]
        )
        self.assertFalse(reader.rows["left"].flags.writeable)

    def test_cross_sequence_swapped_sides_and_escape_rejected(self):
        reader = self.reader()
        for changes in (
            {"sequence_id": "other"},
            {"imu_left_path": self.record["imu_right_path"]},
            {"imu_left_path": "../outside.txt"},
            {"rgb_path": "other/other/image_left/000000.png"},
        ):
            with self.assertRaises(ValueError):
                reader.bracket_record({**self.record, **changes})
        with self.assertRaisesRegex(ValueError, "same dataset root"):
            load_sample(self.root / "other", self.record, self.cal, imu_sequence=reader)

    def test_nonfinite_query_and_nonadjacent_duplicate_rejected(self):
        reader = self.reader()
        with self.assertRaises(ValueError):
            reader.bracket_record({**self.record, "timestamp_seconds": np.nan})
        path = self.root / self.record["imu_left_path"]
        rows = self.original_rows.copy()
        rows[2, 0] = rows[0, 0]
        np.savetxt(path, rows)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.reader()

    def test_valid_between_rows_returns_unmodified_neighbors(self):
        reader = SequenceImu(self.root, self.record, max_gap_seconds=1.0, sort=True)
        result = reader.bracket_record({**self.record, "timestamp_seconds": 0.5})
        for bracket in result.values():
            self.assertTrue(bracket["valid"])
            self.assertEqual(bracket["reason"], "bracketed")
            np.testing.assert_array_equal(
                bracket["measurement_rows"], self.original_rows[[0, 2], 1:]
            )

    def test_manifest_rejects_nonadjacent_duplicate_imu_times(self):
        sequence = (self.root / self.record["imu_left_path"]).parent
        (self.root / "groundtruth_map.ply").touch()
        np.savetxt(sequence / "frame_times.txt", [0, 1, 2])
        poses = np.zeros((3, 8))
        poses[:, 0] = [0, 1, 2]
        poses[:, 7] = 1
        np.savetxt(sequence / "traj_gt.txt", poses)
        for folder, extension in MODALITIES.values():
            directory = sequence / folder
            directory.mkdir(exist_ok=True)
            for index in range(3):
                (directory / f"{index:06d}{extension}").touch(exist_ok=True)
        rows = self.original_rows.copy()
        rows[2, 0] = rows[0, 0]
        np.savetxt(sequence / "imu_left.txt", rows)
        with patch.object(torwic_loader, "calibration", return_value=self.cal):
            with self.assertRaisesRegex(ValueError, "Duplicate IMU"):
                build_manifest(self.root)

    def test_ingestion_cli_rejects_raw_output_before_loading(self):
        arguments = [
            "torwic_loader.py",
            "--root",
            str(self.root),
            "--output",
            str(self.root / "reports"),
        ]
        with patch.object(sys, "argv", arguments):
            with patch.object(torwic_loader, "build_manifest") as build:
                with self.assertRaisesRegex(ValueError, "immutable raw root"):
                    torwic_loader.main()
                build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
