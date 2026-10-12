"""Contract checks for timing, leakage protection and camera-axis projection."""

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

BASE = Path(__file__).resolve().parents[1] / "src"
for folder in ["ingestion", "profiling", "validation"]:
    sys.path.insert(0, str(BASE / folder))
from imu_timing import load_imu, bracket_frames
from dataset_profile import select_training, safe_path
from torwic_loader import MODALITIES
from sensor_projection import project_lidar_depth, stable_depth_pixels


class TimingTests(unittest.TestCase):
    def test_sort_is_explicit_and_leaves_raw_file_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "imu.txt"
            rows = np.column_stack(
                [np.array([0.0, 2.0, 1.0]), np.arange(18).reshape(3, 6)]
            )
            np.savetxt(path, rows)
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                load_imu(path)
            ordered, report = load_imu(path, sort=True)
            np.testing.assert_array_equal(ordered, rows[[0, 2, 1]])
            self.assertEqual(report["backward_raw_intervals"], 1)
            self.assertTrue(report["rows_reordered"])
            self.assertEqual(path.read_bytes(), original)

    def test_duplicates_rejected_even_if_nonadjacent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "imu.txt"
            np.savetxt(path, np.column_stack([[0, 1, 0], np.zeros((3, 6))]))
            with self.assertRaises(ValueError):
                load_imu(path, sort=True)

    def test_exact_endpoints_gap_limit_and_no_extrapolation(self):
        left, right, valid, gaps = bracket_frames(
            [0, 1, 2], [-0.1, 0, 0.5, 1, 2, 2.1], max_gap_seconds=0.4
        )
        np.testing.assert_array_equal(valid, [False, True, False, True, True, False])
        np.testing.assert_array_equal(left, [-1, 0, -1, 1, 2, -1])
        np.testing.assert_array_equal(right, left)
        self.assertEqual(gaps[2], 1)
        self.assertTrue(np.isnan(gaps[0]))


class SelectionTests(unittest.TestCase):
    def record(self):
        row = {
            "split": "train",
            "sequence_id": "Aisle_CCW_Run_1",
            "frame_id": "000000",
            "sample_id": "Aisle_CCW_Run_1/000000",
        }
        for key in list(MODALITIES) + ["imu_left_path", "imu_right_path"]:
            row[key] = "Aisle_CCW_Run_1/Aisle_CCW_Run_1/file"
        return row

    def test_test_split_excluded_and_training_path_leak_rejected(self):
        row = self.record()
        self.assertEqual(select_training([{"split": "test"}, row]), [row])
        row["lidar_path"] = "Aisle_CCW/Aisle_CCW/lidar/000000.pcd"
        with self.assertRaises(ValueError):
            select_training([row])

    def test_duplicate_ids_and_path_escape_rejected(self):
        row = self.record()
        with self.assertRaises(ValueError):
            select_training([row, row])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                safe_path(Path(temporary).resolve(), "../outside")


class ProjectionTests(unittest.TestCase):
    def cal(self):
        cal = {
            "Camera1.width": 8,
            "Camera1.height": 8,
            "Camera1.fx": 1.0,
            "Camera1.fy": 1.0,
            "Camera1.cx": 2.0,
            "Camera1.cy": 2.0,
            "Camera1.D": [0, 0, 0, 0, 0],
        }
        cal.update({f"T_cam1_os.p{axis}": 0.0 for axis in "xyz"})
        cal.update({f"T_cam1_os.q{axis}": float(axis == "w") for axis in "xyzw"})
        return cal

    def test_nearest_surface_axial_depth_and_behind_camera_rejection(self):
        depth, count = project_lidar_depth(
            np.array([[0, 0, 2], [0, 0, 1], [0, 0, -1]]), self.cal(), "left"
        )
        self.assertEqual(count, 2)
        self.assertEqual(depth[2, 2], 1)
        self.assertEqual(np.isfinite(depth).sum(), 1)

    def test_zero_distortion_equivalence_and_nonzero_shift(self):
        xyz = np.array([[2.0, 0.0, 1.0]])
        cal = self.cal()
        a, _ = project_lidar_depth(xyz, cal, "left")
        b, _ = project_lidar_depth(xyz, cal, "left", apply_distortion=True)
        np.testing.assert_array_equal(a, b)
        cal["Camera1.D"][0] = 0.1
        c, _ = project_lidar_depth(xyz, cal, "left", apply_distortion=True)
        self.assertFalse(np.array_equal(a, c))

    def test_depth_edges_and_unknowns_excluded_from_stable_region(self):
        raw = np.full((5, 5), 1000, dtype=np.uint16)
        self.assertTrue(stable_depth_pixels(raw)[2, 2])
        raw[2, 3] = 0
        self.assertFalse(stable_depth_pixels(raw)[2, 2])


if __name__ == "__main__":
    unittest.main()
