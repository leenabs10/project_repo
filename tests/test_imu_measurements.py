"""Verified fixtures for IMU metadata gates and conservative linear alignment."""

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ingestion"))
from imu_measurements import align_verified_measurements, require_measurement_policy


class MeasurementPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "imu.txt"
        self.rows = np.array(
            [
                [0, 10, 20, 30, 1, 2, 3],
                [1, 20, 40, 60, 2, 4, 6],
                [3, 40, 80, 120, 4, 8, 12],
            ],
            dtype=float,
        )
        np.savetxt(self.path, self.rows)
        self.original_bytes = self.path.read_bytes()
        self.policy = {
            "evidence_status": "confirmed",
            "source_sha256": hashlib.sha256(self.original_bytes).hexdigest(),
            "angular_velocity_columns_xyz": [4, 5, 6],
            "linear_acceleration_columns_xyz": [1, 2, 3],
            "angular_velocity_units": "rad/s",
            "linear_acceleration_units": "m/s^2",
            "measurement_frame": "fixture_imu",
            "timestamps_match_frame_basis": True,
            "max_gap_seconds": 1.0,
            "evidence_references": {
                field: "Known synthetic test fixture"
                for field in (
                    "column_mapping",
                    "units",
                    "coordinate_frame",
                    "clock_basis",
                    "gap_policy",
                )
            },
        }
        self.contract = {
            "schema_version": 1,
            "interpolation_enabled": True,
            "streams": {"imu.txt": self.policy},
        }

    def test_disabled_and_unconfirmed_contracts_are_blocked(self):
        self.contract["interpolation_enabled"] = False
        with self.assertRaisesRegex(ValueError, "disabled"):
            require_measurement_policy(self.root, "imu.txt", self.contract)
        self.contract["interpolation_enabled"] = True
        self.policy["evidence_status"] = "unverified"
        with self.assertRaisesRegex(ValueError, "confirmed"):
            require_measurement_policy(self.root, "imu.txt", self.contract)

    def test_all_evidence_dimensions_required(self):
        for field in self.policy["evidence_references"]:
            contract = copy.deepcopy(self.contract)
            contract["streams"]["imu.txt"]["evidence_references"][field] = "  "
            with self.assertRaisesRegex(ValueError, "evidence"):
                require_measurement_policy(self.root, "imu.txt", contract)

    def test_column_aliases_time_and_boolean_indices_rejected(self):
        for columns in ([1, 2, 3], [0, 5, 6], [True, 5, 6], [4, 5], "4,5,6"):
            contract = copy.deepcopy(self.contract)
            contract["streams"]["imu.txt"]["angular_velocity_columns_xyz"] = columns
            with self.assertRaises(ValueError):
                require_measurement_policy(self.root, "imu.txt", contract)

    def test_units_frame_clock_and_gap_must_be_explicit(self):
        for key, value in (
            ("angular_velocity_units", "deg/s"),
            ("linear_acceleration_units", "g"),
            ("measurement_frame", ""),
            ("timestamps_match_frame_basis", False),
            ("max_gap_seconds", None),
            ("max_gap_seconds", np.nan),
            ("max_gap_seconds", True),
            ("max_gap_seconds", -1),
        ):
            contract = copy.deepcopy(self.contract)
            contract["streams"]["imu.txt"][key] = value
            with self.assertRaises(ValueError):
                require_measurement_policy(self.root, "imu.txt", contract)

    def test_stale_unknown_and_escaped_sources_rejected(self):
        for path in ("../escape.txt", "unknown.txt", str(self.path)):
            with self.assertRaises(ValueError):
                require_measurement_policy(self.root, path, self.contract)
        self.path.write_text("changed source")
        with self.assertRaisesRegex(ValueError, "source version"):
            require_measurement_policy(self.root, "imu.txt", self.contract)

    def test_known_linear_alignment_exact_matches_and_invalid_masks(self):
        result = align_verified_measurements(
            self.root, "imu.txt", [-0.1, 0, 0.5, 1, 2, 3, 3.1], self.contract
        )
        np.testing.assert_array_equal(
            result["valid"], [False, True, True, True, False, True, False]
        )
        np.testing.assert_allclose(result["angular_velocity_rad_s"][2], [1.5, 3, 4.5])
        np.testing.assert_allclose(result["linear_acceleration_m_s2"][2], [15, 30, 45])
        np.testing.assert_array_equal(
            result["angular_velocity_rad_s"][[1, 3, 5]], self.rows[:, 4:]
        )
        self.assertTrue(
            np.isnan(result["angular_velocity_rad_s"][~result["valid"]]).all()
        )
        self.assertTrue(
            np.isnan(result["linear_acceleration_m_s2"][~result["valid"]]).all()
        )
        np.testing.assert_array_equal(
            result["bracket_indices"][~result["valid"]], np.full((3, 2), -1)
        )
        self.assertEqual(result["reason"][4], "gap_exceeds_limit")
        self.assertEqual(self.path.read_bytes(), self.original_bytes)

    def test_mapping_permutation_controls_output_axes(self):
        self.policy["angular_velocity_columns_xyz"] = [6, 4, 5]
        result = align_verified_measurements(self.root, "imu.txt", [0.5], self.contract)
        np.testing.assert_allclose(result["angular_velocity_rad_s"][0], [4.5, 1.5, 3])
        self.assertEqual(result["measurement_frame"], "fixture_imu")
        self.assertFalse(result["frame_conversion_applied"])
        self.assertFalse(result["sensor_fusion_applied"])

    def test_sorting_stays_explicit_and_empty_queries_are_supported(self):
        np.savetxt(self.path, self.rows[[0, 2, 1]])
        self.policy["source_sha256"] = hashlib.sha256(
            self.path.read_bytes()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "Backward"):
            align_verified_measurements(self.root, "imu.txt", [0.5], self.contract)
        result = align_verified_measurements(
            self.root, "imu.txt", [], self.contract, sort=True
        )
        self.assertEqual(result["angular_velocity_rad_s"].shape, (0, 3))
        self.assertTrue(result["ordering_report"]["rows_reordered"])


if __name__ == "__main__":
    unittest.main()
