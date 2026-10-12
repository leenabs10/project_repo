"""Reject corrupted native synthetic depth, validity and semantic exports."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "validation"))
from renderer_pilot_checks import validate_arrays


class PilotArrayTests(unittest.TestCase):
    def arrays(self):
        camera = {"height": 3, "width": 4}
        rgb = np.full((3, 4, 3), 120, dtype=np.uint8)
        depth = np.full((3, 4), 3, dtype=np.float32)
        valid = np.ones((3, 4), dtype=bool)
        semantic = np.full((3, 4), 7, dtype=np.uint8)
        depth[0, 0], valid[0, 0], semantic[0, 0] = np.nan, False, 0
        return rgb, depth, valid, semantic, camera

    def test_valid_background_and_measurements_are_preserved(self):
        arrays = self.arrays()
        original = [array.copy() for array in arrays[:4]]
        validate_arrays(*arrays)
        for before, after in zip(original, arrays[:4]):
            np.testing.assert_array_equal(before, after)

    def test_invented_depth_and_visible_unknowns_are_rejected(self):
        for field, value in ((1, 3), (2, True), (3, 7)):
            arrays = self.arrays()
            arrays[field][0, 0] = value
            with self.assertRaises(ValueError):
                validate_arrays(*arrays)

    def test_negative_nonfinite_and_unknown_classes_are_rejected(self):
        for field, value in ((1, -1), (1, np.inf), (3, 16)):
            arrays = self.arrays()
            arrays[field][1, 1] = value
            with self.assertRaises(ValueError):
                validate_arrays(*arrays)


if __name__ == "__main__":
    unittest.main()
