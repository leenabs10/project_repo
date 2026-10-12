"""Analytic depth and categorical-label contracts, independent of Blender."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "synthesis"))
from renderer_geometry import axial_depth, ray_lengths, semantic_ids


class RendererGeometryTests(unittest.TestCase):
    def camera(self):
        return {"width": 5, "height": 3, "fx": 2.0, "fy": 3.0, "cx": 2.0, "cy": 1.0}

    def test_range_converts_to_constant_axial_plane(self):
        camera = self.camera()
        ranges = 10 * ray_lengths(camera)
        depth, valid = axial_depth(
            ranges, camera, "euclidean_ray_range", far_clip_metres=20
        )
        np.testing.assert_allclose(depth, np.full((3, 5), 10))
        self.assertTrue(valid.all())
        self.assertGreater(ranges[0, 0], ranges[1, 2])

    def test_axial_pass_is_not_normalized_twice(self):
        source = np.full((3, 5), 10, dtype=np.float32)
        depth, valid = axial_depth(source, self.camera(), "axial_z", far_clip_metres=20)
        np.testing.assert_array_equal(depth, source)
        self.assertTrue(valid.all())

    def test_invalid_depth_and_bad_contracts_fail_closed(self):
        source = np.ones((3, 5), dtype=float)
        source[0] = [0, -1, np.inf, np.nan, 100]
        original = source.copy()
        depth, valid = axial_depth(source, self.camera(), "axial_z", far_clip_metres=20)
        self.assertFalse(valid[0].any())
        self.assertTrue(np.isnan(depth[0]).all())
        np.testing.assert_array_equal(source, original)
        for convention in ("unknown", "distance_guess"):
            with self.assertRaises(ValueError):
                axial_depth(source, self.camera(), convention, far_clip_metres=20)
        with self.assertRaises(ValueError):
            ray_lengths({**self.camera(), "fx": 0})
        with self.assertRaises(ValueError):
            axial_depth(source[:2], self.camera(), "axial_z", far_clip_metres=20)

    def test_class_indices_are_discrete_and_never_interpolated(self):
        np.testing.assert_array_equal(
            semantic_ids(np.array([[0.0, 1.0, 7.0, 15.0]])), [[0, 1, 7, 15]]
        )
        for raw in (np.array([[6.5]]), np.array([[16.0]]), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                semantic_ids(raw)


if __name__ == "__main__":
    unittest.main()
