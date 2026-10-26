"""Contact fixtures cover triangle interiors, vertical faces and preserved roles."""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from src.synthesis.mesh_stacking import pair_gap, refine_stacks, vertical_gap
from src.synthesis.warehouse_balanced import build_balanced_job
from tests import test_balanced_scene


class MeshContactTests(unittest.TestCase):
    def test_crossing_edges_find_contact_without_contained_vertices(self):
        lower = np.array([[-2, -1, 0], [2, -1, 0], [0, 2, 3.0]])
        upper = np.array([[-2, 1, 3], [2, 1, 3], [0, -2, 6.0]])
        # z_lower=y+1, z_upper=4-y: first contact occurs on crossing edges.
        self.assertAlmostEqual(pair_gap(lower, upper), 1.0)
        gap, _ = vertical_gap(lower[None], upper[None])
        self.assertAlmostEqual(gap, 1.0)
        self.assertAlmostEqual(
            vertical_gap(lower[None], (upper - [0, 0, 1])[None])[0], 0.0
        )
        self.assertLess(vertical_gap(lower[None], (upper - [0, 0, 1.1])[None])[0], 0)
        self.assertTrue(
            np.isinf(vertical_gap(lower[None], (upper + [10, 0, 0])[None])[0])
        )

    def test_vertical_faces_and_collinear_projections_are_retained(self):
        lower = np.array([[0.0, 0, 0], [0, 1, 0], [0, 0, 1]])
        horizontal = np.array([[-1.0, -1, 2], [1, -1, 2], [0, 2, 2]])
        self.assertAlmostEqual(pair_gap(lower, horizontal), 1.0)
        self.assertAlmostEqual(pair_gap(lower, lower + [0, 0, 3]), 2.0)

    def test_refinement_preserves_sources_roles_and_every_non_z_field(self):
        job = build_balanced_job(*test_balanced_scene.BalancedSceneTests().inputs())
        source = copy.deepcopy(job)
        # A sheared box has AABB contact but a real surface gap. Its true
        # thickness is 0.6 while its bounding height is 1.0.
        vertices = np.array(
            [
                [x, y, z + 0.4 * x]
                for z in (-0.3, 0.3)
                for y in (-0.5, 0.5)
                for x in (-0.5, 0.5)
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                [0, 4, 5],
                [0, 5, 1],
                [2, 3, 7],
                [2, 7, 6],
                [0, 2, 6],
                [0, 6, 4],
                [1, 5, 7],
                [1, 7, 3],
            ]
        )
        with patch(
            "src.synthesis.mesh_stacking.load_normalized_mesh",
            return_value=dict(vertices=vertices, triangles=faces),
        ):
            result, evidence = refine_stacks(job, ".")
        self.assertEqual(job, source)
        self.assertTrue(
            any(row["upper_translation_z_metres"] < 0 for row in evidence["contacts"])
        )
        self.assertTrue(
            all(
                abs(row["revised_gap_metres"] - 0.0002) < 1e-7
                for row in evidence["contacts"]
            )
        )
        self.assertFalse(evidence["training_eligible"])
        for before, after in zip(source["scenes"], result["scenes"]):
            normalized = copy.deepcopy(after)
            for old, new in zip(before["objects"], normalized["objects"]):
                self.assertEqual(old["center_metres"][:2], new["center_metres"][:2])
                if old["role"] != "goods":
                    self.assertEqual(old, new)
                new["center_metres"][2] = old["center_metres"][2]
            self.assertEqual(before, normalized)
        with self.assertRaises(ValueError):
            refine_stacks(job, ".", clearance_metres=0)
        with self.assertRaises(ValueError):
            vertical_gap(np.zeros((0, 3, 3)), np.zeros((1, 3, 3)))


if __name__ == "__main__":
    unittest.main()
