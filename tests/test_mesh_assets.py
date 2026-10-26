"""Triangle fixtures verify UV, normalization, version and source boundaries."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.adapters.generation.material_assets import verify_material_assets
from src.adapters.generation.mesh_assets import load_normalized_mesh
from src.adapters.storage.local import file_hash
from src.core.scene import validate_scene_job
from src.validation.mesh_geometry import triangle_distance, scene_intersections
from src.synthesis.warehouse_surfaces import bind_surfaces
from tests.test_scene_pipeline import fixture_job


class MeshAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.vertices = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
            ],
            np.float32,
        )
        self.triangles = np.array(
            [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], np.int32
        )
        self.uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
        self.save_mesh()

    def tearDown(self):
        self.temporary.cleanup()

    def save_mesh(self):
        path = self.root / "shape.npz"
        np.savez_compressed(
            path, vertices=self.vertices, triangles=self.triangles, uv=self.uv
        )
        self.binding = dict(
            asset_id="fixture.mesh", path="shape.npz", sha256=file_hash(path)
        )

    def job_and_catalog(self):
        job = fixture_job()
        job["schema_version"] = "dubaidata.scene/1.1"
        for name in ("color", "roughness"):
            Image.new("RGB", (16, 16), (120, 100, 80)).save(self.root / f"{name}.png")
        maps = {
            role: dict(
                path=f"{name}.png",
                sha256=file_hash(self.root / f"{name}.png"),
                color_space=space,
            )
            for role, name, space in (
                ("base_color", "color", "sRGB"),
                ("roughness", "roughness", "Non-Color"),
            )
        }
        catalog = dict(
            schema_version="dubaidata.material-assets/1.0",
            assets={
                "fixture.mesh": dict(
                    license="CC0-1.0",
                    source_page="https://example.invalid/fixture",
                    tile_size_metres=[1, 1, 1],
                    maps=maps,
                    mesh=self.binding,
                )
            },
        )
        path = self.root / "catalog.json"
        path.write_text(json.dumps(catalog))
        job["asset_catalog_sha256"] = file_hash(path)
        job["materials"]["uv_fixture"] = dict(
            color=[0.5, 0.4, 0.3],
            roughness=0.8,
            metallic=0,
            texture=False,
            texture_set=dict(
                asset_id="fixture.mesh",
                tile_size_metres=[1, 1, 1],
                base_color_mix=1,
                projection="UV",
                maps=maps,
            ),
        )
        for scene in job["scenes"]:
            body = next(obj for obj in scene["objects"] if obj["role"] == "goods")
            body.update(
                asset_id="registered.normalized_triangle_mesh/1.0",
                mesh=self.binding,
                material="uv_fixture",
                bevel_metres=0,
            )
        return job, catalog

    def test_registered_mesh_and_uv_material_are_verified(self):
        job, _ = self.job_and_catalog()
        validate_scene_job(job)
        verified = verify_material_assets(job, self.root)
        self.assertEqual(verified["shape.npz"], self.binding["sha256"])
        loaded = load_normalized_mesh(self.binding, self.root)
        np.testing.assert_array_equal(loaded["uv"], self.uv)

    def test_invalid_mesh_arrays_are_rejected(self):
        for kind in ("indices", "uv", "bounds", "degenerate"):
            with self.subTest(kind=kind):
                original = (self.vertices.copy(), self.triangles.copy(), self.uv.copy())
                if kind == "indices":
                    self.triangles[0, 0] = 99
                elif kind == "uv":
                    self.uv[0, 0] = float("nan")
                elif kind == "bounds":
                    self.vertices *= 1000
                else:
                    self.triangles[0] = [0, 0, 0]
                self.save_mesh()
                with self.assertRaises(ValueError):
                    load_normalized_mesh(self.binding, self.root)
                self.vertices, self.triangles, self.uv = original

    def test_unsafe_paths_and_changed_bytes_fail(self):
        with self.assertRaises(ValueError):
            load_normalized_mesh(dict(self.binding, path="../outside.npz"), self.root)
        (self.root / "shape.npz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_normalized_mesh(self.binding, self.root)

    def test_mesh_contract_requires_version_and_matching_uv_asset(self):
        job, _ = self.job_and_catalog()
        for change in ("version", "asset", "projection"):
            with self.subTest(change=change):
                altered = copy.deepcopy(job)
                if change == "version":
                    altered["schema_version"] = "dubaidata.scene/1.0"
                elif change == "asset":
                    altered["materials"]["uv_fixture"]["texture_set"][
                        "asset_id"
                    ] = "different"
                else:
                    altered["materials"]["uv_fixture"]["texture_set"][
                        "projection"
                    ] = "BOX"
                with self.assertRaises(ValueError):
                    validate_scene_job(altered)

    def test_surface_binding_preserves_geometry_and_rejects_unknown_materials(self):
        job, catalog = self.job_and_catalog()
        config = dict(
            schema_version="dubaidata.surface-bindings/1.0",
            surfaces={"concrete": dict(asset_id="fixture.mesh", base_color_mix=0.5)},
        )
        changed = bind_surfaces(job, catalog, job["asset_catalog_sha256"], config)
        self.assertEqual(changed["scenes"], job["scenes"])
        config["surfaces"]["typo"] = config["surfaces"].pop("concrete")
        with self.assertRaises(ValueError):
            bind_surfaces(job, catalog, job["asset_catalog_sha256"], config)


class IndependentMeshRayTests(unittest.TestCase):
    def test_two_sided_triangle_uses_axial_parameter(self):
        triangles = np.array([[[-1, -1, 10], [1, -1, 10], [0, 1, 10]]], dtype=float)
        self.assertAlmostEqual(
            triangle_distance(np.zeros(3), np.array([0, 0, 1]), triangles), 10
        )
        self.assertAlmostEqual(
            triangle_distance(np.zeros(3), np.array([0, 0, 2]), triangles[:, ::-1]), 5
        )
        self.assertTrue(
            np.isinf(triangle_distance(np.zeros(3), np.array([1, 0, 1]), triangles))
        )

    def test_nearest_cuboid_wins_without_mesh_or_simulator(self):
        job = dict(camera=dict(intrinsics=[20, 20, 16, 16]))
        capture = dict(
            optical_camera_to_world=np.eye(4).tolist(),
            objects=[
                dict(
                    instance_id=1,
                    center_metres=[0, 0, 10.5],
                    dimensions_metres=[4, 4, 1],
                ),
                dict(
                    instance_id=2,
                    center_metres=[0, 0, 8.5],
                    dimensions_metres=[2, 2, 1],
                ),
            ],
        )
        result = scene_intersections(job, capture, None, [(16, 16)])[0]
        self.assertEqual(result["instance_id"], 2)
        self.assertEqual(result["depth_axial_metres"], 8)
