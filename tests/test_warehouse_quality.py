"""Bounded asset conversion and authored coverage preserve geometry and task meaning."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.adapters.generation.gltf_assets import convert_opaque_gltf
from src.adapters.generation.mesh_assets import load_normalized_mesh
from src.synthesis.warehouse_quality import (
    build_quality_job,
    validate_quality_configuration,
)
from src.validation.scene_quality import (
    sample_role_pixels,
    stable_interior,
    verify_scene_quality,
)


def source_fixture():
    """A known tetrahedron has nonzero metric bounds and a simple UV correspondence."""
    vertices = np.array(
        [[-1, -2, -3], [1, -2, -3], [-1, 2, -3], [-1, -2, 3]], dtype="<f4"
    )
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype="<f4")
    indices = np.array([0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2], dtype="<u2")
    parts = [vertices.tobytes(), uv.tobytes(), indices.tobytes()]
    views, offset = [], 0
    for part in parts:
        views.append(dict(buffer=0, byteOffset=offset, byteLength=len(part)))
        offset += len(part)
    document = dict(
        asset=dict(version="2.0"),
        nodes=[dict(mesh=0)],
        buffers=[dict(byteLength=offset, uri="fixture.bin")],
        bufferViews=views,
        materials=[dict(alphaMode="OPAQUE")],
        meshes=[
            dict(
                primitives=[
                    dict(
                        attributes=dict(POSITION=0, TEXCOORD_0=1), indices=2, material=0
                    )
                ]
            )
        ],
        accessors=[
            dict(bufferView=0, componentType=5126, count=4, type="VEC3"),
            dict(bufferView=1, componentType=5126, count=4, type="VEC2"),
            dict(bufferView=2, componentType=5123, count=12, type="SCALAR"),
        ],
    )
    return document, b"".join(parts)


class GltfConversionTests(unittest.TestCase):
    def test_metric_axes_uvs_and_triangle_winding_are_preserved(self):
        document, binary = source_fixture()
        arrays, evidence = convert_opaque_gltf(document, binary)
        self.assertEqual(evidence["original_extent_metres"], [2, 6, 4])
        np.testing.assert_array_equal(arrays["uv"][:, 1], [1, 1, 0, 0])
        np.testing.assert_array_equal(arrays["vertices"][0], [-0.5, 0.5, -0.5])
        self.assertEqual(evidence["triangle_count"], 4)
        # The glTF Y-up -> Z-up mapping is a proper rotation, not a reflection.
        self.assertEqual(np.linalg.det(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])), 1)

    def test_unsupported_transparency_transforms_and_accessor_ranges_fail(self):
        for kind in ("alpha", "rotation", "sparse", "range", "indices", "extension"):
            with self.subTest(kind=kind):
                document, binary = source_fixture()
                if kind == "alpha":
                    document["materials"][0]["alphaMode"] = "BLEND"
                elif kind == "rotation":
                    document["nodes"][0]["rotation"] = [0, 0, 0, 1]
                elif kind == "sparse":
                    document["accessors"][0]["sparse"] = {}
                elif kind == "range":
                    document["bufferViews"][0]["byteLength"] -= 1
                elif kind == "indices":
                    binary = binary[:-2] + np.array([99], dtype="<u2").tobytes()
                else:
                    document["extensionsUsed"] = ["KHR_draco_mesh_compression"]
                with self.assertRaises(ValueError):
                    convert_opaque_gltf(document, binary)

    def test_disk_budget_is_checked_before_mesh_hashing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "oversized.npz"
            with path.open("wb") as stream:
                stream.truncate(32 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(ValueError, "byte budget"):
                load_normalized_mesh(dict(path=path.name, sha256="0" * 64), folder)


def planner_inputs():
    profile = dict(split="train", status="passed", manifest_sha256="a" * 64)
    camera = dict(
        width=1280,
        height=720,
        intrinsics=[600, 600, 640, 360],
        near_metres=0.05,
        far_metres=60,
        model="pinhole_zero_distortion",
    )
    catalog = dict(assets={})
    for name, extent in (
        ("cardboard_box_01", [0.387, 0.516, 0.342]),
        ("cement_bag", [0.462, 0.700, 0.180]),
        ("plastic_crate_01", [0.298, 0.408, 0.264]),
    ):
        for variant in ("front", "back"):
            identifier = f"polyhaven.{name}.{variant}.4k"
            catalog["assets"][identifier] = dict(
                asset_id=identifier,
                tile_size_metres=[1, 1, 1],
                conversion=dict(original_extent_metres=extent),
                mesh=dict(
                    asset_id=identifier, path=f"{name}/{variant}.npz", sha256="b" * 64
                ),
                maps={
                    role: dict(
                        path=f"{name}/{role}.png", sha256="c" * 64, color_space=space
                    )
                    for role, space in (
                        ("base_color", "sRGB"),
                        ("roughness", "Non-Color"),
                    )
                },
            )
    for name in ("wooden_planks", "hangar_concrete_floor"):
        catalog["assets"][f"polyhaven.{name}.4k"] = dict(
            asset_id=f"polyhaven.{name}.4k",
            tile_size_metres=[1, 1, 1],
            maps=copy.deepcopy(
                catalog["assets"]["polyhaven.cement_bag.front.4k"]["maps"]
            ),
        )
    config = json.loads(
        (Path(__file__).parents[1] / "configs/warehouse_goods_quality.json").read_text()
    )
    return profile, "d" * 64, camera, catalog, "e" * 64, config


class WarehouseQualityTests(unittest.TestCase):
    def test_portable_plan_is_repeatable_and_does_not_mutate_inputs(self):
        inputs = planner_inputs()
        before = copy.deepcopy(inputs)
        first = build_quality_job(*inputs)
        self.assertEqual(first, build_quality_job(*inputs))
        self.assertEqual(inputs, before)
        self.assertEqual(len(first["scenes"]), 8)
        self.assertGreater(
            len({json.dumps(c["optical_camera_to_world"]) for c in first["scenes"]}), 3
        )
        self.assertEqual(first["conditioning"]["profile_split"], "train")
        for capture in first["scenes"]:
            for obj in capture["objects"]:
                if "mesh" not in obj:
                    continue
                asset = inputs[3]["assets"][obj["mesh"]["asset_id"]]
                scales = (
                    np.asarray(obj["dimensions_metres"])
                    / asset["conversion"]["original_extent_metres"]
                )
                np.testing.assert_allclose(scales, scales[0])
                if "plastic_crate_01" in obj["mesh"]["asset_id"]:
                    self.assertEqual((obj["role"], obj["class_id"]), ("equipment", 0))
                else:
                    self.assertEqual((obj["role"], obj["class_id"]), ("goods", 1))

    def test_support_overhang_and_separate_load_penetration_are_rejected(self):
        for kind in ("overhang", "overlap", "floor_collision"):
            with self.subTest(kind=kind):
                inputs = planner_inputs()
                config = inputs[-1]
                if kind == "overhang":
                    config["cases"][1]["loads"][0]["xy"] = [-0.26, -1.6]
                elif kind == "overlap":
                    config["cases"][3]["loads"][1]["xy"] = config["cases"][3]["loads"][
                        0
                    ]["xy"].copy()
                else:
                    config["cases"][5]["loads"][1]["xy"] = [0.35, 3.8]
                with self.assertRaises(ValueError):
                    build_quality_job(*inputs)

    def test_split_and_configuration_boundaries_fail_before_rendering(self):
        for kind in ("split", "seed", "unknown", "camera", "equipment_load"):
            with self.subTest(kind=kind):
                inputs = planner_inputs()
                if kind == "split":
                    inputs[0]["split"] = "validation"
                elif kind == "seed":
                    inputs[-1]["stock_seed"] = True
                elif kind == "unknown":
                    inputs[-1]["cases"][0]["extra"] = True
                elif kind == "camera":
                    inputs[-1]["cases"][0]["camera"][1] = 0
                else:
                    inputs[-1]["cases"][1]["loads"][0]["asset"] = "plastic_crate_01"
                with self.assertRaises(ValueError):
                    build_quality_job(*inputs)




class RoleRaySamplingTests(unittest.TestCase):
    def test_reference_binding_cannot_be_omitted(self):
        with self.assertRaisesRegex(ValueError, "explicit SHA256"):
            verify_scene_quality(
                "missing", "a" * 64, "missing", "missing", None, "unused"
            )

    def test_sampling_balances_roles_and_excludes_boundaries(self):
        ids = np.full((64, 64), 3, dtype=np.uint16)
        ids[:32, :32] = 1
        ids[:32, 32:] = 2
        objects = [
            dict(instance_id=1, role="goods", mesh={}),
            dict(instance_id=2, role="equipment", mesh={}),
            dict(instance_id=3, role="structure"),
        ]
        pixels, groups = sample_role_pixels(ids, objects)
        self.assertEqual(
            groups, {"goods_mesh": 96, "equipment_mesh": 64, "structure": 96}
        )
        self.assertEqual(len(set(pixels)), 256)
        self.assertEqual((pixels, groups), sample_role_pixels(ids, objects))
        interior = stable_interior(ids)
        self.assertTrue(all(interior[v, u] for u, v in pixels))
        self.assertEqual({int(ids[v, u]) for u, v in pixels[:96]}, {1})
        self.assertEqual({int(ids[v, u]) for u, v in pixels[96:160]}, {2})
        self.assertEqual({int(ids[v, u]) for u, v in pixels[160:]}, {3})

    def test_missing_equipment_coverage_is_not_silently_replaced(self):
        ids = np.ones((64, 64), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "equipment_mesh"):
            sample_role_pixels(ids, [dict(instance_id=1, role="goods", mesh={})])
