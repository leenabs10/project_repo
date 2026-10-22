"""Coverage fixtures prevent floating, overlapping or silently missing loads."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from src.synthesis.warehouse_scene import build_job
from src.synthesis.warehouse_scenarios import validate_coverage
from src.validation.goods_scene_coverage import measure_target_coverage
from tests.test_scene_pipeline import fixture_job


def configuration():
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/warehouse_goods_coverage.json"
        ).read_text(encoding="utf-8")
    )


def resolve(config):
    return build_job(
        dict(status="passed", split="train", manifest_sha256="a" * 64),
        "b" * 64,
        fixture_job()["camera"],
        coverage=config,
    )


class WarehouseCoverageTests(unittest.TestCase):
    def test_cases_are_deterministic_and_keep_equipment_background(self):
        job = resolve(configuration())
        self.assertEqual(job, resolve(configuration()))
        self.assertEqual(len(job["scenes"]), 7)
        for scene in job["scenes"]:
            self.assertTrue(
                all(
                    obj["class_id"] == int(obj["role"] == "goods")
                    for obj in scene["objects"]
                )
            )

    def test_invalid_controls_fail(self):
        changes = (
            lambda c: c["cases"][0].update(camera_x_metres=float("nan")),
            lambda c: c["cases"][0].update(camera_x_metres=True),
            lambda c: c["cases"][0].update(camera_x_metres=2),
            lambda c: c["cases"][0].update(sample_id="../escape"),
            lambda c: c["cases"][0].update(unknown_control=1),
            lambda c: c["cases"][1]["loads"][0].update(support="floating"),
            lambda c: c["cases"][1]["loads"][0].update(center_xy_metres=[0, 0]),
            lambda c: c["cases"][1]["loads"][0].update(dimensions_metres=[2, 1, 1]),
            lambda c: c["cases"][1]["loads"][0].update(material="unknown"),
            lambda c: c["cases"][1].update(sample_id=c["cases"][0]["sample_id"]),
        )
        for mutate in changes:
            with self.subTest(mutation=mutate):
                config = configuration()
                mutate(config)
                with self.assertRaises(ValueError):
                    resolve(config)

    def test_overlapping_goods_bodies_fail(self):
        config = configuration()
        loads = config["cases"][1]["loads"]
        loads.append(copy.deepcopy(loads[0]))
        with self.assertRaisesRegex(ValueError, "bodies overlap"):
            validate_coverage(config)

    def test_container_penetration_fails_during_resolution(self):
        config = configuration()
        config["cases"][4]["loads"][0]["center_xy_metres"] = [1.1, 1.2]
        with self.assertRaisesRegex(ValueError, "penetrates equipment"):
            resolve(config)

    def test_empty_case_does_not_mean_empty_warehouse(self):
        scene = resolve(configuration())["scenes"][0]
        self.assertGreater(sum(obj["role"] == "goods" for obj in scene["objects"]), 0)
        self.assertEqual(configuration()["cases"][0]["loads"], [])

    def test_known_projected_plane_distinguishes_equipment_occlusion(self):
        config = {
            "schema_version": "dubaidata.warehouse-coverage/1.0",
            "cases": [
                dict(
                    sample_id="fixture",
                    objective="Known partial occlusion",
                    camera_x_metres=0,
                    loads=[
                        dict(
                            support="floor",
                            center_xy_metres=[0, 2],
                            dimensions_metres=[1, 0.5, 1],
                            material="paper",
                        )
                    ],
                )
            ],
        }
        job = dict(
            camera=dict(
                width=64,
                height=64,
                intrinsics=[200, 200, 32, 32],
                near_metres=0.05,
                far_metres=60,
            )
        )
        capture = dict(
            sample_id="fixture",
            optical_camera_to_world=[
                [1, 0, 0, 0],
                [0, 0, 1, -6],
                [0, -1, 0, 0.5],
                [0, 0, 0, 1],
            ],
            objects=[
                dict(
                    instance_id=1,
                    role="goods",
                    center_metres=[0, 2, 0.5],
                    dimensions_metres=[1, 0.5, 1],
                    material="paper",
                ),
                dict(instance_id=2, role="equipment"),
            ],
        )
        depth = np.full((64, 64), 7.75, dtype=np.float32)
        instances = np.ones((64, 64), dtype=np.uint16)
        clear = measure_target_coverage(job, config, capture, depth, instances)[
            "targets"
        ][0]
        self.assertEqual(clear["visible_surface_fraction"], 1)
        depth[:, :32] = 6
        instances[:, :32] = 2
        hidden = measure_target_coverage(job, config, capture, depth, instances)[
            "targets"
        ][0]
        self.assertGreater(hidden["equipment_occlusion_fraction"], 0.4)
        depth[:] = 6
        with self.assertRaisesRegex(ValueError, "insufficient visible"):
            measure_target_coverage(job, config, capture, depth, instances)


if __name__ == "__main__":
    unittest.main()
