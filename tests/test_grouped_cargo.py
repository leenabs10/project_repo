"""Cargo groups preserve source assets and reject unsafe/support-invalid plans."""

import copy
from pathlib import Path
import tempfile
import unittest

from src.pipeline.scene import run_scene
from src.synthesis.grouped_cargo import build_grouped_cargo
from src.synthesis.warehouse_balanced import build_balanced_job
from tests.test_balanced_scene import BalancedSceneTests
from tests.test_scene_pipeline import fixture_job


class GroupedCargoTests(unittest.TestCase):
    def inputs(self):
        job = build_balanced_job(*BalancedSceneTests().inputs())
        scene = job["scenes"][0]
        template = next(obj for obj in scene["objects"] if obj["role"] == "goods" and "mesh" in obj)
        support = copy.deepcopy(next(obj for obj in scene["objects"] if obj["role"] == "structure" and obj["material"] == "wood"))
        support.update(instance_id=max(obj["instance_id"] for obj in scene["objects"]) + 1,
                       center_metres=[20, 20, 1], dimensions_metres=[4, 4, 0.07])
        scene["objects"].append(support)
        config = dict(schema_version="dubaidata.grouped-cargo/1.0", job_id="grouped_fixture",
                      cases=[dict(sample_id=scene["sample_id"], groups=[dict(
                          support_instance_id=support["instance_id"], template_instance_ids=[template["instance_id"]],
                          counts_xyz=[1, 2, 2], gap_xy_metres=[0.04, 0.12],
                          center_xy_offset_metres=[0, 0])])])
        return job, config

    def test_sources_dimensions_roles_and_unselected_scenes_are_preserved(self):
        source, config = self.inputs()
        preserved = copy.deepcopy(source)
        job, lineage = build_grouped_cargo(source, config)
        self.assertEqual(source, preserved)
        self.assertEqual(job["scenes"][1:], source["scenes"][1:])
        old_objects = source["scenes"][0]["objects"]
        self.assertEqual(job["scenes"][0]["objects"][:len(old_objects)], old_objects)
        self.assertEqual(len(lineage), 4)
        lookup = {obj["instance_id"]: obj for obj in old_objects}
        for obj, entry in zip(job["scenes"][0]["objects"][len(old_objects):], lineage):
            template = lookup[entry["template_instance_id"]]
            self.assertEqual({key: value for key, value in obj.items() if key not in {"instance_id", "center_metres"}},
                             {key: value for key, value in template.items() if key not in {"instance_id", "center_metres"}})
        self.assertEqual(build_grouped_cargo(source, config), (job, lineage))

    def test_overhang_overlap_unknown_fields_and_background_templates_fail(self):
        for kind in ("overhang", "offset_overhang", "overlap", "unknown", "template", "duplicate", "budget", "gap"):
            with self.subTest(kind=kind):
                source, config = self.inputs()
                group = config["cases"][0]["groups"][0]
                if kind == "overhang":
                    source["scenes"][0]["objects"][-1]["dimensions_metres"][:2] = [0.1, 0.1]
                elif kind == "offset_overhang":
                    group["center_xy_offset_metres"] = [3.0, 0]
                elif kind == "overlap":
                    config["cases"][0]["groups"].append(copy.deepcopy(group))
                elif kind == "unknown":
                    group["transparent_wrapping"] = True
                elif kind == "template":
                    group["template_instance_ids"] = [group["support_instance_id"]]
                elif kind == "duplicate":
                    config["cases"].append(copy.deepcopy(config["cases"][0]))
                elif kind == "budget":
                    group["counts_xyz"] = [4, 4, 4]
                else:
                    group["gap_xy_metres"] = [float("nan"), 0.1]
                with self.assertRaises(ValueError):
                    build_grouped_cargo(source, config)

    def test_runtime_worker_limits_fail_before_output_is_created(self):
        for threads in (0, 17, True, "4"):
            with self.subTest(threads=threads), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "attempt"
                with self.assertRaisesRegex(ValueError, "CPU threads"):
                    run_scene(fixture_job(), {"cpu_threads": threads}, output, Path(temporary))
                self.assertFalse(output.exists())
