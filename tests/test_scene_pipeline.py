"""Analytic scene fixtures cover taxonomy, coordinate and admission boundaries."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash
from src.core.contracts import validate_job
from src.core.scene import validate_scene_job
from src.adapters.tasks.goods_scene import validate_goods_scene
from src.pipeline.scene import run_scene
from src.synthesis.warehouse_scene import build_job
from src.validation.scene_dataset import export_dataset, validate_arrays


def fixture_job():
    return build_job(
        dict(status="passed", split="train", manifest_sha256="a" * 64),
        "b" * 64,
        dict(
            width=32,
            height=18,
            intrinsics=[20, 20, 16, 9],
            near_metres=0.05,
            far_metres=60,
            model="pinhole_zero_distortion",
        ),
    )


class SceneContractTests(unittest.TestCase):
    def test_resolved_plan_is_deterministic_and_equipment_is_background(self):
        job = fixture_job()
        self.assertEqual(job, fixture_job())
        self.assertEqual(validate_job(job), job)
        for capture in job["scenes"]:
            for obj in capture["objects"]:
                self.assertEqual(obj["class_id"], int(obj["role"] == "goods"))

    def test_invalid_contracts_fail(self):
        mutations = (
            lambda j: j["conditioning"].update(profile_split="validation"),
            lambda j: j.update(required_modalities=["rgb", "lidar"]),
            lambda j: j.update(units="millimetres"),
            lambda j: j["camera"].update(intrinsics=[0, 20, 16, 9]),
            lambda j: j["scenes"][0].update(sample_id="../outside"),
            lambda j: j["scenes"][0]["objects"][0].update(class_id=1),
            lambda j: j["scenes"][0]["objects"][0].update(dimensions_metres=[1, -1, 1]),
            lambda j: j["scenes"][0]["objects"][1].update(instance_id=1),
            lambda j: j["scenes"][0]["objects"][0].update(asset_id="unknown_mesh"),
            lambda j: j["scenes"][0]["optical_camera_to_world"][0].__setitem__(0, -1),
            lambda j: j["render"].update(seed=-1),
            lambda j: j.update(command="arbitrary_worker"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                job = fixture_job()
                mutate(job)
                with self.assertRaises((ValueError, TypeError)):
                    validate_goods_scene(job)

    def test_unsupported_backend_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "attempt"
            job = fixture_job()
            job["backend"] = "blenderproc"
            with self.assertRaises(NotImplementedError):
                run_scene(job, {}, output, Path(temporary))
            self.assertFalse(output.exists())

    def test_metric_material_bounds_are_enforced(self):
        for field, value in (
            ("scale_per_metre", [0, 1, 1]),
            ("color_variation", 0.9),
            ("roughness_variation", -0.1),
            ("bump_metres", 0.1),
        ):
            with self.subTest(field=field):
                job = fixture_job()
                job["materials"]["concrete"]["detail"][field] = value
                with self.assertRaises(ValueError):
                    validate_scene_job(job)
        job = fixture_job()
        job["materials"]["diffuser"]["emission_strength"] = 101
        with self.assertRaises(ValueError):
            validate_scene_job(job)

    def test_legacy_scene_materials_remain_supported(self):
        job = fixture_job()
        for surface in job["materials"].values():
            surface.pop("detail", None)
            surface.pop("emission_strength", None)
        validate_scene_job(job)

    def test_fixture_illumination_is_below_opaque_housing(self):
        job = fixture_job()
        for scene in job["scenes"]:
            diffusers = [
                obj for obj in scene["objects"] if obj["material"] == "diffuser"
            ]
            self.assertEqual(len(diffusers), len(scene["lights"]))
            for diffuser, light in zip(diffusers, scene["lights"], strict=True):
                self.assertEqual(diffuser["class_id"], 0)
                bottom = (
                    diffuser["center_metres"][2] - diffuser["dimensions_metres"][2] / 2
                )
                self.assertLess(light["center_metres"][2], bottom)

    def test_penetrating_load_is_rejected(self):
        job = fixture_job()
        load = next(
            obj
            for obj in job["scenes"][1]["objects"]
            if obj["center_metres"] == [0.8, 1.8, 0.45]
        )
        load["center_metres"][1] = 1.2
        with self.assertRaisesRegex(ValueError, "penetrates equipment"):
            validate_goods_scene(job)

    def test_core_accepts_another_declared_taxonomy(self):
        job = fixture_job()
        job["taxonomy"] = {"0": "environment", "2": "customer_material"}
        for scene in job["scenes"]:
            for obj in scene["objects"]:
                if obj["class_id"] == 1:
                    obj["class_id"] = 2
        validate_scene_job(job)
        with self.assertRaisesRegex(ValueError, "binary taxonomy"):
            validate_goods_scene(job)


class ScenePayloadTests(unittest.TestCase):
    def setUp(self):
        self.camera = fixture_job()["camera"]
        self.objects = [
            dict(instance_id=1, class_id=0),
            dict(instance_id=2, class_id=1),
        ]
        self.rgb = np.full((18, 32, 3), 127, dtype=np.uint8)
        self.depth = np.full((18, 32), 10, dtype=np.float32)
        self.valid = np.ones((18, 32), dtype=np.bool_)
        self.semantic = np.zeros((18, 32), dtype=np.uint8)
        self.semantic[:, 16:] = 1
        self.instances = self.semantic.astype(np.uint16) + 1

    def check(self):
        validate_arrays(
            self.rgb,
            self.depth,
            self.valid,
            self.semantic,
            self.instances,
            self.camera,
            self.objects,
        )

    def test_known_metric_plane(self):
        self.check()

    def test_wrong_units_are_rejected(self):
        self.depth *= 1000
        with self.assertRaisesRegex(ValueError, "metric"):
            self.check()

    def test_invalids_are_not_silently_repaired(self):
        self.valid[0, 0] = False
        with self.assertRaises(ValueError):
            self.check()

    def test_equipment_cannot_be_goods(self):
        self.semantic[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "taxonomy"):
            self.check()

    def test_unknown_instance_is_rejected(self):
        self.instances[0, 0] = 99
        with self.assertRaisesRegex(ValueError, "absent"):
            self.check()

    def test_export_roundtrip_and_checksum_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = fixture_job()
            job["scenes"] = [job["scenes"][0]]
            capture = job["scenes"][0]
            # Replace the fixture's payload identities by actual resolved IDs.
            equipment = next(
                obj for obj in capture["objects"] if obj["role"] == "equipment"
            )
            goods = next(obj for obj in capture["objects"] if obj["role"] == "goods")
            instances = np.where(
                self.semantic == 1, goods["instance_id"], equipment["instance_id"]
            ).astype(np.uint16)
            folder = root / capture["sample_id"]
            folder.mkdir()
            Image.fromarray(self.rgb).save(folder / "rgb.png")
            for name, array in (
                ("depth_axial_m", self.depth),
                ("depth_valid", self.valid),
                ("semantic_ids", self.semantic),
                ("instance_ids", instances),
            ):
                np.save(folder / f"{name}.npy", array, allow_pickle=False)
            hashes = {
                path.relative_to(root).as_posix(): file_hash(path)
                for path in folder.iterdir()
            }
            profile = dict(
                status="rendered_geometry_passed",
                backend=job["backend"],
                job_sha256=hashlib.sha256(
                    json.dumps(job, indent=2).encode("utf-8")
                ).hexdigest(),
                training_eligible=False,
                physical_sensor_realism_certified=False,
                fixture=dict(maximum_error_metres=0.00001),
                records=[
                    dict(
                        sample_id=capture["sample_id"],
                        artifact_sha256=hashes,
                        render_seconds=1,
                        geometry=dict(
                            interior_rays=256,
                            instance_agreement=1,
                            depth_error_metres_p50_p95_max=[0, 0, 0],
                        ),
                    )
                ],
            )
            (root / "worker_profile.json").write_text(
                json.dumps(profile), encoding="utf-8"
            )
            exported = export_dataset(job, root, root / "export")
            self.assertFalse(exported["training_eligible"])
            self.assertEqual(exported["captures"], 1)
            first_record = json.loads(
                (root / "export/samples.jsonl").read_text(encoding="utf-8")
            )
            # Exercise the actual Windows document bytes as well as the LF
            # fixture; normalized metadata must bind to either exact document.
            windows_document = (
                json.dumps(job, indent=2).replace("\n", "\r\n").encode("utf-8")
            )
            profile["job_sha256"] = hashlib.sha256(windows_document).hexdigest()
            (root / "worker_profile.json").write_text(
                json.dumps(profile), encoding="utf-8"
            )
            export_dataset(job, root, root / "windows_export")
            windows_record = json.loads(
                (root / "windows_export/samples.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(
                windows_record["source_metadata"]["job_sha256"], profile["job_sha256"]
            )
            self.assertEqual(
                file_hash(root / "windows_export/generation_job.json"),
                profile["job_sha256"],
            )
            self.assertEqual(
                json.loads(
                    (root / "windows_export/generation_job.json").read_text(
                        encoding="utf-8"
                    )
                ),
                job,
            )
            changed_job = copy.deepcopy(job)
            changed_job["render"]["seed"] += 1
            with self.assertRaisesRegex(ValueError, "different resolved scene job"):
                export_dataset(changed_job, root, root / "wrong_job_export")
            self.assertFalse((root / "wrong_job_export").exists())
            profile["job_sha256"] = hashlib.sha256(
                json.dumps(changed_job, indent=2).encode("utf-8")
            ).hexdigest()
            (root / "worker_profile.json").write_text(
                json.dumps(profile), encoding="utf-8"
            )
            export_dataset(changed_job, root, root / "second_export")
            second_record = json.loads(
                (root / "second_export/samples.jsonl").read_text(encoding="utf-8")
            )
            self.assertNotEqual(first_record["dataset_id"], second_record["dataset_id"])
            self.assertEqual(first_record["group_id"], second_record["group_id"])
            # Restore the original binding before exercising payload corruption.
            profile["job_sha256"] = hashlib.sha256(
                json.dumps(job, indent=2).encode("utf-8")
            ).hexdigest()
            (root / "worker_profile.json").write_text(
                json.dumps(profile), encoding="utf-8"
            )
            with Image.open(
                root / "export" / capture["sample_id"] / "instance_ids.png"
            ) as image:
                np.testing.assert_array_equal(np.asarray(image), instances)
            (folder / "rgb.png").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                export_dataset(job, root, root / "corrupted_export")


if __name__ == "__main__":
    unittest.main()
