"""Admission/target fixtures keep rendered labels separate from real references."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash
from src.training.goods_benchmark import (
    fixture_run,
    load_references,
    native_metrics,
    target_tensor,
    train_run,
)
from src.training.scene_inputs import load_scene_inputs, select_scene_training_sample
from src.validation.artifacts import artifact_hashes, write_json
from src.validation.scene_admission import admit_scene_dataset, require_snapshot_digest
from src.validation.scene_dataset import export_dataset
from tests.test_scene_pipeline import fixture_job


class SceneTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.fixture = cls.base / "reference_fixture"
        fixture_run(cls.fixture)
        cls.references = cls.fixture / "references"
        profile = json.loads((cls.references / "profile.json").read_text())
        profile["manifest_sha256"] = "a" * 64
        write_json(cls.references / "profile.json", profile)
        cls.reference_sha = file_hash(cls.references / "profile.json")
        cls.task = cls.fixture / "task.json"
        cls.real_rows = load_references(
            cls.references, cls.reference_sha, cls.task, fixture=True
        )
        cls.config = json.loads((cls.fixture / "config.json").read_text())
        cls.config.update(
            experiment_kind="real_plus_scene",
            epochs=2,
            checkpoint_policy="final_epoch",
            generation_seed=7331,
            synthetic_mixing=dict(
                strategy="replace_training_slot", substitution_probability=0.5
            ),
        )
        job = fixture_job()
        job["camera"].update(width=32, height=32, intrinsics=[20, 20, 16, 16])
        capture = job["scenes"][0]
        capture.update(
            sample_id="known_plane",
            scenario="Known analytic plane split into background and goods",
            optical_camera_to_world=np.eye(4).tolist(),
        )
        capture["objects"] = [
            dict(
                instance_id=identifier,
                asset_id="builtin.metric_cuboid/1.0",
                role="equipment" if identifier == 1 else "goods",
                class_id=identifier - 1,
                center_metres=[-4 if identifier == 1 else 4, 0, 8.5],
                dimensions_metres=[8, 16, 1],
                material="paper",
                bevel_metres=0,
            )
            for identifier in (1, 2)
        ]
        job["scenes"] = [capture]
        worker = cls.base / "worker"
        folder = worker / "known_plane"
        folder.mkdir(parents=True)
        semantic = np.zeros((32, 32), np.uint8)
        semantic[:, 16:] = 1
        rgb = np.full((32, 32, 3), [45, 75, 110], np.uint8)
        rgb[semantic == 1] = [175, 135, 80]
        Image.fromarray(rgb).save(folder / "rgb.png")
        for name, array in (
            ("depth_axial_m", np.full((32, 32), 8, np.float32)),
            ("depth_valid", np.ones((32, 32), bool)),
            ("semantic_ids", semantic),
            ("instance_ids", semantic.astype(np.uint16) + 1),
        ):
            np.save(folder / f"{name}.npy", array, allow_pickle=False)
        job_sha = hashlib.sha256(json.dumps(job, indent=2).encode()).hexdigest()
        write_json(
            worker / "worker_profile.json",
            dict(
                status="rendered_geometry_passed",
                backend=job["backend"],
                job_sha256=job_sha,
                training_eligible=False,
                physical_sensor_realism_certified=False,
                fixture=dict(maximum_error_metres=0),
                records=[
                    dict(
                        sample_id="known_plane",
                        artifact_sha256={
                            path.relative_to(worker).as_posix(): file_hash(path)
                            for path in folder.iterdir()
                        },
                        render_seconds=0,
                        geometry=dict(
                            interior_rays=256,
                            instance_agreement=1,
                            depth_error_metres_p50_p95_max=[0, 0, 0],
                        ),
                    )
                ],
            ),
        )
        cls.dataset = cls.base / "dataset"
        export_dataset(job, worker, cls.dataset)
        cls.dataset_sha = file_hash(cls.dataset / "profile.json")
        cls.evidence = cls.base / "evidence"
        cls.evidence.mkdir()
        # Exact front-plane intersections provide independent known-fixture rays;
        # this is test evidence, never a customer renderer qualification.
        rays = []
        for v in range(2, 30):
            for u in range(2, 30):
                if u == 16:
                    continue
                rays.append(
                    dict(u=u, v=v, depth_metres=8, instance_id=1 if u < 16 else 2)
                )
        write_json(
            cls.evidence / "plane_fixture.json",
            dict(purpose="known_fixture_only", rays=rays[:256]),
        )
        write_json(
            cls.evidence / "profile.json",
            dict(
                dataset_profile_sha256=cls.dataset_sha,
                job_sha256=job_sha,
                captures=1,
                independent_rays=256,
                independent_instance_agreement=[1],
                independent_depth_p95_metres=[0],
                reference_artifacts_changed=0,
                artifact_sha256=artifact_hashes(cls.evidence),
            ),
        )
        cls.evidence_sha = file_hash(cls.evidence / "profile.json")
        cls.decisions = cls.base / "decisions.json"
        write_json(
            cls.decisions,
            dict(
                schema_version="dubaidata.scene-decisions/1.0",
                dataset_profile_sha256=cls.dataset_sha,
                job_sha256=job_sha,
                assessor=dict(name="Known plane fixture", method="known_fixture"),
                samples=[
                    dict(
                        sample_id="known_plane",
                        appearance="accept",
                        labels="accept",
                        notes="Known fixture only",
                    )
                ],
            ),
        )
        cls.admission = cls.base / "admission"
        cls.admit(cls.decisions, cls.admission)

    @classmethod
    def admit(cls, decisions, output, fixture=True):
        return admit_scene_dataset(
            cls.dataset,
            cls.dataset_sha,
            cls.evidence,
            cls.evidence_sha,
            decisions,
            cls.references,
            cls.reference_sha,
            cls.task,
            output,
            fixture=fixture,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def load(self, root=None, fixture=True):
        root = root or self.admission
        return load_scene_inputs(
            root,
            file_hash(root / "profile.json"),
            self.references,
            self.reference_sha,
            self.real_rows,
            self.config,
            fixture=fixture,
        )

    def test_scene_selection_uses_its_own_target_and_is_deterministic(self):
        scenes = self.load()
        self.assertTrue(
            (self.admission / "quality_evidence/plane_fixture.json").is_file()
        )
        source = self.real_rows[0]
        for epoch in range(50):
            selected = select_scene_training_sample(
                self.references, source, self.admission, scenes, self.config, epoch
            )
            self.assertEqual(
                selected,
                select_scene_training_sample(
                    self.references, source, self.admission, scenes, self.config, epoch
                ),
            )
            if selected[2]:
                root, row, _ = selected
                target = target_tensor(root, row, self.config)
                self.assertEqual(int((target == 255).sum()), 0)
                self.assertNotEqual(
                    int((target == 1).sum()),
                    int(
                        (target_tensor(self.references, source, self.config) == 1).sum()
                    ),
                )
                break
        else:
            self.fail("Known schedule did not select any scene input")
        with self.assertRaisesRegex(ValueError, "training-only"):
            select_scene_training_sample(
                self.references,
                self.real_rows[-1],
                self.admission,
                scenes,
                self.config,
                0,
            )

    def test_pending_decisions_fail_before_output_allocation(self):
        decisions = json.loads(self.decisions.read_text())
        decisions["samples"][0]["appearance"] = "pending"
        path = self.base / "pending.json"
        write_json(path, decisions)
        output = self.base / "pending_attempt"
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.admit(path, output)
        self.assertFalse(output.exists())

    def test_fixture_cannot_be_used_as_customer_admission(self):
        with self.assertRaises(ValueError):
            self.load(fixture=False)
        output = self.base / "customer_attempt"
        with self.assertRaises(ValueError):
            self.admit(self.decisions, output, fixture=False)
        self.assertFalse(output.exists())

    def test_eligible_manifest_tampering_is_rejected_even_when_rehashed(self):
        root = self.base / "changed_manifest"
        shutil.copytree(self.admission, root)
        path = root / "eligible_manifest.jsonl"
        row = json.loads(path.read_text())
        row["split"] = "validation"
        path.write_text(json.dumps(row) + "\n")
        profile = json.loads((root / "profile.json").read_text())
        profile["artifact_sha256"] = artifact_hashes(root)
        write_json(root / "profile.json", profile)
        with self.assertRaisesRegex(ValueError, "Eligible scene manifest"):
            self.load(root)

    def test_known_fixture_train_reload_and_final_policy(self):
        configuration = self.base / "scene_training.json"
        write_json(configuration, self.config)
        with patch(
            "src.training.goods_benchmark.native_metrics", wraps=native_metrics
        ) as evaluate:
            result = train_run(
                self.references,
                self.reference_sha,
                self.task,
                configuration,
                None,
                self.base / "scene_training",
                17,
                "cpu",
                fixture=True,
                synthetic=self.admission,
                synthetic_sha=file_hash(self.admission / "profile.json"),
            )
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertEqual(result["total_training_presentations"], 8)
        self.assertGreater(result["synthetic_presentations"], 0)
        self.assertEqual(
            result["real_presentations"] + result["synthetic_presentations"], 8
        )
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["strict_reload"], "passed")
        self.assertEqual(result["mixing_unit"], "RGB_and_rendered_target")
        history = json.loads((self.base / "scene_training/history.json").read_text())
        self.assertIsNone(history[0]["validation"])
        self.assertIsNotNone(history[1]["validation"])


class SceneDigestTests(unittest.TestCase):
    def test_mandatory_digests_reject_empty_and_malformed_values(self):
        for value in (None, "", "a" * 63, "G" * 64):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    require_snapshot_digest(value)
        require_snapshot_digest("a" * 64)
