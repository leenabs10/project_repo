"""Background-only supervision is explicit, finite and matched across arms."""

import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from src.training.auxiliary_inputs import (
    balanced_auxiliary_loss,
    prepare_auxiliary_donors,
    select_auxiliary_donor,
    validate_auxiliary_sampling,
)
from src.validation.scene_dataset import validate_arrays
from src.validation.scene_quality import sample_role_pixels
from src.adapters.tasks.goods_scene import validate_goods_scene
from src.core.scene import validate_scene_job
from tests.test_scene_pipeline import fixture_job
from tests import test_scene_training as training_fixture
from src.validation.artifacts import artifact_hashes, write_json
from src.adapters.storage.local import file_hash
from src.validation.scene_dataset import export_dataset
from src.validation.scene_admission import admit_scene_dataset
from src.pipeline.scene_benchmark import run_arm, compare_arms

POLICY = "background_only_zero_foreground_mass"


class BackgroundAuxiliaryTests(unittest.TestCase):
    def test_negative_loss_mass_ignore_and_gradients(self):
        logits = torch.tensor(
            [[[[0.0, 1.0, 3.0]], [[2.0, -1.0, 4.0]]]], requires_grad=True
        )
        target = torch.tensor([[[0, 0, 255]]])
        with self.assertRaises(ValueError):
            balanced_auxiliary_loss(logits, target, 0.15)
        loss = balanced_auxiliary_loss(
            logits, target, 0.15, missing_class_policy=POLICY
        )
        expected = 0.85 * torch.nn.functional.cross_entropy(
            logits[:, :, :, :2], target[:, :, :2]
        )
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(
            torch.equal(
                logits.grad[:, :, :, 2], torch.zeros_like(logits.grad[:, :, :, 2])
            )
        )
        self.assertGreater(logits.grad[0, 1, 0, 0].item(), 0)
        for values in (
            torch.ones_like(target),
            torch.full_like(target, 255),
            torch.full_like(target, 2),
        ):
            with self.subTest(), self.assertRaises(ValueError):
                balanced_auxiliary_loss(
                    logits, values, 0.15, missing_class_policy=POLICY
                )
        bad = logits.detach().clone()
        bad[0, 0, 0, 2] = float("nan")
        with self.assertRaises(ValueError):
            balanced_auxiliary_loss(bad, target, 0.15, missing_class_policy=POLICY)

    def test_mixed_loss_is_unchanged_under_new_policy(self):
        logits = torch.randn(1, 2, 3, 4, requires_grad=True)
        target = torch.tensor([[[0, 1, 255, 0], [1, 0, 1, 0], [0, 1, 0, 1]]])
        old = balanced_auxiliary_loss(logits, target, 0.15)
        new = balanced_auxiliary_loss(logits, target, 0.15, missing_class_policy=POLICY)
        torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_donors_and_sixty_input_presence_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index in range(4):
                mask = np.zeros((4, 4), np.uint8)
                if index >= 2:
                    mask[:2] = 1
                Image.fromarray(mask).save(root / f"{index}.png")
                rows.append(
                    dict(
                        sample_id=f"train/{index}",
                        split="train",
                        target_path=f"{index}.png",
                    )
                )
            original = copy.deepcopy(rows)
            cfg = dict(
                experiment_kind="real_only",
                generation_seed=7331,
                training_resolution=[4, 4],
                auxiliary_training=dict(
                    strategy="preserve_real_exposure",
                    interval_updates=2,
                    loss_weight=0.2,
                    foreground_fraction=0.25,
                    missing_class_policy=POLICY,
                ),
                auxiliary_sampling=dict(strategy="balanced_presence_cycle"),
            )
            donors = prepare_auxiliary_donors(root, rows, None, [], cfg)
            self.assertEqual(rows, original)
            selected = [
                select_auxiliary_donor(donors, cfg, i) for i in range(2, 121, 2)
            ]
            self.assertEqual(
                Counter(r["auxiliary_target_kind"] for _, r in selected),
                {"background_only": 30, "mixed": 30},
            )
            self.assertEqual(
                set(Counter(r["sample_id"] for _, r in selected).values()), {15}
            )
            self.assertEqual(
                selected,
                [
                    select_auxiliary_donor(list(reversed(donors)), cfg, i)
                    for i in range(2, 121, 2)
                ],
            )
            with self.assertRaises(ValueError):
                select_auxiliary_donor(
                    [d for d in donors if d[1]["auxiliary_target_kind"] == "mixed"],
                    cfg,
                    2,
                )
            legacy = copy.deepcopy(cfg)
            legacy.pop("auxiliary_sampling")
            legacy["auxiliary_training"].pop("missing_class_policy")
            self.assertEqual(
                len(prepare_auxiliary_donors(root, rows, None, [], legacy)), 2
            )
            invalid = copy.deepcopy(cfg)
            invalid["auxiliary_training"].pop("missing_class_policy")
            with self.assertRaises(ValueError):
                validate_auxiliary_sampling(invalid)

    def test_negative_native_arrays_require_explicit_visible_class_contract(self):
        job = fixture_job()
        job["required_visible_class_ids"] = [0]
        validate_scene_job(job)
        validate_goods_scene(job)
        for required in ([1], [0, 0], [True], [2]):
            invalid = copy.deepcopy(job)
            invalid["required_visible_class_ids"] = required
            with self.assertRaises(ValueError):
                validate_goods_scene(invalid)
        rgb = np.zeros((18, 32, 3), np.uint8)
        depth = np.ones((18, 32), np.float32)
        valid = np.ones((18, 32), bool)
        target = np.zeros((18, 32), np.uint8)
        ids = np.ones((18, 32), np.uint16)
        objects = [dict(instance_id=1, class_id=0)]
        args = (rgb, depth, valid, target, ids, job["camera"], objects)
        with self.assertRaises(ValueError):
            validate_arrays(*args)
        validate_arrays(*args, required_visible_class_ids=[0])
        target[0, 0] = 1
        with self.assertRaises(ValueError):
            validate_arrays(*args, required_visible_class_ids=[0])

    def test_negative_geometry_sampler_reports_zero_goods_and_same_ray_budget(self):
        ids = np.ones((40, 40), np.uint16)
        ids[:, 20:] = 2
        objects = [
            dict(instance_id=1, role="equipment", mesh={}),
            dict(instance_id=2, role="structure"),
        ]
        with self.assertRaises(ValueError):
            sample_role_pixels(ids, objects)
        pixels, counts = sample_role_pixels(ids, objects, allow_background_only=True)
        self.assertEqual(len(set(pixels)), 256)
        self.assertEqual(counts, dict(goods_mesh=0, equipment_mesh=64, structure=192))

    def test_negative_export_admission_and_three_seed_matched_training(self):
        """Known planes exercise actual export, admission, training and reloads."""
        fixture = training_fixture.SceneTrainingTests
        fixture.setUpClass()
        try:
            root = fixture.base / "negative_integration"
            root.mkdir()
            references = root / "references"
            shutil.copytree(fixture.references, references)
            rows = [
                json.loads(line)
                for line in (references / "references.jsonl").read_text().splitlines()
            ]
            train = [r for r in rows if r["split"] == "train"]
            for row in train[:2]:
                path = references / row["target_path"]
                values = np.array(Image.open(path))
                values[:] = 0
                Image.fromarray(values).save(path)
                row["target_sha256"] = file_hash(path)
            (references / "references.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            reference_profile = json.loads((references / "profile.json").read_text())
            reference_profile["artifact_sha256"] = artifact_hashes(references)
            write_json(references / "profile.json", reference_profile)
            reference_sha = file_hash(references / "profile.json")

            job = json.loads((fixture.dataset / "generation_job.json").read_text())
            positive = copy.deepcopy(job["scenes"][0])
            positive["sample_id"] = "mixed_plane"
            negative = copy.deepcopy(positive)
            negative["sample_id"] = "background_plane"
            for obj in negative["objects"]:
                obj.update(role="equipment", class_id=0)
            job.update(required_visible_class_ids=[0], scenes=[negative, positive])
            worker = root / "worker"
            worker.mkdir()
            records = []
            for scene in job["scenes"]:
                destination = worker / scene["sample_id"]
                shutil.copytree(fixture.base / "worker/known_plane", destination)
                if scene["sample_id"] == "background_plane":
                    np.save(
                        destination / "semantic_ids.npy",
                        np.zeros((32, 32), np.uint8),
                        allow_pickle=False,
                    )
                records.append(
                    dict(
                        sample_id=scene["sample_id"],
                        artifact_sha256={
                            p.relative_to(worker).as_posix(): file_hash(p)
                            for p in destination.iterdir()
                        },
                        render_seconds=0,
                        geometry=dict(
                            interior_rays=256,
                            instance_agreement=1,
                            depth_error_metres_p50_p95_max=[0, 0, 0],
                        ),
                    )
                )
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
                    records=records,
                ),
            )
            dataset = root / "dataset"
            export_dataset(job, worker, dataset)
            dataset_sha = file_hash(dataset / "profile.json")
            evidence = root / "evidence"
            evidence.mkdir()
            shutil.copyfile(
                fixture.evidence / "plane_fixture.json", evidence / "plane_fixture.json"
            )
            write_json(
                evidence / "profile.json",
                dict(
                    dataset_profile_sha256=dataset_sha,
                    job_sha256=job_sha,
                    captures=2,
                    independent_rays=512,
                    independent_instance_agreement=[1, 1],
                    independent_depth_p95_metres=[0, 0],
                    reference_artifacts_changed=0,
                    artifact_sha256=artifact_hashes(evidence),
                ),
            )
            decisions = root / "decisions.json"
            write_json(
                decisions,
                dict(
                    schema_version="dubaidata.scene-decisions/1.0",
                    dataset_profile_sha256=dataset_sha,
                    job_sha256=job_sha,
                    assessor=dict(
                        name="Known negative/mixed planes", method="known_fixture"
                    ),
                    samples=[
                        dict(
                            sample_id=s["sample_id"],
                            appearance="accept",
                            labels="accept",
                            notes="Analytic fixture only",
                        )
                        for s in job["scenes"]
                    ],
                ),
            )
            admission = root / "admission"
            admit_scene_dataset(
                dataset,
                dataset_sha,
                evidence,
                file_hash(evidence / "profile.json"),
                decisions,
                references,
                reference_sha,
                fixture.task,
                admission,
                fixture=True,
            )
            fg = bg = 0
            for row in train:
                target = np.array(
                    Image.open(references / row["target_path"]).resize(
                        tuple(fixture.config["training_resolution"]),
                        Image.Resampling.NEAREST,
                    )
                )
                fg += int((target == 1).sum())
                bg += int((target == 0).sum())
            arms = []
            for kind in ("real_only", "real_plus_scene"):
                config = copy.deepcopy(fixture.config)
                config.update(
                    experiment_kind=kind,
                    epochs=4,
                    model_seeds=[17, 29, 43],
                    selection_metric=None,
                    auxiliary_sampling=dict(strategy="balanced_presence_cycle"),
                    auxiliary_training=dict(
                        strategy="preserve_real_exposure",
                        interval_updates=2,
                        loss_weight=0.2,
                        foreground_fraction=fg / (fg + bg),
                        missing_class_policy=POLICY,
                    ),
                )
                if kind == "real_only":
                    config.pop("synthetic_mixing")
                else:
                    config["synthetic_mixing"] = dict(strategy="auxiliary_only")
                path = root / (kind + ".json")
                write_json(path, config)
                arm = root / kind
                run_arm(
                    references,
                    reference_sha,
                    fixture.task,
                    path,
                    None,
                    arm,
                    synthetic=admission if kind != "real_only" else None,
                    synthetic_sha256=(
                        file_hash(admission / "profile.json")
                        if kind != "real_only"
                        else None
                    ),
                    device="cpu",
                    fixture=True,
                )
                for seed in (17, 29, 43):
                    result = json.loads((arm / f"seed_{seed}/profile.json").read_text())
                    self.assertEqual(result["primary_real_presentations"], 16)
                    self.assertEqual(result["optimizer_updates"], 4)
                    self.assertEqual(
                        result["auxiliary_target_presentations"],
                        {"background_only": 1, "mixed": 1},
                    )
                arms.append(arm)
            compared = compare_arms(
                arms[0],
                file_hash(arms[0] / "profile.json"),
                arms[1],
                file_hash(arms[1] / "profile.json"),
                references,
                reference_sha,
                root / "comparison",
                fixture=True,
            )
            self.assertTrue(compared["matched_comparison_complete"])
            self.assertFalse(compared["candidate_promoted"])
        finally:
            fixture.tearDownClass()


if __name__ == "__main__":
    unittest.main()
