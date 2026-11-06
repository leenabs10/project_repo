"""Collection isolation, acceptance and payload integrity using analytic fixtures."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash
from src.training.goods_benchmark import target_tensor, train_run
from src.training.scene_collections import create_scene_collection
from src.training.scene_inputs import load_scene_inputs
from src.validation.artifacts import artifact_hashes, write_json
import tests.test_scene_training as fixture_support


class SceneCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.first = type("FirstPlane", (fixture_support.SceneTrainingTests,), {})
        cls.second = type("SecondPlane", (fixture_support.SceneTrainingTests,), {
            "fixture_scene_id": "second_plane", "fixture_color_shift": 1,
        })
        cls.first.setUpClass()
        cls.second.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.second.tearDownClass()
        cls.first.tearDownClass()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.members = [(data.admission, file_hash(data.admission / "profile.json"))
                        for data in (self.first, self.second)]

    def create(self, members=None, name="collection"):
        data = self.first
        output = self.base / name
        create_scene_collection(
            self.members if members is None else members,
            data.references, data.reference_sha, data.real_rows, data.config,
            output, fixture=True,
        )
        return output

    def load(self, root, *, fixture=True, references_sha=None):
        data = self.first
        return load_scene_inputs(
            root, file_hash(root / "profile.json"), data.references,
            references_sha or data.reference_sha, data.real_rows, data.config,
            fixture=fixture,
        )

    def rehash(self, root):
        profile = json.loads((root / "profile.json").read_text())
        profile["artifact_sha256"] = artifact_hashes(root)
        write_json(root / "profile.json", profile)

    def test_portable_deterministic_collection_preserves_inputs_and_targets(self):
        before = {path: file_hash(path / "profile.json") for path, _ in self.members}
        root = self.create()
        reverse = self.create(list(reversed(self.members)), "reverse")
        self.assertEqual(file_hash(root / "profile.json"), file_hash(reverse / "profile.json"))
        rows = self.load(root)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertTrue(row["rgb_path"].startswith("members/"))
            self.assertEqual(file_hash(root / row["rgb_path"]), row["source_rgb_sha256"])
            target = target_tensor(root, row, self.first.config)
            with Image.open(root / row["target_path"]) as image:
                native = np.array(image)
            self.assertEqual(native.shape, (32, 32))
            self.assertEqual(int((native == 1).sum()), 512)
            self.assertEqual(set(target.unique().tolist()), {0, 1})
        moved = self.base / "moved"
        shutil.copytree(root, moved)
        self.assertEqual(self.load(moved), rows)
        self.assertEqual(before, {path: file_hash(path / "profile.json")
                                  for path, _ in self.members})

    def test_duplicates_pending_mismatches_and_nested_inputs_rejected(self):
        for label, members in (
            ("duplicate", [self.members[0], self.members[0]]),
            ("empty", []),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.create(members, label)
            self.assertFalse((self.base / label).exists())
        changed = self.base / "pending"
        shutil.copytree(self.first.admission, changed)
        profile = json.loads((changed / "profile.json").read_text())
        profile["qualification_status"] = "pending"
        write_json(changed / "profile.json", profile)
        with self.assertRaises(ValueError):
            self.create([(changed, file_hash(changed / "profile.json"))], "rejected")
        self.assertFalse((self.base / "rejected").exists())
        root = self.create()
        with self.assertRaises(ValueError):
            self.create([(root, file_hash(root / "profile.json"))], "nested")
        with self.assertRaises(ValueError):
            self.load(root, fixture=False)
        with self.assertRaises(ValueError):
            self.load(root, references_sha="f" * 64)

    def test_tampered_flattened_manifest_and_member_payload_rejected(self):
        root = self.create()
        manifest = root / "eligible_manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        rows[0]["target_path"] = "../outside.npy"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.rehash(root)
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            self.load(root)
        root = self.create(name="payload")
        row = self.load(root)[0]
        with (root / row["target_path"]).open("ab") as stream:
            stream.write(b"changed")
        self.rehash(root)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.load(root)

    def test_rehashed_member_path_escape_rejected(self):
        root = self.create()
        profile = json.loads((root / "profile.json").read_text())
        profile["members"][0]["path"] = "../outside"
        write_json(root / "profile.json", profile)
        with self.assertRaisesRegex(ValueError, "member path"):
            self.load(root)

    def test_distinct_jobs_with_identical_rgb_are_rejected_before_allocation(self):
        third = type("DuplicateAppearance", (fixture_support.SceneTrainingTests,), {
            "fixture_scene_id": "third_plane",
        })
        third.setUpClass()
        self.addCleanup(third.tearDownClass)
        members = [self.members[0],
                   (third.admission, file_hash(third.admission / "profile.json"))]
        with self.assertRaisesRegex(ValueError, "Duplicate.*source_rgb_sha256"):
            self.create(members, "duplicate_rgb")
        self.assertFalse((self.base / "duplicate_rgb").exists())

    def test_output_overlap_and_member_cohort_mismatch_rejected(self):
        data = self.first
        with self.assertRaisesRegex(ValueError, "overlaps"):
            create_scene_collection(
                self.members, data.references, data.reference_sha, data.real_rows,
                data.config, data.admission / "nested", fixture=True,
            )
        changed = self.base / "wrong_cohort"
        shutil.copytree(data.admission, changed)
        profile = json.loads((changed / "profile.json").read_text())
        profile["manifest_sha256"] = "b" * 64
        write_json(changed / "profile.json", profile)
        with self.assertRaisesRegex(ValueError, "matching real cohort"):
            self.create([(changed, file_hash(changed / "profile.json"))], "mismatch")
        self.assertFalse((self.base / "mismatch").exists())

    def test_collection_reaches_training_with_unchanged_budget_and_strict_reload(self):
        root = self.create()
        data = self.first
        config = copy.deepcopy(data.config)
        path = self.base / "config.json"
        write_json(path, config)
        result = train_run(
            data.references, data.reference_sha, data.task, path, None,
            self.base / "training", 17, "cpu", fixture=True,
            synthetic=root, synthetic_sha=file_hash(root / "profile.json"),
        )
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertEqual(result["total_training_presentations"], 8)
        self.assertGreater(result["synthetic_presentations"], 0)
        self.assertEqual(result["strict_reload"], "passed")
