"""Source-bound draft preloading and technical validation of known human fixtures."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash
from src.validation.artifacts import verify_snapshot
from src.validation.draft_review import prepare_draft_review
from src.validation.reference_labels import import_references
from src.validation.review_server import create_server


class NativeReferenceReviewTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing declared 4x2 customer fixture without inheriting its tests.
        from test_benchmark_preparation import BenchmarkPreparationTests

        self.fixture = BenchmarkPreparationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.base = self.fixture.base
        for split in ("train", "validation"):
            submission = self.fixture.submission(split)
            submission["reviewer_kind"] = "ai"
            self.fixture.write_submission(submission, split + ".json")
        self.fixture.import_targets("ai_drafts")
        self.drafts = self.base / "ai_drafts"
        self.packet = self.base / "review_packet"
        prepare_draft_review(
            self.fixture.packet,
            self.fixture.packet_sha,
            self.drafts,
            file_hash(self.drafts / "profile.json"),
            self.fixture.definition_path,
            self.packet,
            self.fixture.raw,
        )
        self.sha = file_hash(self.packet / "profile.json")

    def reviewed(self, split="train"):
        entry = next(
            row
            for row in json.loads((self.packet / "references.json").read_text())
            if row["binding"]["split"] == split
        )
        result = self.fixture.submission(split)
        result.update(
            entry["binding"], reviewer_kind="human", reviewer="Known fixture reviewer"
        )
        return result

    def test_all_drafts_are_exactly_preloaded_with_parent_lineage(self):
        profile = verify_snapshot(self.packet, self.sha)
        self.assertEqual(profile["selected_frames"], {"train": 1, "validation": 1})
        self.assertFalse(profile["training_eligible"])
        originals = {
            row["sample_id"]: row
            for row in [
                json.loads(line)
                for line in (self.drafts / "references.jsonl").read_text().splitlines()
            ]
        }
        for entry in json.loads((self.packet / "references.json").read_text()):
            binding = entry["binding"]
            draft = originals[binding["sample_id"]]
            self.assertEqual(
                file_hash(self.packet / entry["folder"] / "candidate.png"),
                draft["target_sha256"],
            )
            self.assertEqual(
                binding["parent_submission_sha256"], draft["submission_sha256"]
            )
            self.assertEqual(
                binding["parent_packet_profile_sha256"], self.fixture.packet_sha
            )
            mask = np.array(Image.open(self.packet / entry["folder"] / "candidate.png"))
            self.assertIn(255, np.unique(mask))

    def test_stale_draft_binding_and_raw_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare_draft_review(
                self.fixture.packet,
                self.fixture.packet_sha,
                self.drafts,
                "0" * 64,
                self.fixture.definition_path,
                self.base / "stale",
                self.fixture.raw,
            )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            prepare_draft_review(
                self.fixture.packet,
                self.fixture.packet_sha,
                self.drafts,
                file_hash(self.drafts / "profile.json"),
                self.fixture.definition_path,
                self.fixture.raw / "review",
                self.fixture.raw,
            )

    def server(self):
        server = create_server(
            self.packet, self.sha, self.base / "review_state", self.fixture.raw, 0
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def test_save_validates_before_selection_and_preserves_versions(self):
        server = self.server()
        store = server.review_store
        self.assertEqual(store.progress()["reviewed"], [])
        bad = self.reviewed()
        bad["mask_rle"] = [[1, 9]]
        with self.assertRaises(ValueError):
            store.save(bad)
        self.assertEqual(store.progress()["reviewed"], [])
        first = self.reviewed()
        store.save(first)
        changed = copy.deepcopy(first)
        changed["mask_rle"] = [[1, 1], [0, 7]]
        store.save(changed)
        self.assertEqual(store.load("train/frame")["mask_rle"], changed["mask_rle"])
        self.assertEqual(len(list((store.state / "history").rglob("*.json"))), 2)
        ai = self.reviewed()
        ai["reviewer_kind"] = "ai"
        with self.assertRaisesRegex(ValueError, "confirmed reference"):
            store.save(ai)
        self.assertFalse(store.validate_cohort()["benchmark_reference_ready"])
        store.save(self.reviewed("validation"))
        self.assertTrue(store.validate_cohort()["benchmark_reference_ready"])
        verify_snapshot(self.packet, self.sha)

    def test_http_requires_local_origin_and_token(self):
        server = self.server()
        url = f"http://127.0.0.1:{server.server_port}"
        with urlopen(url + "/api/session") as response:
            session = json.load(response)
        payload = json.dumps(self.reviewed()).encode()
        wrong = Request(
            url + "/api/reviews",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Origin": "https://invalid.example",
            },
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(wrong)
        self.assertEqual(caught.exception.code, 403)
        valid = Request(
            url + "/api/reviews",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Origin": url,
                "X-Review-Token": session["token"],
            },
        )
        with urlopen(valid) as response:
            self.assertEqual(json.load(response)["reviewed"], ["train/frame"])

    def test_name_prefill_changes_only_served_view_not_review_or_packet(self):
        server = create_server(
            self.packet,
            self.sha,
            self.base / "prefill_state",
            self.fixture.raw,
            0,
            'Fixture "reviewer" <name>',
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        entry = json.loads((self.packet / "references.json").read_text())[0]
        path = self.packet / entry["folder"] / "index.html"
        original = path.read_bytes()
        url = f"http://127.0.0.1:{server.server_port}/{entry['folder']}/index.html"
        with urlopen(url) as response:
            html = response.read().decode("utf-8")
        self.assertIn('value="Fixture &quot;reviewer&quot; &lt;name&gt;"', html)
        self.assertNotIn('id="confirmed" type="checkbox" checked', html)
        self.assertEqual(server.review_store.progress()["reviewed"], [])
        self.assertEqual(path.read_bytes(), original)
        verify_snapshot(self.packet, self.sha)

    @unittest.skipUnless(
        shutil.which("node"), "Node is needed for editor behavior checks"
    )
    def test_native_editor_class_colours_zoom_undo_and_human_export(self):
        entry = json.loads((self.packet / "references.json").read_text())[0]
        candidate = (
            np.array(Image.open(self.packet / entry["folder"] / "candidate.png"))
            .ravel()
            .tolist()
        )
        result = subprocess.run(
            [
                shutil.which("node"),
                str(Path(__file__).with_name("native_editor_smoke.js")),
                str(self.packet / entry["folder"] / "index.html"),
                json.dumps(candidate),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        submission = json.loads(result.stdout)
        self.assertEqual(submission["reviewer_kind"], "human")
        self.assertEqual(
            submission["parent_draft_profile_sha256"],
            file_hash(self.drafts / "profile.json"),
        )
        folder = self.base / "human_export"
        folder.mkdir()
        (folder / "review.json").write_text(json.dumps(submission))
        imported = import_references(
            self.packet,
            folder,
            self.base / "human_import",
            self.sha,
            source_root=self.fixture.raw,
        )
        self.assertEqual(
            imported["qualified_reference_counts"], {"train": 1, "validation": 0}
        )


if __name__ == "__main__":
    unittest.main()
