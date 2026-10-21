"""Explicit AI refinement geometry, lineage and human-approval isolation."""

import json
import unittest

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash
from src.validation.artifacts import artifact_hashes, verify_snapshot
from src.validation.guided_review import (
    apply_polygon_edits,
    encode_runs,
    prepare_guided_review,
)
from src.validation.reference_labels import decode_mask_runs
from src.validation.review_server import ReviewStore


class PolygonRefinementTests(unittest.TestCase):
    def test_edit_changes_only_explicit_pixels_and_roundtrips(self):
        previous = np.zeros((4, 4), dtype=np.uint8)
        edit = {
            "class_id": 1,
            "polygon": [[1, 1], [2, 1], [2, 2], [1, 2]],
            "reason": "Known fixture goods square",
        }
        result = apply_polygon_edits(previous, [edit], [4, 4])
        expected = np.zeros((4, 4), dtype=np.uint8)
        expected[1:3, 1:3] = 1
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(previous, np.zeros((4, 4), dtype=np.uint8))
        np.testing.assert_array_equal(
            decode_mask_runs(encode_runs(result), 4, 4), expected
        )

    def test_invalid_polygons_ids_and_coordinate_grids_fail(self):
        good = {"class_id": 1, "polygon": [[0, 0], [2, 0], [2, 2]], "reason": "Fixture"}
        for change in (
            {"class_id": 7},
            {"class_id": True},
            {"polygon": [[0, 0], [2, 0]]},
            {"polygon": [[0, 0], [-1, 0], [2, 2]]},
            {"polygon": [[0, 0], [float("nan"), 0], [2, 2]]},
            {"uncertain_boundary_pixels": -1},
            {"reason": ""},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                apply_polygon_edits(
                    np.zeros((4, 4), np.uint8), [{**good, **change}], [4, 4]
                )
        with self.assertRaises(ValueError):
            apply_polygon_edits(np.zeros((4, 4), np.uint8), [good], [0, 4])


class GuidedPacketTests(unittest.TestCase):
    def setUp(self):
        from test_native_reference_review import NativeReferenceReviewTests

        self.fixture = NativeReferenceReviewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.base = self.fixture.base
        self.parent = self.fixture.packet
        self.entries = json.loads((self.parent / "references.json").read_text())
        self.specification = {
            "reviewer_kind": "ai",
            "coordinate_size": [4, 2],
            "samples": {
                entry["binding"]["sample_id"]: {
                    "edits": [
                        {
                            "class_id": 0,
                            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                            "reason": "Known background",
                        }
                    ],
                    "notes": ['Check <script>alert("not executable")</script>'],
                }
                for entry in self.entries
            },
            "examples": [
                {
                    "sample_id": self.entries[0]["binding"]["sample_id"],
                    "title": "Known example",
                    "description": "Fixture only",
                }
            ],
        }
        self.plan = self.base / "refinements.json"

    def prepare(self, name="guided"):
        self.plan.write_text(json.dumps(self.specification))
        return prepare_guided_review(
            self.parent,
            self.fixture.sha,
            self.plan,
            self.base / name,
            self.fixture.fixture.raw,
        )

    def test_packet_retains_sources_parent_bytes_and_ai_only_status(self):
        before = file_hash(self.parent / "profile.json")
        profile = self.prepare()
        packet = self.base / "guided"
        self.assertFalse(profile["training_eligible"])
        self.assertEqual(profile["qualified_reference_frames"], 0)
        self.assertEqual(profile["test_sensor_files_opened"], 0)
        self.assertEqual(file_hash(self.parent / "profile.json"), before)
        verify_snapshot(self.parent, before)
        verify_snapshot(packet, file_hash(packet / "profile.json"))
        store = ReviewStore(
            packet,
            file_hash(packet / "profile.json"),
            self.base / "guided_state",
            self.fixture.fixture.raw,
        )
        self.assertEqual(store.progress()["reviewed"], [])
        for original, refined in zip(
            self.entries, json.loads((packet / "references.json").read_text())
        ):
            folder = packet / refined["folder"]
            self.assertEqual(
                file_hash(folder / "rgb.png"), original["binding"]["source_rgb_sha256"]
            )
            self.assertEqual(
                file_hash(folder / "previous_candidate.png"),
                original["binding"]["candidate_sha256"],
            )
            self.assertEqual(
                refined["binding"]["previous_candidate_sha256"],
                original["binding"]["candidate_sha256"],
            )
            submission = json.loads((folder / "ai_draft.json").read_text())
            self.assertFalse(submission["reviewed"])
            self.assertTrue(submission["human_review_pending"])
            with self.assertRaisesRegex(ValueError, "confirmed reference"):
                store.save(submission)
            with Image.open(folder / "candidate.png") as image:
                np.testing.assert_array_equal(
                    np.array(image), decode_mask_runs(submission["mask_rle"], 4, 2)
                )
            html = (folder / "index.html").read_text()
            self.assertIn("&lt;script&gt;", html)
            self.assertNotIn('<script>alert("not executable")</script>', html)
            self.assertIn("../../guide.html", html)
        self.assertIn("Known example", (packet / "guide.html").read_text())
        index = (packet / "index.html").read_text()
        self.assertIn("async function refreshProgress()", index)
        self.assertIn("window.addEventListener('pageshow', refreshProgress)", index)

    def test_sparse_parent_can_be_refined_but_final_coverage_limit_still_applies(self):
        # Sparse extension hints may fail the review limit; correcting them must
        # be possible without allowing an excessive-ignore final submission.
        entries = json.loads((self.parent / "references.json").read_text())
        entry = entries[0]
        sparse = np.full((2, 4), 255, dtype=np.uint8)
        sparse[0, 0] = 0
        candidate = self.parent / entry["folder"] / "candidate.png"
        Image.fromarray(sparse).save(candidate)
        entry["binding"]["candidate_sha256"] = file_hash(candidate)
        (self.parent / "references.json").write_text(json.dumps(entries))
        # Exercise a parent with no index JavaScript, as in the six-frame packet.
        (self.parent / "index.html").write_text(
            "<!doctype html><title>Sparse hints</title>"
        )
        profile = json.loads((self.parent / "profile.json").read_text())
        profile["artifact_sha256"] = artifact_hashes(self.parent)
        (self.parent / "profile.json").write_text(json.dumps(profile))
        self.fixture.sha = file_hash(self.parent / "profile.json")
        result = self.prepare("sparse_corrected")
        self.assertFalse(result["training_eligible"])
        html = (self.base / "sparse_corrected/index.html").read_text()
        self.assertIn("async function refreshProgress()", html)
        self.specification["samples"][entry["binding"]["sample_id"]]["edits"] = []
        with self.assertRaisesRegex(ValueError, "ignored-pixel limit"):
            self.prepare("sparse_uncorrected")

    def test_membership_and_protected_outputs_are_rejected(self):
        removed = self.specification["samples"].pop(
            self.entries[0]["binding"]["sample_id"]
        )
        with self.assertRaisesRegex(ValueError, "membership"):
            self.prepare("missing")
        self.specification["samples"][self.entries[0]["binding"]["sample_id"]] = removed
        self.plan.write_text(json.dumps(self.specification))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            prepare_guided_review(
                self.parent,
                self.fixture.sha,
                self.plan,
                self.fixture.fixture.raw / "bad",
                self.fixture.fixture.raw,
            )

    def test_out_of_cohort_example_fails_without_success_profile(self):
        self.specification["examples"][0]["sample_id"] = "test/not-in-packet"
        with self.assertRaises((ValueError, StopIteration)):
            self.prepare("bad_example")
        profile = json.loads((self.base / "bad_example/profile.json").read_text())
        self.assertEqual(profile["status"], "failed")
        self.assertFalse(profile["training_eligible"])


if __name__ == "__main__":
    unittest.main()
