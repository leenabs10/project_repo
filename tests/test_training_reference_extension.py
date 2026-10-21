"""Keep annotation expansion training-only, source-bound and unqualified."""

import json
from unittest.mock import patch
import unittest

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash
from src.validation.artifacts import verify_snapshot
from src.validation.training_reference_extension import prepare_training_extension


class TrainingReferenceExtensionTests(unittest.TestCase):
    def setUp(self):
        from test_benchmark_preparation import BenchmarkPreparationTests

        self.fixture = BenchmarkPreparationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        fixture = self.fixture
        # An existing validation-only snapshot leaves the training fixture new.
        fixture.write_submission(fixture.submission("validation"), "validation.json")
        fixture.import_targets("previous")
        self.previous = fixture.base / "previous"
        self.spec = fixture.base / "hints.json"
        self.write_spec("train/frame")

    def write_spec(self, sample_id):
        self.spec.write_text(
            json.dumps(
                {
                    "coordinate_size": [4, 2],
                    "samples": [
                        {
                            "sample_id": sample_id,
                            "polygons": [
                                {
                                    "class_id": 0,
                                    "polygon": [[0, 0], [1, 0], [1, 2], [0, 2]],
                                    "reason": "Known fixture background",
                                }
                            ],
                        }
                    ],
                }
            )
        )

    def prepare(self):
        fixture = self.fixture
        return prepare_training_extension(
            fixture.adapter,
            fixture.definition_path,
            self.previous,
            file_hash(self.previous / "profile.json"),
            self.spec,
            fixture.base / "extension",
        )

    def test_unknown_pixels_stay_ignored_and_prior_snapshot_unchanged(self):
        before = file_hash(self.previous / "profile.json")
        self.prepare()
        output = self.fixture.base / "extension"
        profile = verify_snapshot(output)
        entry = json.loads((output / "references.json").read_text())[0]
        folder = output / entry["folder"]
        mask = np.asarray(Image.open(folder / "candidate.png"))
        np.testing.assert_array_equal(mask, [[0, 0, 255, 255], [0, 0, 255, 255]])
        self.assertEqual(
            file_hash(folder / "rgb.png"), file_hash(self.fixture.raw / "train_rgb.png")
        )
        self.assertFalse(profile["training_eligible"])
        self.assertEqual(profile["qualified_reference_frames"], 0)
        self.assertEqual(profile["development_sensor_files_opened"], 0)
        self.assertEqual(entry["binding"]["parent_reference_profile_sha256"], before)
        self.assertEqual(file_hash(self.previous / "profile.json"), before)

    def test_nontraining_selection_rejected_before_payload_read(self):
        for sample_id in ("test/frame", "validation/frame", "missing/frame"):
            with self.subTest(sample_id=sample_id):
                self.write_spec(sample_id)
                with patch(
                    "src.validation.training_reference_extension.source_path",
                    side_effect=AssertionError("Payload read before split check"),
                ):
                    with self.assertRaisesRegex(ValueError, "unique new training"):
                        self.prepare()

    def test_stale_parent_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare_training_extension(
                self.fixture.adapter,
                self.fixture.definition_path,
                self.previous,
                "0" * 64,
                self.spec,
                self.fixture.base / "extension",
            )
