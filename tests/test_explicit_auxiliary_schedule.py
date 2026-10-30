"""Frozen auxiliary sequences preserve exposure when source IDs are versioned."""

from collections import Counter
import copy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.training.auxiliary_inputs import (
    prepare_auxiliary_donors,
    select_auxiliary_donor,
    validate_auxiliary_sampling,
)


def configuration():
    """Use a generic eight-donor, 120-update fixture, independent of any dataset."""
    return {
        "experiment_kind": "real_plus_scene",
        "generation_seed": 7331,
        "synthetic_mixing": {"strategy": "auxiliary_only"},
        "training_resolution": [4, 4],
        "epochs": 15,
        "batch_size": 4,
        "auxiliary_training": {
            "strategy": "preserve_real_exposure",
            "interval_updates": 2,
            "loss_weight": 0.2,
            "foreground_fraction": 0.25,
            "missing_class_policy": "background_only_zero_foreground_mass",
        },
    }


def donor_rows(prefix="old-job"):
    return [
        {
            "sample_id": f"{prefix}/capture-{index}",
            "split": "train",
            "target_path": f"{index}.png",
            "auxiliary_target_kind": "background_only" if index < 4 else "mixed",
        }
        for index in range(8)
    ]


def legacy_presence_events(rows):
    """Independently reproduce the existing hash inputs and presence-cycle order."""
    events = []
    for update in range(2, 121, 2):
        ordinal = update // 2 - 1
        kind = "background_only" if ordinal % 2 == 0 else "mixed"
        eligible = [row for row in rows if row["auxiliary_target_kind"] == kind]
        cycle, position = divmod(ordinal // 2, len(eligible))
        ordered = sorted(
            eligible,
            key=lambda row: (
                hashlib.sha256(
                    f"7331:{cycle}:{row['sample_id']}:auxiliary-balanced".encode()
                ).digest(),
                row["sample_id"],
            ),
        )
        events.append(
            {
                "update": update,
                "sample_id": ordered[position]["sample_id"],
                "target_kind": kind,
            }
        )
    return events


class ExplicitScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.references = self.root / "references"
        self.scenes = self.root / "scenes"
        self.references.mkdir()
        self.scenes.mkdir()
        self.rows = donor_rows()
        for index, row in enumerate(self.rows):
            mask = np.zeros((4, 4), dtype=np.uint8)
            if index >= 4:
                mask[:2] = 1
            Image.fromarray(mask).save(self.scenes / row["target_path"])
        # All 32 real training inputs count towards updates, including donors
        # that a future policy might not select for its auxiliary stream.
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0] = 1
        Image.fromarray(mask).save(self.references / "train.png")
        self.real_rows = [
            {"sample_id": f"real/{index}", "split": "train", "target_path": "train.png"}
            for index in range(32)
        ]
        self.config = configuration()
        self.config["auxiliary_sampling"] = {
            "strategy": "explicit_update_sequence",
            "events": legacy_presence_events(self.rows),
        }

    def prepare(self, config=None, rows=None, real_rows=None):
        return prepare_auxiliary_donors(
            self.references,
            self.real_rows if real_rows is None else real_rows,
            self.scenes,
            self.rows if rows is None else rows,
            self.config if config is None else config,
        )

    def test_sixty_events_match_legacy_and_survive_job_id_remapping(self):
        original_config, original_rows = copy.deepcopy(self.config), copy.deepcopy(
            self.rows
        )
        donors = self.prepare()
        legacy = copy.deepcopy(self.config)
        legacy["auxiliary_sampling"] = {"strategy": "balanced_presence_cycle"}
        actual = []
        for update in range(1, 121):
            selected = select_auxiliary_donor(donors, self.config, update)
            self.assertEqual(selected, select_auxiliary_donor(donors, legacy, update))
            if selected is not None:
                actual.append(selected[1]["sample_id"])
        self.assertEqual(len(actual), 60)
        self.assertEqual(
            Counter(
                event["target_kind"]
                for event in self.config["auxiliary_sampling"]["events"]
            ),
            {"mixed": 30, "background_only": 30},
        )
        revised_rows = donor_rows("new-job")
        revised_config = copy.deepcopy(self.config)
        for event in revised_config["auxiliary_sampling"]["events"]:
            event["sample_id"] = event["sample_id"].replace("old-job/", "new-job/")
        revised_donors = self.prepare(revised_config, revised_rows)
        revised = [
            select_auxiliary_donor(
                list(reversed(revised_donors)), revised_config, update
            )[1]["sample_id"]
            for update in range(2, 121, 2)
        ]
        self.assertEqual(
            actual, [value.replace("new-job/", "old-job/") for value in revised]
        )
        self.assertEqual(self.config, original_config)
        self.assertEqual(self.rows, original_rows)

    def test_explicit_sequence_does_not_consume_random_state(self):
        donors = self.prepare()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        for update in range(1, 121):
            select_auxiliary_donor(donors, self.config, update)
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(np.random.get_state()[0], numpy_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])

    def test_unknown_fields_and_malformed_events_are_rejected(self):
        first = self.config["auxiliary_sampling"]["events"][0]
        cases = [
            {"strategy": "explicit_update_sequence", "events": [], "seed": 4},
            {"strategy": "explicit_update_sequence", "events": "not-a-list"},
            {"strategy": "explicit_update_sequence", "events": [dict(first, extra=1)]},
        ]
        for field, value in (
            ("update", True),
            ("update", -2),
            ("update", 3),
            ("update", 2.0),
            ("sample_id", ""),
            ("sample_id", 4),
            ("target_kind", "foreground_only"),
        ):
            cases.append(
                {
                    "strategy": "explicit_update_sequence",
                    "events": [dict(first, **{field: value})],
                }
            )
        cases.extend(
            {"strategy": "explicit_update_sequence", "events": events}
            for events in (
                [first, first],
                list(reversed(self.config["auxiliary_sampling"]["events"])),
            )
        )
        for sampling in cases:
            with self.subTest(sampling=sampling), self.assertRaises(ValueError):
                validate_auxiliary_sampling(
                    dict(self.config, auxiliary_sampling=sampling)
                )

    def test_unknown_donors_wrong_kinds_and_incomplete_budgets_fail_before_fit(self):
        for mutation in ("unknown", "kind", "missing", "extra", "gap"):
            invalid = copy.deepcopy(self.config)
            events = invalid["auxiliary_sampling"]["events"]
            if mutation == "unknown":
                events[0]["sample_id"] = "unregistered/capture"
            elif mutation == "kind":
                events[0]["target_kind"] = "mixed"
            elif mutation == "missing":
                events.pop()
            elif mutation == "extra":
                events.append(dict(events[-1], update=122))
            else:
                events.pop(10)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.prepare(invalid)
        for field, value in (("epochs", True), ("batch_size", 0), ("epochs", 14)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare(dict(self.config, **{field: value}))
        donors = self.prepare()
        with self.assertRaisesRegex(ValueError, "requested update"):
            select_auxiliary_donor(donors, self.config, 122)

    def test_invalid_sources_and_native_masks_are_rejected(self):
        for mutation in ("duplicate", "split", "escape", "kind", "missing"):
            rows = copy.deepcopy(self.rows)
            if mutation == "duplicate":
                rows[-1]["sample_id"] = rows[0]["sample_id"]
            elif mutation == "split":
                rows[0]["split"] = "validation"
            elif mutation == "escape":
                rows[0]["target_path"] = "../references/train.png"
            elif mutation == "kind":
                Image.fromarray(np.zeros((4, 4), np.uint8)).save(
                    self.scenes / rows[-1]["target_path"]
                )
            else:
                rows[0]["target_path"] = "absent.png"
            with (
                self.subTest(mutation=mutation),
                self.assertRaises((ValueError, OSError)),
            ):
                self.prepare(rows=rows)
            if mutation == "kind":
                mask = np.zeros((4, 4), np.uint8)
                mask[:2] = 1
                Image.fromarray(mask).save(self.scenes / self.rows[-1]["target_path"])
        # Check native pixels before resize: a one-pixel illegal ID would
        # otherwise disappear when a larger image is downsampled to 4x4.
        masks = [
            np.zeros((8, 8), np.uint16),
            np.zeros((8, 8, 3), np.uint8),
            np.full((8, 8), 255, np.uint8),
            np.ones((8, 8), np.uint8),
        ]
        illegal = np.zeros((8, 8), np.uint8)
        illegal[0, 0] = 2
        masks.append(illegal)
        for mask in masks:
            Image.fromarray(mask).save(self.scenes / self.rows[0]["target_path"])
            with (
                self.subTest(shape=mask.shape, dtype=mask.dtype),
                self.assertRaises(ValueError),
            ):
                self.prepare()
        Image.fromarray(illegal).save(self.references / "train.png")
        with self.assertRaisesRegex(ValueError, "categorical"):
            self.prepare()

    def test_real_replay_and_short_last_batch_use_the_same_explicit_contract(self):
        config = copy.deepcopy(self.config)
        config.update(experiment_kind="real_only", epochs=3, batch_size=3)
        config.pop("synthetic_mixing")
        rows = copy.deepcopy(self.real_rows[:4])
        config["auxiliary_sampling"]["events"] = [
            {
                "update": update,
                "sample_id": rows[0]["sample_id"],
                "target_kind": "mixed",
            }
            for update in (2, 4, 6)
        ]
        donors = self.prepare(config, real_rows=rows)
        self.assertEqual(len(donors), 4)
        self.assertEqual(
            [
                select_auxiliary_donor(donors, config, update)[1]["sample_id"]
                for update in (2, 4, 6)
            ],
            ["real/0"] * 3,
        )


if __name__ == "__main__":
    unittest.main()
