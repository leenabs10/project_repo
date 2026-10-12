"""Prevent mislabeled goods targets and trajectory leakage in task preparation."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "profiling"))
from task_dataset_review import evaluation_records, goods_mask
from torwic_loader import CONFIRMED_SPLITS, MODALITIES


class TaskReviewTests(unittest.TestCase):
    def records(self):
        result = []
        for sequence, (split, date) in CONFIRMED_SPLITS.items():
            record = {
                "sequence_id": sequence,
                "split": split,
                "collection_date": date,
                "sample_id": f"{sequence}/000000",
                "frame_id": "000000",
                "timestamp_seconds": 0.0,
            }
            for key in list(MODALITIES) + ["imu_left_path", "imu_right_path"]:
                record[key] = f"{sequence}/{sequence}/file"
            result.append(record)
        return result

    def test_goods_binary_one_is_class_seven_not_driveable_ground(self):
        source = np.array([[1, 7, 0, 15]], dtype=np.uint8)
        target = goods_mask(source)
        np.testing.assert_array_equal(target, [[0, 1, 0, 0]])
        np.testing.assert_array_equal(source, [[1, 7, 0, 15]])
        with self.assertRaises(ValueError):
            goods_mask(np.array([[16]], dtype=np.uint8))

    def test_confirmed_groups_are_separate_and_cross_paths_fail(self):
        records = self.records()
        self.assertEqual(
            set(evaluation_records(records)), {"train", "validation", "test"}
        )
        changed = copy.deepcopy(records)
        changed[0]["rgb_path"] = records[1]["rgb_path"]
        with self.assertRaises(ValueError):
            evaluation_records(changed)

    def test_relabelled_dates_duplicates_and_missing_splits_fail(self):
        records = self.records()
        for key, value in (("split", "train"), ("collection_date", "2022-06-15")):
            changed = copy.deepcopy(records)
            changed[-1][key] = value
            with self.assertRaises(ValueError):
                evaluation_records(changed)
        with self.assertRaises(ValueError):
            evaluation_records(records + [records[0]])
        with self.assertRaises(ValueError):
            evaluation_records(records[:2])


if __name__ == "__main__":
    unittest.main()
