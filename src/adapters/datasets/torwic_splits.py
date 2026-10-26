"""Verify TorWIC trajectory/date/path membership using metadata only.

Returns train/validation/test row groups without reading sensor payloads.
Rejects duplicate IDs, inconsistent source paths/dates, and incomplete splits.
"""

from pathlib import Path

from src.ingestion.torwic_loader import CONFIRMED_SPLITS, MODALITIES


def evaluation_records(records):
    """Verify trajectory/path/date membership before freezing evaluation metadata."""
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("Duplicate dataset sample IDs")
    groups = {}
    for record in records:
        sequence = record["sequence_id"]
        expected = CONFIRMED_SPLITS.get(sequence)
        if expected is None or (record["split"], record["collection_date"]) != expected:
            raise ValueError("Unknown or inconsistent confirmed sequence split/date")
        if record["sample_id"] != f"{sequence}/{record['frame_id']}":
            raise ValueError("Inconsistent sample/sequence/frame IDs")
        for key in list(MODALITIES) + ["imu_left_path", "imu_right_path"]:
            if Path(record[key]).parts[:2] != (sequence, sequence):
                raise ValueError("Record modality path crosses its assigned trajectory")
        groups.setdefault(record["split"], []).append(
            {
                "sample_id": record["sample_id"],
                "sequence_id": sequence,
                "collection_date": record["collection_date"],
                "timestamp_seconds": record["timestamp_seconds"],
                "rgb_path": record["rgb_path"],
                "semantic_path": record["semantic_path"],
            }
        )
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("Expected complete train, validation and held-out test groups")
    return groups
