"""Verify qualified goods cohorts without importing training orchestration.

Inputs are a cohort root, expected profile SHA256, and a task-definition path.
Returns reference-row dictionaries after checking source hashes, provenance,
splits, native RGB geometry, and uint8 target IDs 0/1/255. Raises ValueError on
qualification mismatches; missing/unreadable artifacts propagate I/O errors.
No jobs execute at import time and locked-test rows are never admitted.
"""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash, source_path
from src.adapters.tasks.goods import load_definition, validate_target
from src.validation.artifacts import verify_snapshot


def load_references(root, expected_sha, task, *, fixture=False):
    """Validate provenance, splits, identities and native targets before model I/O.

    Evaluation reads only the exported development cohort. There is no held-out
    test flag, source-dataset scan, or automatic pretrained-weight download.
    """
    root, task = Path(root), Path(task)
    profile = verify_snapshot(root, expected_sha)
    definition = load_definition(task)
    if profile.get("status") != "passed" or not profile.get(
        "benchmark_reference_ready"
    ):
        raise ValueError("Qualified references are required before benchmark execution")
    if (
        profile["protocol_sha256"] != file_hash(task)
        or profile["task_id"] != definition["task_id"]
    ):
        raise ValueError("Reference task/protocol mismatch")
    rows = [
        json.loads(line)
        for line in (root / "references.jsonl").read_text().splitlines()
    ]
    expected_origin = (
        "fixture_ground_truth" if fixture else definition["reference_origin"]
    )
    if fixture != (profile.get("purpose") == "known_fixture_only"):
        raise ValueError("Fixture evidence cannot be used as a real benchmark")
    seen, groups, rgb_splits = set(), {}, {}
    counts = Counter()
    for row in rows:
        if row["split"] not in {"train", "validation"}:
            raise ValueError("Held-out test references are not supported by this CLI")
        if row["sample_id"] in seen:
            raise ValueError("Duplicate reference sample")
        seen.add(row["sample_id"])
        if row["reference_origin"] != expected_origin or (
            not fixture and row.get("reviewer_kind", "human") != "human"
        ):
            raise ValueError(
                "AI/predicted targets cannot enter the real reference benchmark"
            )
        if (
            row["protocol_sha256"] != file_hash(task)
            or row["task_id"] != definition["task_id"]
        ):
            raise ValueError("Reference row task binding mismatch")
        group = row["group_id"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError("A complete capture group cannot cross dataset splits")
        groups[group] = row["split"]
        digest = row["source_rgb_sha256"]
        if digest in rgb_splits and rgb_splits[digest] != row["split"]:
            raise ValueError("RGB payload is reused across train/development splits")
        rgb_splits[digest] = row["split"]
        rgb_path, target_path = source_path(root, row["rgb_path"]), source_path(
            root, row["target_path"]
        )
        if (
            file_hash(rgb_path) != digest
            or file_hash(target_path) != row["target_sha256"]
        ):
            raise ValueError("Reference payload hash mismatch")
        with Image.open(rgb_path) as image:
            if image.mode != "RGB" or list(image.size) != row["image_size"]:
                raise ValueError("Reference RGB geometry/encoding mismatch")
        with Image.open(target_path) as image:
            target = np.array(image)
        validate_target(
            target,
            tuple(reversed(row["image_size"])),
            definition["maximum_ignore_fraction"],
        )
        counts[row["split"]] += 1
    if dict(counts) != profile["reference_counts"] or any(
        counts[split] < minimum
        for split, minimum in definition["minimum_reference_frames"].items()
    ):
        raise ValueError("Reference cohort is incomplete or counts changed")
    if not fixture and profile.get("qualified_reference_counts", dict(counts)) != dict(
        counts
    ):
        raise ValueError("Qualified reference counts disagree with human provenance")
    return rows
