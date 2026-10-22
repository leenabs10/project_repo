"""Validated RGB derivatives and source-balanced, budget-preserving sampling.

This module accepts completed admission snapshots, never pending generation
packets. Targets always come from the fixed real reference cohort. A condition
can replace its own source input; it cannot add source slots or touch evaluation.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash, source_path
from src.validation.artifacts import verify_snapshot


def validate_mixing(settings):
    """Require explicit source-slot replacement and supported bounded RGB gains."""
    if (
        not isinstance(settings, dict)
        or set(settings) != {"strategy", "substitution_probability", "conditions"}
        or settings["strategy"] != "replace_source_input"
    ):
        raise ValueError("Unsupported hybrid mixing configuration")
    probability = settings["substitution_probability"]
    if (
        type(probability) not in (int, float)
        or not math.isfinite(probability)
        or not 0 < probability <= 1
    ):
        raise ValueError("Hybrid substitution probability must be in (0, 1]")
    conditions = settings["conditions"]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Explicit RGB condition definitions are required")
    identities = set()
    for condition in conditions:
        if (
            not isinstance(condition, dict)
            or set(condition) != {"condition_id", "method", "gain"}
            or not isinstance(condition["condition_id"], str)
            or not condition["condition_id"].strip()
            or condition["method"] != "rgb_gain"
            or type(condition["gain"]) not in (int, float)
            or not math.isfinite(condition["gain"])
            or condition["gain"] <= 0
            or condition["gain"] == 1
            or condition["condition_id"] in identities
        ):
            raise ValueError("Expected unique, finite nonidentity RGB gain conditions")
        identities.add(condition["condition_id"])


def load_synthetic_inputs(
    root, expected_sha, references, references_sha, rows, config, *, fixture=False
):
    """Return source-indexed admitted derivatives after native-pixel verification.

    Bind the admission, task, manifest, source RGB and unchanged categorical
    target to the same reference version. Independently reconstruct each gain
    from native uint8 RGB; this detects unchanged inputs and fabricated lineage.
    Only declared training sources can have alternatives. Fixture and customer
    evidence are deliberately incompatible.
    """
    root, references = Path(root), Path(references)
    profile = verify_snapshot(root, expected_sha)
    reference_profile = verify_snapshot(references, references_sha)
    if (
        profile.get("status") != "passed"
        or profile.get("qualification_status") != "accepted"
        or profile.get("training_eligible") is not True
        or not str(profile.get("reviewer") or "").strip()
        or fixture != (profile.get("purpose") == "known_fixture_only")
    ):
        raise ValueError(
            "Completed condition admission is required; pending inputs are excluded"
        )
    if (
        profile.get("reference_profile_sha256") != references_sha
        or profile.get("protocol_sha256") != reference_profile["protocol_sha256"]
        or profile.get("task_id") != reference_profile["task_id"]
        or not reference_profile.get("manifest_sha256")
        or profile.get("manifest_sha256") != reference_profile["manifest_sha256"]
    ):
        raise ValueError("Synthetic reference/task/manifest version mismatch")
    entries = [
        json.loads(line)
        for line in (root / "eligible_manifest.jsonl").read_text().splitlines()
    ]
    if not entries or len(entries) != profile.get("eligible_variants"):
        raise ValueError("Admitted variant count mismatch")
    sources = {row["sample_id"]: row for row in rows}
    seen, source_conditions, index = set(), set(), {}
    for entry in entries:
        record = entry["record"]
        source = sources.get(entry["source_sample_id"])
        if (
            source is None
            or source["split"] != "train"
            or record["split"] != "train"
            or record["group_id"] != source["group_id"]
        ):
            raise ValueError(
                "Synthetic inputs must belong to a fixed training source/group"
            )
        identity = record["sample_id"]
        if identity in seen or identity in sources:
            raise ValueError("Duplicate or conflicting synthetic identity")
        seen.add(identity)
        if (
            entry.get("training_eligible") is not True
            or type(entry.get("generation_seed")) is not int
            or entry["generation_seed"] < 0
            or entry["task_id"] != source["task_id"]
            or entry["target_origin"] != source["reference_origin"]
            or entry["target_sha256"] != source["target_sha256"]
            or entry.get("source_rgb_sha256") != source["source_rgb_sha256"]
            or record["image_size"] != source["image_size"]
        ):
            raise ValueError("Synthetic target/source binding mismatch")
        condition = entry.get("condition")
        if condition not in config["synthetic_mixing"]["conditions"]:
            raise ValueError("Unsupported or undeclared synthetic condition")
        source_condition = (source["sample_id"], condition["condition_id"])
        if source_condition in source_conditions:
            raise ValueError("Duplicate condition for the same source")
        source_conditions.add(source_condition)
        rgb = record["modalities"]["rgb"]
        if rgb["sha256"] in {
            row["source_rgb_sha256"] for row in rows if row["split"] == "validation"
        }:
            raise ValueError("Synthetic RGB payload is reused across dataset splits")
        rgb_path = source_path(root, rgb["path"])
        target_path = source_path(root, entry["target_path"])
        if (
            rgb.get("status") != "available"
            or file_hash(rgb_path) != rgb["sha256"]
            or file_hash(target_path) != source["target_sha256"]
        ):
            raise ValueError("Synthetic RGB/target payload checksum mismatch")
        with Image.open(rgb_path) as image:
            if image.mode != "RGB" or list(image.size) != source["image_size"]:
                raise ValueError("Synthetic native RGB geometry/encoding mismatch")
            actual = np.array(image)
        with Image.open(source_path(references, source["rgb_path"])) as image:
            original = np.array(image)
        expected = np.rint(
            np.clip(original.astype(np.float32) * condition["gain"], 0, 255)
        ).astype(np.uint8)
        if not np.array_equal(actual, expected) or np.array_equal(actual, original):
            raise ValueError(
                "Synthetic pixels do not match a changed declared RGB gain"
            )
        index.setdefault(source["sample_id"], []).append(
            {
                "sample_id": identity,
                "rgb_path": rgb["path"],
                "root": root,
                "condition": condition,
            }
        )
    for alternatives in index.values():
        alternatives.sort(key=lambda row: row["sample_id"])
    return index


def select_training_input(references, row, alternatives, config, epoch):
    """Choose an RGB input without changing source order, targets or torch RNG.

    The hash draw depends on generation seed, source ID and zero-based epoch;
    every model seed sees the same per-source condition schedule. Unequal numbers
    of accepted variants never increase a source's representation. Validation
    rows are refused instead of being silently augmented.
    """
    if row["split"] != "train":
        raise ValueError("Condition mixing is training-only")
    choices = alternatives.get(row["sample_id"], [])
    if not choices:
        return Path(references), row, False
    message = json.dumps(
        [config["generation_seed"], epoch, row["sample_id"]],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(message.encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    if draw >= config["synthetic_mixing"]["substitution_probability"]:
        return Path(references), row, False
    selected = choices[int.from_bytes(digest[8:16], "big") % len(choices)]
    return selected["root"], selected, True
