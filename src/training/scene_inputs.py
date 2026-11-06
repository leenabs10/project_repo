"""Load admitted scene RGB/targets and preserve a declared optimizer budget."""

import hashlib
import json
import math
from pathlib import Path

from src.adapters.storage.local import file_hash
from src.validation.artifacts import verify_snapshot
from src.validation.scene_admission import (
    accepted_scene_ids,
    inspect_scene_dataset,
    require_snapshot_digest,
)


def validate_scene_mixing(settings):
    """Declare slot replacement or paired auxiliary input, without extra updates."""
    if settings == {"strategy": "auxiliary_only"}:
        return
    if (
        not isinstance(settings, dict)
        or set(settings) != {"strategy", "substitution_probability"}
        or settings["strategy"] != "replace_training_slot"
        or type(settings["substitution_probability"]) not in (int, float)
        or not math.isfinite(settings["substitution_probability"])
        or not 0 < settings["substitution_probability"] <= 0.5
    ):
        raise ValueError(
            "Scene mixing requires bounded explicit training-slot replacement"
        )


def load_scene_inputs(
    root,
    expected_sha256,
    references,
    references_sha256,
    real_rows,
    config,
    *,
    fixture=False,
):
    """Return exact admitted scene rows with their own renderer-visible targets.

    Admission is rebound to the current task, cohort and training manifest. Every
    native payload is checked again; an eligibility flag alone is insufficient.
    Fixture snapshots cannot be used for customer experiments.
    """
    require_snapshot_digest(expected_sha256)
    profile = verify_snapshot(root, expected_sha256)
    if profile.get("kind") == "scene_collection":
        from src.training.scene_collections import load_scene_collection

        return load_scene_collection(
            root,
            expected_sha256,
            references,
            references_sha256,
            real_rows,
            config,
            fixture=fixture,
        )
    return load_admitted_scene_inputs(
        root,
        expected_sha256,
        references,
        references_sha256,
        real_rows,
        config,
        fixture=fixture,
    )


def load_admitted_scene_inputs(
    root,
    expected_sha256,
    references,
    references_sha256,
    real_rows,
    config,
    *,
    fixture=False,
):
    """Load a single admitted job; nested collections are rejected."""
    require_snapshot_digest(expected_sha256)
    require_snapshot_digest(references_sha256)
    root = Path(root)
    validate_scene_mixing(config["synthetic_mixing"])
    profile = verify_snapshot(root, expected_sha256)
    real_profile = verify_snapshot(references, references_sha256)
    if (
        profile.get("status") != "passed"
        or profile.get("kind") != "scene_admission"
        or profile.get("qualification_status") != "accepted"
        or profile.get("training_eligible") is not True
        or fixture != (profile.get("purpose") == "known_fixture_only")
        or fixture != (real_profile.get("purpose") == "known_fixture_only")
        or profile.get("reference_profile_sha256") != references_sha256
        or profile.get("protocol_sha256") != real_profile.get("protocol_sha256")
        or profile.get("task_id") != real_profile.get("task_id")
        or profile.get("manifest_sha256") != real_profile.get("manifest_sha256")
    ):
        raise ValueError(
            "Completed scene admission and matching real cohort are required"
        )
    verify_snapshot(root / "quality_evidence", profile["geometry_evidence_sha256"])
    dataset_profile, job, source_rows = inspect_scene_dataset(
        root / "dataset", profile["source_dataset_profile_sha256"]
    )
    if (
        job["conditioning"]["manifest_sha256"] != profile["manifest_sha256"]
        or dataset_profile["job_sha256"] != profile["job_sha256"]
    ):
        raise ValueError("Admitted scene conditioning/job mismatch")
    if file_hash(root / "decisions.json") != profile["decisions_sha256"]:
        raise ValueError("Scene acceptance provenance changed")
    decisions = json.loads((root / "decisions.json").read_text(encoding="utf-8"))
    accepted = accepted_scene_ids(
        decisions,
        profile["source_dataset_profile_sha256"],
        profile["job_sha256"],
        {row["scene_sample_id"] for row in source_rows},
        fixture=fixture,
    )
    if (
        decisions["assessor"] != profile["assessor"]
        or file_hash(root / "geometry_evidence.json")
        != profile["geometry_evidence_sha256"]
    ):
        raise ValueError("Scene assessor or geometry evidence binding mismatch")
    expected_rows = [
        dict(
            row,
            task_id=profile["task_id"],
            protocol_sha256=profile["protocol_sha256"],
            training_eligible=True,
        )
        for row in source_rows
        if row["scene_sample_id"] in accepted
    ]
    rows = [
        json.loads(line)
        for line in (root / "eligible_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if (
        rows != expected_rows
        or len(rows) != profile.get("eligible_samples")
        or not rows
    ):
        raise ValueError("Eligible scene manifest differs from its verified acceptance")
    real_ids = {row["sample_id"] for row in real_rows}
    real_groups = {row["group_id"] for row in real_rows}
    real_hashes = {row["source_rgb_sha256"] for row in real_rows}
    if any(
        row["split"] != "train"
        or row["sample_id"] in real_ids
        or row["group_id"] in real_groups
        or row["source_rgb_sha256"] in real_hashes
        or row["image_size"] != config["native_reference_resolution"]
        for row in rows
    ):
        raise ValueError("Scene samples conflict with real splits or native geometry")
    return sorted(rows, key=lambda row: row["sample_id"])


def select_scene_training_sample(
    references, source_row, scene_root, scenes, config, epoch
):
    """Replace both RGB and target for a training slot using a separate hash RNG.

    Source-slot order, torch RNG, batch size and update count remain unchanged.
    The model sees fewer real presentations when scenes replace slots; report the
    actual counts. Evaluation never calls this sampler.
    """
    if source_row["split"] != "train":
        raise ValueError("Scene sampling is training-only")
    if not scenes:
        raise ValueError("Scene sampling requires a nonempty admitted batch")
    message = json.dumps(
        [config["generation_seed"], epoch, source_row["sample_id"]],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(message.encode("utf-8")).digest()
    if (
        int.from_bytes(digest[:8], "big") / 2**64
        >= config["synthetic_mixing"]["substitution_probability"]
    ):
        return Path(references), source_row, False
    return (
        Path(scene_root),
        scenes[int.from_bytes(digest[8:16], "big") % len(scenes)],
        True,
    )
