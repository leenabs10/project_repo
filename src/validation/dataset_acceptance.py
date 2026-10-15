"""Source-bound condition decisions and corrected targets for dataset admission."""

import copy
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.adapters.tasks.goods import load_definition, validate_target
from src.validation.artifacts import (
    artifact_hashes,
    verify_snapshot,
    write_json,
    write_implementation_snapshot,
)


def prepare_decisions(
    batch: Path, expected_batch_sha256: str, output: Path, *, source_root: Path
) -> Path:
    """Create a fresh feedback template bound to the exact generated batch snapshot."""
    batch = Path(batch)
    profile = verify_snapshot(batch, expected_batch_sha256)
    if profile.get("status") != "passed":
        raise ValueError("Only completed batches can receive decisions")
    output = reserve_output(output, (batch, source_root))
    fields = [
        "batch_profile_sha256",
        "variant_id",
        "source_sample_id",
        "appearance_decision",
        "mask_decision",
        "notes",
    ]
    with (output / "decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for line in (batch / "attempts.jsonl").read_text().splitlines():
            attempt = json.loads(line)
            if attempt["automated_status"] == "passed":
                writer.writerow(
                    {
                        "batch_profile_sha256": expected_batch_sha256,
                        "variant_id": attempt["variant_id"],
                        "source_sample_id": attempt["source_sample_id"],
                        "appearance_decision": "pending",
                        "mask_decision": "pending",
                        "notes": "",
                    }
                )
    return output / "decisions.csv"


def read_decisions(
    path: Path, attempts: list[dict], reviewer: str, expected_batch_sha256: str
) -> dict:
    """Read a copied decision CSV, rejecting stale, unknown and rejected variants."""
    by_id = {
        row["variant_id"]: row
        for row in attempts
        if row["automated_status"] == "passed"
    }
    if len(by_id) != sum(row["automated_status"] == "passed" for row in attempts):
        raise ValueError("Duplicate generated variant identity")
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        expected_fields = [
            "batch_profile_sha256",
            "variant_id",
            "source_sample_id",
            "appearance_decision",
            "mask_decision",
            "notes",
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError("Unexpected decision CSV columns")
        decisions = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed decision CSV row")
            identity = row["variant_id"]
            if row["batch_profile_sha256"] != expected_batch_sha256:
                raise ValueError("Decision refers to a different batch snapshot")
            if (
                identity not in by_id
                or row["source_sample_id"] != by_id[identity]["source_sample_id"]
            ):
                raise ValueError("Decision source binding is stale or unknown")
            if identity in decisions:
                raise ValueError("Duplicate variant decision")
            if any(
                row[field] not in {"pending", "accept", "reject"}
                for field in ("appearance_decision", "mask_decision")
            ):
                raise ValueError("Decision must be pending, accept or reject")
            if (
                any(
                    row[field] != "pending"
                    for field in ("appearance_decision", "mask_decision")
                )
                and not reviewer.strip()
            ):
                raise ValueError("Explicit decisions require a reviewer identity")
            decisions[identity] = row
    return decisions


def qualify_dataset(
    batch: Path,
    expected_batch_sha256: str,
    decisions_path: Path,
    reviewer: str,
    references: Path,
    definition_path: Path,
    output: Path,
    *,
    source_root: Path,
    expected_reference_sha256: str
) -> dict:
    """Export only explicitly accepted train variants with corrected source targets.

    Appearance acceptance alone cannot admit a predicted semantic mask. Source
    hashes, task version, split, cohort size and corrected target identity must agree.
    Original batches/references remain immutable; this is a new derivative snapshot.
    """
    batch, references, decisions_path, definition_path = map(
        Path, (batch, references, decisions_path, definition_path)
    )
    definition = load_definition(definition_path)
    protocol_sha = file_hash(definition_path)
    batch_profile = verify_snapshot(batch, expected_batch_sha256)
    reference_profile = verify_snapshot(references, expected_reference_sha256)
    if (
        batch_profile.get("status") != "passed"
        or batch_profile.get("backend") != "real_sensor_conditions"
        or batch_profile.get("source_split") != "train"
    ):
        raise ValueError("Only completed training condition batches can be qualified")
    if (
        reference_profile.get("status") != "passed"
        or reference_profile.get("manifest_sha256") != batch_profile["manifest_sha256"]
        or reference_profile.get("protocol_sha256") != protocol_sha
    ):
        raise ValueError("Reference dataset or task definition mismatch")
    attempts = [
        json.loads(line) for line in (batch / "attempts.jsonl").read_text().splitlines()
    ]
    feedback_sha = file_hash(decisions_path)
    decisions = read_decisions(
        decisions_path, attempts, reviewer, expected_batch_sha256
    )
    reference_rows = [
        json.loads(line)
        for line in (references / "references.jsonl").read_text().splitlines()
    ]
    reference_index = {row["sample_id"]: row for row in reference_rows}
    if len(reference_index) != len(reference_rows):
        raise ValueError("Duplicate qualified reference identity")
    counts = {
        split: sum(row["split"] == split for row in reference_rows)
        for split in ("train", "validation")
    }
    cohort_ready = all(
        counts[split] >= required
        for split, required in definition["minimum_reference_frames"].items()
    )
    if (
        counts != reference_profile["reference_counts"]
        or cohort_ready != reference_profile["benchmark_reference_ready"]
    ):
        raise ValueError("Reference cohort count/status mismatch")
    output = reserve_output(
        output, (batch, references, decisions_path, definition_path, source_root)
    )
    write_json(
        output / "profile.json", {"status": "running", "training_eligible": False}
    )
    try:
        write_implementation_snapshot(output)
        accepted, dispositions = [], []
        for attempt in attempts:
            reasons = []
            decision = decisions.get(attempt["variant_id"], {})
            reference = reference_index.get(attempt["source_sample_id"])
            if attempt["automated_status"] != "passed":
                reasons.append("automated_rejection")
            if any(
                decision.get(field) != "accept"
                for field in ("appearance_decision", "mask_decision")
            ):
                reasons.append("condition_acceptance_incomplete_or_rejected")
            if not reference_profile["benchmark_reference_ready"]:
                reasons.append("reference_cohort_incomplete")
            if reference is None:
                reasons.append("corrected_source_target_unavailable")
            else:
                if (
                    reference["split"] != "train"
                    or reference["group_id"] != attempt["source_group_id"]
                    or reference["task_id"] != definition["task_id"]
                    or reference["protocol_sha256"] != protocol_sha
                    or reference["reference_origin"] != definition["reference_origin"]
                ):
                    raise ValueError("Qualified target split/task/provenance mismatch")
                if reference["source_rgb_sha256"] != attempt["source_sha256"][
                    "rgb"
                ] or reference["source_semantic_sha256"] != attempt[
                    "source_sha256"
                ].get(
                    "semantic"
                ):
                    raise ValueError(
                        "Corrected target does not belong to variant source"
                    )
            if not reasons:
                folder = source_path(output, "samples/" + attempt["variant_id"])
                folder.mkdir(parents=True)
                record = copy.deepcopy(attempt["record"])
                for name, modality in record["modalities"].items():
                    if modality["status"] != "available":
                        continue
                    source = source_path(batch, modality["path"])
                    destination = folder / (name + source.suffix)
                    shutil.copyfile(source, destination)
                    if file_hash(destination) != modality["sha256"]:
                        raise ValueError("Qualified modality export checksum mismatch")
                    modality["path"] = destination.relative_to(output).as_posix()
                target_source = source_path(references, reference["target_path"])
                with Image.open(target_source) as image:
                    target = np.array(image)
                width, height = reference["image_size"]
                validate_target(
                    target, (height, width), definition["maximum_ignore_fraction"]
                )
                target_path = folder / "goods_target.png"
                shutil.copyfile(target_source, target_path)
                if file_hash(target_path) != reference["target_sha256"]:
                    raise ValueError("Qualified task target checksum mismatch")
                accepted.append(
                    {
                        "record": record,
                        "source_sample_id": attempt["source_sample_id"],
                        "task_id": definition["task_id"],
                        "target_path": target_path.relative_to(output).as_posix(),
                        "target_sha256": reference["target_sha256"],
                        "target_origin": reference["reference_origin"],
                        "training_eligible": True,
                    }
                )
            dispositions.append(
                {
                    "variant_id": attempt["variant_id"],
                    "source_sample_id": attempt["source_sample_id"],
                    "eligible": not reasons,
                    "reasons": reasons,
                }
            )
        if file_hash(decisions_path) != feedback_sha:
            raise ValueError("Decision file changed during qualification")
        verify_snapshot(batch, expected_batch_sha256)
        verify_snapshot(references, expected_reference_sha256)
        if file_hash(definition_path) != protocol_sha:
            raise ValueError("Task definition changed during qualification")
        (output / "eligible_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8"
        )
        write_json(output / "dispositions.json", dispositions)
        coverage = {}
        for attempt in attempts:
            condition_id = attempt["condition"]["condition_id"]
            condition = coverage.setdefault(
                condition_id, {"requested": 0, "accepted": 0}
            )
            condition["requested"] += 1
            condition["accepted"] += int(
                any(
                    row["record"]["sample_id"] == attempt["variant_id"]
                    for row in accepted
                )
            )
        result = {
            "status": "passed",
            "qualification_status": (
                "accepted" if accepted else "acceptance_in_progress"
            ),
            "task_id": definition["task_id"],
            "protocol_sha256": protocol_sha,
            "source_batch_profile_sha256": expected_batch_sha256,
            "reference_profile_sha256": expected_reference_sha256,
            "decisions_sha256": feedback_sha,
            "reviewer": reviewer.strip() or None,
            "eligible_variants": len(accepted),
            "training_eligible": bool(accepted),
            "condition_coverage": coverage,
            "delivered_coverage_complete": bool(coverage)
            and all(row["requested"] == row["accepted"] for row in coverage.values()),
            "artifact_sha256": artifact_hashes(output),
        }
        write_json(output / "profile.json", result)
        return result
    except Exception as error:
        write_json(
            output / "profile.json",
            {"status": "failed", "error": str(error), "training_eligible": False},
        )
        raise
