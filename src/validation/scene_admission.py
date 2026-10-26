"""Admit source-bound rendered goods samples after explicit quality decisions.

Original canonical exports remain quarantined and immutable. Admission creates a
separate snapshot; it never changes source eligibility or treats geometry checks
as appearance acceptance. Only visible renderer labels supply scene targets.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.adapters.tasks.goods import load_definition
from src.adapters.tasks.goods_scene import validate_goods_scene
from src.core.contracts import validate_record
from src.validation.artifacts import artifact_hashes, verify_snapshot, write_json
from src.validation.scene_dataset import validate_arrays


def require_snapshot_digest(value):
    """Reject omitted/empty identities at mandatory admission boundaries."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("An explicit SHA256 snapshot identity is required")


def inspect_scene_dataset(root, expected_sha256):
    """Recheck canonical native payloads and reconstruct traceable training rows.

    The return value contains the immutable source profile, resolved job and rows.
    Depth uses axial metres; semantic/instance maps remain discrete uint8/uint16.
    Missing, conflicting or duplicate scene identities fail before admission.
    """
    require_snapshot_digest(expected_sha256)
    root = Path(root)
    profile = verify_snapshot(root, expected_sha256)
    if (
        profile.get("status") != "structural_checks_passed"
        or profile.get("training_eligible") is not False
    ):
        raise ValueError("Expected an unmodified quarantined canonical scene export")
    job_path = root / "generation_job.json"
    if file_hash(job_path) != profile.get("job_sha256"):
        raise ValueError("Canonical scene job binding mismatch")
    job = validate_goods_scene(json.loads(job_path.read_text(encoding="utf-8")))
    if (
        "asset_catalog_sha256" in job
        and file_hash(root / "asset_catalog.json") != job["asset_catalog_sha256"]
    ):
        raise ValueError("Canonical scene material catalog mismatch")
    records = [
        validate_record(json.loads(line))
        for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [record["sample_id"] for record in records] != [
        capture["sample_id"] for capture in job["scenes"]
    ] or len(records) != profile.get("captures"):
        raise ValueError("Canonical capture membership differs from its resolved job")
    rows = []
    for record, capture in zip(records, job["scenes"], strict=True):
        if (
            record["split"] != "train"
            or record["group_id"] != job["job_id"]
            or record["source_metadata"].get("job_sha256") != profile["job_sha256"]
            or record["source_metadata"].get("conditioning") != job["conditioning"]
            or record["source_metadata"].get("origin") != "scene_simulation"
            or record["source_metadata"].get("taxonomy") != job["taxonomy"]
            or record["dataset_id"] != f'{job["job_id"]}.{profile["job_sha256"]}'
            or record["image_size"] != [job["camera"]["width"], job["camera"]["height"]]
        ):
            raise ValueError("Scene split, task or job lineage mismatch")
        payloads = {}
        for name in ("rgb", "depth", "depth_valid", "semantic", "instance"):
            modality = record["modalities"][name]
            if (
                modality.get("status") != "available"
                or profile["artifact_sha256"].get(modality["path"])
                != modality["sha256"]
            ):
                raise ValueError("Canonical scene modality is not checksum-bound")
            path = source_path(root, modality["path"])
            if file_hash(path) != modality["sha256"]:
                raise ValueError("Canonical modality checksum mismatch")
            if path.suffix == ".npy":
                payloads[name] = np.load(path, allow_pickle=False)
            else:
                with Image.open(path) as image:
                    if image.format != "PNG" or (name == "rgb" and image.mode != "RGB"):
                        raise ValueError(
                            "Scene images must use native lossless PNG encoding"
                        )
                    payloads[name] = np.array(image)
        depth_spec = record["modalities"]["depth"]
        if any(
            depth_spec.get(key) != value
            for key, value in dict(
                units="metres",
                scale_to_metres=1.0,
                invalid_raw_value=0,
                frame="optical_camera_axial_z",
            ).items()
        ):
            raise ValueError("Scene depth convention differs from clean axial metres")
        validate_arrays(
            payloads["rgb"],
            payloads["depth"],
            payloads["depth_valid"],
            payloads["semantic"],
            payloads["instance"],
            job["camera"],
            capture["objects"],
            required_visible_class_ids=job.get("required_visible_class_ids", [0, 1]),
        )
        rows.append(
            dict(
                sample_id=f'{record["dataset_id"]}/{record["sample_id"]}',
                scene_sample_id=record["sample_id"],
                dataset_id=record["dataset_id"],
                split="train",
                group_id=record["group_id"],
                image_size=record["image_size"],
                rgb_path="dataset/" + record["modalities"]["rgb"]["path"],
                source_rgb_sha256=record["modalities"]["rgb"]["sha256"],
                target_path="dataset/" + record["modalities"]["semantic"]["path"],
                target_sha256=record["modalities"]["semantic"]["sha256"],
                reference_origin="renderer_visible_surfaces",
                job_sha256=profile["job_sha256"],
            )
        )
    return profile, job, rows


def accepted_scene_ids(
    decisions, dataset_sha256, job_sha256, sample_ids, *, fixture=False
):
    """Require completed per-sample decisions bound to exactly this rendered batch."""
    if (
        not isinstance(decisions, dict)
        or set(decisions)
        != {
            "schema_version",
            "dataset_profile_sha256",
            "job_sha256",
            "assessor",
            "samples",
        }
        or decisions["schema_version"] != "dubaidata.scene-decisions/1.0"
        or decisions["dataset_profile_sha256"] != dataset_sha256
        or decisions["job_sha256"] != job_sha256
    ):
        raise ValueError(
            "Appearance decisions are not bound to this exact scene export"
        )
    assessor = decisions["assessor"]
    expected_method = "known_fixture" if fixture else "operator_confirmation"
    if (
        not isinstance(assessor, dict)
        or set(assessor) != {"name", "method"}
        or not isinstance(assessor["name"], str)
        or not assessor["name"].strip()
        or assessor["method"] != expected_method
    ):
        raise ValueError(
            "Explicit assessor provenance is required; fixture and customer decisions are separate"
        )
    rows = decisions["samples"]
    if not isinstance(rows, list) or len(rows) != len(sample_ids):
        raise ValueError(
            "All scene samples require explicit appearance and label decisions"
        )
    seen, accepted = set(), set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"sample_id", "appearance", "labels", "notes"}
            or row["sample_id"] not in sample_ids
            or row["sample_id"] in seen
            or row["appearance"] not in ("accept", "reject")
            or row["labels"] not in ("accept", "reject")
            or not isinstance(row["notes"], str)
        ):
            raise ValueError("Incomplete, conflicting or duplicate scene decisions")
        seen.add(row["sample_id"])
        if row["appearance"] == row["labels"] == "accept":
            accepted.add(row["sample_id"])
    if not accepted:
        raise ValueError(
            "No scene samples passed both appearance and label qualification"
        )
    return accepted


def admit_scene_dataset(
    dataset,
    dataset_sha256,
    verification,
    verification_sha256,
    decisions_path,
    references,
    references_sha256,
    task_path,
    output,
    *,
    fixture=False,
):
    """Create a fresh admission snapshot after all source and decision checks pass.

    Conditioning must match the qualified real cohort's manifest and task. No
    development/test image may be reused as synthetic RGB. Rejected samples remain
    preserved in the copied source export but are excluded from the eligible manifest.
    """
    dataset, verification, decisions_path, references, task_path = map(
        Path, (dataset, verification, decisions_path, references, task_path)
    )
    require_snapshot_digest(verification_sha256)
    require_snapshot_digest(references_sha256)
    dataset_profile, job, rows = inspect_scene_dataset(dataset, dataset_sha256)
    evidence = verify_snapshot(verification, verification_sha256)
    if (
        evidence.get("dataset_profile_sha256") != dataset_sha256
        or evidence.get("job_sha256") != dataset_profile["job_sha256"]
        or evidence.get("captures") != len(rows)
        or evidence.get("independent_rays", 0) < 128 * len(rows)
        or len(evidence.get("independent_instance_agreement", [])) != len(rows)
        or any(
            not np.isfinite(value) or value < 0.99 or value > 1
            for value in evidence["independent_instance_agreement"]
        )
        or len(evidence.get("independent_depth_p95_metres", [])) != len(rows)
        or any(
            not np.isfinite(value) or not 0 <= value <= 0.005
            for value in evidence["independent_depth_p95_metres"]
        )
        or evidence.get("reference_artifacts_changed") != 0
    ):
        raise ValueError(
            "Independent geometry evidence does not qualify this exact scene export"
        )
    task = load_definition(task_path)
    reference_profile = verify_snapshot(references, references_sha256)
    if (
        reference_profile.get("status") != "passed"
        or reference_profile.get("benchmark_reference_ready") is not True
        or fixture != (reference_profile.get("purpose") == "known_fixture_only")
        or reference_profile.get("protocol_sha256") != file_hash(task_path)
        or reference_profile.get("task_id") != task["task_id"]
        or reference_profile.get("manifest_sha256")
        != job["conditioning"]["manifest_sha256"]
    ):
        raise ValueError("Scene conditioning/task/reference cohort mismatch")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    accepted = accepted_scene_ids(
        decisions,
        dataset_sha256,
        dataset_profile["job_sha256"],
        {row["scene_sample_id"] for row in rows},
        fixture=fixture,
    )
    reference_rows = [
        json.loads(line)
        for line in (references / "references.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if any(row.get("split") not in ("train", "validation") for row in reference_rows):
        raise ValueError(
            "Held-out references are not permitted in development admission"
        )
    real_ids = {row["sample_id"] for row in reference_rows}
    real_groups = {row["group_id"] for row in reference_rows}
    real_rgb_hashes = {row["source_rgb_sha256"] for row in reference_rows}
    eligible = [
        dict(
            row,
            task_id=task["task_id"],
            protocol_sha256=file_hash(task_path),
            training_eligible=True,
        )
        for row in rows
        if row["scene_sample_id"] in accepted
    ]
    if any(
        row["sample_id"] in real_ids
        or row["group_id"] in real_groups
        or row["source_rgb_sha256"] in real_rgb_hashes
        for row in eligible
    ):
        raise ValueError(
            "Scene identities, families or RGB payloads overlap real references"
        )
    source_hashes = {
        path: file_hash(path)
        for path in (
            dataset / "profile.json",
            verification / "profile.json",
            decisions_path,
            references / "profile.json",
            task_path,
        )
    }
    output = reserve_output(
        output,
        (
            dataset,
            verification,
            decisions_path,
            references,
            task_path,
            Path(__file__).parents[2] / "data/raw",
        ),
    )
    try:
        shutil.copytree(dataset, output / "dataset")
        shutil.copytree(verification, output / "quality_evidence")
        verify_snapshot(output / "quality_evidence", verification_sha256)
        shutil.copyfile(decisions_path, output / "decisions.json")
        shutil.copyfile(
            verification / "profile.json", output / "geometry_evidence.json"
        )
        shutil.copyfile(task_path, output / "task.json")
        (output / "eligible_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in eligible), encoding="utf-8"
        )
        verify_snapshot(output / "dataset", dataset_sha256)
        if any(file_hash(path) != expected for path, expected in source_hashes.items()):
            raise ValueError("Admission source metadata changed during copying")
        profile = dict(
            status="passed",
            kind="scene_admission",
            purpose="known_fixture_only" if fixture else "qualified_scene_dataset",
            qualification_status="accepted",
            training_eligible=True,
            assessor=decisions["assessor"],
            reference_profile_sha256=references_sha256,
            task_id=task["task_id"],
            protocol_sha256=file_hash(task_path),
            manifest_sha256=job["conditioning"]["manifest_sha256"],
            source_dataset_profile_sha256=dataset_sha256,
            job_sha256=dataset_profile["job_sha256"],
            geometry_evidence_sha256=verification_sha256,
            decisions_sha256=file_hash(decisions_path),
            eligible_samples=len(eligible),
            rejected_samples=len(rows) - len(eligible),
            held_out_test_evaluated=False,
            physical_sensor_realism_certified=False,
            utility_measured=False,
            artifact_sha256=artifact_hashes(output),
        )
        write_json(output / "profile.json", profile)
        return profile
    except Exception as error:
        write_json(
            output / "profile.json",
            dict(status="failed", training_eligible=False, error=str(error)),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "dataset",
        "verification",
        "decisions",
        "references",
        "task",
        "output",
    ):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("dataset", "verification", "references"):
        parser.add_argument("--" + option + "-sha256", required=True)
    args = parser.parse_args()
    result = admit_scene_dataset(
        args.dataset,
        args.dataset_sha256,
        args.verification,
        args.verification_sha256,
        args.decisions,
        args.references,
        args.references_sha256,
        args.task,
        args.output,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "artifact_sha256"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
