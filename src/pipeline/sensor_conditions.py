"""Traceable training-only condition generation with explicit quality quarantine."""

import copy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

from src.adapters.generation.capabilities import BACKENDS
from src.adapters.generation.sensor_conditions import (
    choose_donor,
    condition_seed,
    transform_pair,
)
from src.validation.sensor_conditions import check_variant
from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.core.contracts import validate_job, validate_record, validate_sensor_policy
from src.pipeline.cache import validate_membership
from src.validation.sensor_review import write_review


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _checked_png(path: Path, array: np.ndarray) -> None:
    """Write losslessly and decode the result before declaring export integrity."""
    Image.fromarray(array).save(path)
    with Image.open(path) as image:
        restored = np.array(image)
    if restored.dtype != array.dtype or not np.array_equal(restored, array):
        raise ValueError("Lossless PNG round-trip failed")


def _snapshot(adapter, record, output, pair):
    """Copy clean bytes once; preserve canonical units, pose, labels and frame names."""
    clean = copy.deepcopy(record)
    identity = hashlib.sha256(record["sample_id"].encode()).hexdigest()[:24]
    folder = output / "references" / identity
    folder.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    source_files = {}
    for name, reference in clean["modalities"].items():
        if reference["status"] != "available":
            continue
        if name not in {"rgb", "depth", "semantic", "lidar"}:
            raise NotImplementedError(
                "Preservation of this modality is not implemented"
            )
        source = source_path(adapter.root, reference["path"])
        before = file_hash(source)
        if reference.get("sha256", before) != before:
            raise ValueError("Source payload differs from its declared checksum")
        if name in {"rgb", "depth"} and pair.provenance[name + "_sha256"] != before:
            raise ValueError("Decoded pair changed before reference export")
        destination = folder / (name + source.suffix.lower())
        shutil.copyfile(source, destination)
        if file_hash(destination) != before or file_hash(source) != before:
            raise ValueError("Clean source copy is not byte-identical")
        if name == "semantic":
            with Image.open(destination) as image:
                labels = np.array(image)
            if labels.shape != pair.depth_raw.shape or labels.dtype.kind not in "ui":
                raise ValueError(
                    "Semantic reference must be an integer camera-sized mask"
                )
        source_hashes[name] = before
        source_files[reference["path"]] = before
        reference.update(path=destination.relative_to(output).as_posix(), sha256=before)
    return validate_record(clean), source_hashes, source_files


def _preflight(adapter, job, policy, records):
    """Reject unsupported requests and held-out donors before opening sensor files."""
    validate_job(job)
    validate_sensor_policy(policy)
    BACKENDS[job["backend"]].require(set(job["required_modalities"]))
    if any(item["dataset_id"] != adapter.dataset_id for item in records):
        raise ValueError("Adapter returned a different dataset")
    validate_membership(records)
    if (
        job["dataset_id"] != adapter.dataset_id
        or policy["dataset_id"] != adapter.dataset_id
        or job["source_manifest_sha256"] != adapter.manifest_sha256
        or policy["source_manifest_sha256"] != adapter.manifest_sha256
        or file_hash(adapter.manifest) != adapter.manifest_sha256
    ):
        raise ValueError("Dataset/manifest binding mismatch")
    by_id = {record["sample_id"]: record for record in records}
    selected_ids = set(job["source_sample_ids"])
    donor_ids = set()
    for condition in job["conditions"]:
        if condition["method"] == "rgb_gain":
            low, high = policy["bounds"]["rgb_gain_range"]
            if not low <= condition["gain"] <= high:
                raise ValueError("Requested RGB gain exceeds training policy")
        else:
            donor_ids.update(condition["donor_sample_ids"])
    for sample_id in selected_ids | donor_ids:
        record = by_id.get(sample_id)
        if record is None or record["split"] != "train":
            raise ValueError(
                "Every source and donor must be a declared training sample"
            )
        if record["group_id"] not in policy["training_groups"]:
            raise ValueError("Source/donor group lacks training-policy evidence")
        if (
            policy.get("sensor_id")
            and record.get("camera", {}).get("sensor_id") != policy["sensor_id"]
        ):
            raise ValueError("Quality-policy camera does not match source/donor")
        calibration = record["source_metadata"].get("calibration_sha256")
        if calibration and policy["evidence_sha256"].get("calibration") != calibration:
            raise ValueError("Calibration does not match quality evidence")
        names = (
            job["required_modalities"]
            if sample_id in selected_ids
            else ["rgb", "depth"]
        )
        if any(
            record["modalities"].get(name, {}).get("status") != "available"
            for name in names
        ):
            raise ValueError("Requested source modality is unavailable")
        for reference in record["modalities"].values():
            if reference["status"] == "available":
                source_path(adapter.root, reference["path"])
        if sample_id in selected_ids and any(
            name not in {"rgb", "depth", "semantic", "lidar"}
            and reference["status"] == "available"
            for name, reference in record["modalities"].items()
        ):
            raise NotImplementedError(
                "Preservation of this modality is not implemented"
            )
    # Validate donor geometry before reading; masking is a stress operation, not registration.
    for sample_id in selected_ids:
        record = by_id[sample_id]
        for condition in job["conditions"]:
            if condition["method"] == "depth_mask_transfer":
                seed = condition_seed(
                    job["seed"],
                    adapter.dataset_id,
                    sample_id,
                    condition["condition_id"],
                )
                donor = by_id[choose_donor(condition, sample_id, seed)]
                if record["image_size"] != donor["image_size"] or record.get(
                    "camera"
                ) != donor.get("camera"):
                    raise ValueError("Donor must use the same native camera geometry")
    return by_id


def _remember_sources(known: dict, incoming: dict) -> None:
    """A repeated donor/source cannot acquire new bytes partway through a job."""
    for path, digest in incoming.items():
        if path in known and known[path] != digest:
            raise ValueError("Source payload changed between reads")
        known[path] = digest


def generate_conditions(
    adapter, job: dict, policy_path: Path, output: Path, evidence_files: dict
) -> dict:
    """Generate a bounded native batch with one source/donor pair in memory at a time.

    Accepted means automated export/condition checks passed. Every output remains
    ineligible for training until appearance and target provenance are reviewed.
    There is no scene generation, calibrated exposure/noise or joint sensor model.
    """
    started = time.perf_counter()
    validate_job(job)
    policy_path = Path(policy_path)
    if file_hash(policy_path) != job["quality_policy_sha256"]:
        raise ValueError("Quality policy checksum differs from the job")
    policy = validate_sensor_policy(json.loads(policy_path.read_text(encoding="utf-8")))
    if set(evidence_files) != set(policy["evidence_sha256"]):
        raise ValueError("Runtime must supply every declared quality evidence file")
    for key, digest in policy["evidence_sha256"].items():
        if file_hash(Path(evidence_files[key])) != digest:
            raise ValueError("Quality evidence checksum mismatch: " + key)
    by_id = _preflight(adapter, job, policy, list(adapter.records()))
    output = reserve_output(
        output,
        (
            adapter.root,
            adapter.manifest,
            policy_path,
            *(Path(path) for path in evidence_files.values()),
        ),
    )
    report = {
        "schema_version": job["schema_version"],
        "job_id": job["job_id"],
        "dataset_id": adapter.dataset_id,
        "status": "running",
        "manifest_sha256": adapter.manifest_sha256,
        "backend": job["backend"],
        "source_split": "train",
        "training_eligible": False,
        "human_review": "pending",
        "physical_sensor_realism_certified": False,
        "validation_sensor_files_opened": 0,
        "test_sensor_files_opened": 0,
        "limitations": [
            "RGB gain is an 8-bit stress transform, not calibrated exposure or low-light noise.",
            "Donor masks transfer missingness only, not geometry, measurements or a physical failure model.",
            "LiDAR is preserved byte-for-byte; no simulated/corrupted LiDAR is generated.",
            "Unchanged geometry does not resolve real camera-LiDAR registration or certify joint sensor fidelity.",
            "Preserved label provenance is not independent semantic ground truth.",
        ],
    }
    _json(output / "profile.json", report)
    try:
        _json(output / "job.json", job)
        _json(output / "quality_policy.json", policy)
        code_root = Path(__file__).parents[1]
        code_paths = set(code_root.rglob("*.py"))
        code_paths.add(Path(inspect.getfile(type(adapter))))
        code_paths.update((code_root / "core/schemas").glob("*.json"))
        _json(
            output / "implementation.json",
            {
                "files": {
                    path.relative_to(code_root.parent).as_posix(): file_hash(path)
                    for path in sorted(code_paths)
                },
                "python": sys.version,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "Pillow", "jsonschema")
                },
            },
        )
        attempts, clean_records, source_files = [], [], {}
        for sample_id in job["source_sample_ids"]:
            source_record = by_id[sample_id]
            pair = adapter.read_rgb_depth(source_record)
            clean, source_hashes, files = _snapshot(
                adapter, source_record, output, pair
            )
            clean_records.append(clean)
            _remember_sources(source_files, files)
            for condition in job["conditions"]:
                seed = condition_seed(
                    job["seed"],
                    adapter.dataset_id,
                    sample_id,
                    condition["condition_id"],
                )
                donor, donor_metadata = None, None
                if condition["method"] == "depth_mask_transfer":
                    donor_id = choose_donor(condition, sample_id, seed)
                    donor = adapter.read_rgb_depth(by_id[donor_id])
                    donor_metadata = {
                        "sample_id": donor_id,
                        "split": "train",
                        "depth_path": donor.provenance["depth_path"],
                        "depth_sha256": donor.provenance["depth_sha256"],
                        "invalid_raw_value": donor.invalid_raw_value,
                    }
                    _remember_sources(
                        source_files,
                        {
                            donor.provenance[name + "_path"]: donor.provenance[
                                name + "_sha256"
                            ]
                            for name in ("rgb", "depth")
                        },
                    )
                rgb, depth = transform_pair(pair, condition, donor)
                metrics, reasons = check_variant(
                    pair, rgb, depth, condition, policy["bounds"]
                )
                variant_id = (
                    "synthetic-"
                    + hashlib.sha256(
                        json.dumps(
                            [
                                adapter.dataset_id,
                                sample_id,
                                job["quality_policy_sha256"],
                                job["seed"],
                                condition,
                                source_hashes,
                                donor_metadata,
                                "sensor_conditions/1.0",
                            ],
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()[:24]
                )
                attempt = {
                    "variant_id": variant_id,
                    "source_sample_id": sample_id,
                    "source_group_id": source_record["group_id"],
                    "source_split": "train",
                    "condition": condition,
                    "seed": seed,
                    "source_sha256": source_hashes,
                    "donor": donor_metadata,
                    "metrics": metrics,
                    "automated_status": "rejected" if reasons else "passed",
                    "rejection_reasons": reasons,
                    "human_review": "pending" if not reasons else "not_applicable",
                    "training_eligible": False,
                }
                if not reasons:
                    folder = output / "variants" / variant_id
                    folder.mkdir(parents=True, exist_ok=False)
                    _checked_png(folder / "rgb.png", rgb)
                    _checked_png(folder / "depth.png", depth)
                    observed = (depth != pair.invalid_raw_value) & (depth > 0)
                    hidden = (
                        (pair.depth_raw != pair.invalid_raw_value)
                        & (pair.depth_raw > 0)
                        & ~observed
                    )
                    _checked_png(
                        folder / "observed_valid.png", observed.astype(np.uint8) * 255
                    )
                    _checked_png(
                        folder / "hidden_valid.png", hidden.astype(np.uint8) * 255
                    )
                    generated = copy.deepcopy(clean)
                    generated["sample_id"] = variant_id
                    for name in ("rgb", "depth"):
                        path = folder / (name + ".png")
                        generated["modalities"][name].update(
                            path=path.relative_to(output).as_posix(),
                            sha256=file_hash(path),
                        )
                    validate_record(generated)
                    attempt["record"] = generated
                    attempt["masks"] = {
                        name: (folder / (name + ".png")).relative_to(output).as_posix()
                        for name in ("observed_valid", "hidden_valid")
                    }
                attempts.append(attempt)
        # Fail the entire attempt if any bound input changed during generation.
        for relative, digest in source_files.items():
            if file_hash(source_path(adapter.root, relative)) != digest:
                raise ValueError("Source payload changed during generation")
        if (
            file_hash(adapter.manifest) != adapter.manifest_sha256
            or file_hash(policy_path) != job["quality_policy_sha256"]
        ):
            raise ValueError("Bound manifest/policy changed during generation")
        for key, digest in policy["evidence_sha256"].items():
            if file_hash(Path(evidence_files[key])) != digest:
                raise ValueError("Quality evidence changed during generation")
        for name, rows in (
            ("clean_manifest.jsonl", clean_records),
            ("attempts.jsonl", attempts),
        ):
            (output / name).write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
                encoding="utf-8",
            )
        write_review(output, clean_records, attempts)
        passed = sum(attempt["automated_status"] == "passed" for attempt in attempts)
        coverage = {}
        for condition in job["conditions"]:
            rows = [
                attempt
                for attempt in attempts
                if attempt["condition"]["condition_id"] == condition["condition_id"]
            ]
            coverage[condition["condition_id"]] = {
                "requested": len(job["source_sample_ids"]),
                "automated_pass": sum(
                    row["automated_status"] == "passed" for row in rows
                ),
                "rejected": sum(row["automated_status"] == "rejected" for row in rows),
                "human_approved": 0,
            }
        report.update(
            status="passed",
            source_samples=len(clean_records),
            attempted_variants=len(attempts),
            accepted_pending_review=passed,
            rejected_variants=len(attempts) - passed,
            quality_outcome="review_required" if passed else "all_rejected",
            condition_coverage=coverage,
            automated_coverage_complete=passed == len(attempts),
            delivered_coverage_complete=False,
            unique_source_files_verified=len(source_files),
            elapsed_seconds=time.perf_counter() - started,
            artifact_sha256={
                path.relative_to(output).as_posix(): file_hash(path)
                for path in output.rglob("*")
                if path.is_file() and path.name != "profile.json"
            },
        )
        _json(output / "profile.json", report)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        _json(output / "profile.json", report)
        raise
