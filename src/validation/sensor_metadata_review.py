"""Verify disabled real-data IMU policies and summarize unresolved metadata gates."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingestion"))
from imu_measurements import require_measurement_policy, source_hash
from image_geometry import require_projection_policy


def inspect_annotation_metadata(annotation_root):
    """Inspect JSON keys without assigning dates or splits from folder numbers.

    Candidate provenance fields are recorded for manual review, never treated as
    verified source identity. No trajectory RGB or held-out model data is opened.
    """
    key_counts = Counter()
    file_types = Counter()
    candidates = []
    hashes = {}
    words = ("timestamp", "collection", "trajectory", "sequence", "date", "filename")

    def visit(value, relative, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                key_counts[key] += 1
                location = f"{prefix}.{key}" if prefix else key
                if any(word in key.lower() for word in words):
                    candidates.append({"file": relative, "field": location})
                visit(item, relative, location)
        elif isinstance(value, list):
            for item in value:
                visit(item, relative, prefix)

    for path in sorted(annotation_root.rglob("*.json")):
        if not path.resolve().is_relative_to(annotation_root):
            raise ValueError("Annotation metadata path escapes its source root")
        payload = path.read_bytes()
        relative = path.relative_to(annotation_root).as_posix()
        hashes[relative] = hashlib.sha256(payload).hexdigest()
        file_types[path.name] += 1
        visit(json.loads(payload), relative)
    if not hashes:
        raise ValueError("No annotation JSON metadata found")
    return {
        "metadata_files": len(hashes),
        "file_types": dict(file_types),
        "field_names": sorted(key_counts),
        "candidate_provenance_fields": candidates,
        "provenance_verified": False,
        "metadata_hashes": hashes,
    }


def main():
    """Write a fresh review and return failure if a required block is bypassed.

    A successful review means protections work, not that sensor fusion is ready.
    This tool is intentionally scoped to the current unverified metadata state.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--imu-contract", type=Path, required=True)
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    annotations = args.annotations.resolve()
    output = args.output.resolve()
    if (
        output.is_relative_to(root)
        or output.is_relative_to(annotations)
        or output.exists()
    ):
        raise ValueError("Choose a fresh report path outside raw/annotation sources")
    imu = json.loads(args.imu_contract.read_text(encoding="utf-8"))
    geometry = json.loads(args.geometry_contract.read_text(encoding="utf-8"))
    if imu.get("interpolation_enabled") is not False:
        raise ValueError(
            "This review requires real-data interpolation to remain disabled"
        )
    if geometry.get("fusion_enabled") is not False:
        raise ValueError(
            "This review requires unverified geometry fusion to remain disabled"
        )
    checked_streams = []
    for relative, policy in sorted(imu["streams"].items()):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Contract source path escapes the dataset root")
        if source_hash(path) != policy["source_sha256"]:
            raise ValueError(
                f"Source hash differs from reviewed IMU stream: {relative}"
            )
        try:
            require_measurement_policy(root, relative, imu)
        except ValueError as error:
            if str(error) != "IMU measurement interpolation is disabled":
                raise
        else:
            raise ValueError("Unverified IMU interpolation was unexpectedly enabled")
        checked_streams.append(relative)
    if len(checked_streams) != 6:
        raise ValueError(
            "The review must cover both IMU sides in all three trajectories"
        )
    try:
        require_projection_policy(geometry, root / "calibrations.txt")
    except ValueError as error:
        if str(error) != "Camera-LiDAR fusion is disabled by the geometry contract":
            raise
    else:
        raise ValueError("Unverified camera-LiDAR fusion was unexpectedly enabled")
    annotation_review = inspect_annotation_metadata(annotations)
    report = {
        "status": "protections_verified_metadata_pending",
        "ready_for_real_imu_interpolation": False,
        "ready_for_camera_lidar_fusion": False,
        "extra_annotation_training_eligible": False,
        "imu_streams_checked": checked_streams,
        "raw_imu_hashes_match": True,
        "annotation_metadata": annotation_review,
        "trajectory_image_files_opened": 0,
        "contract_hashes": {
            "imu": source_hash(args.imu_contract),
            "geometry": source_hash(args.geometry_contract),
        },
        "code_hashes": {
            "imu_measurements.py": source_hash(
                Path(__file__).resolve().parents[1]
                / "ingestion"
                / "imu_measurements.py"
            ),
            "sensor_metadata_review.py": source_hash(Path(__file__)),
        },
        "remaining_requirements": [
            "Azure SLAM export or verified header showing the six measurement columns and SI units.",
            "Azure IMU coordinate-frame metadata and confirmed frame/IMU timestamp relationship.",
            "A declared and validated IMU resampling gap policy; inspection thresholds are not certification.",
            "Azure exported-image convention and matching camera_info/intrinsics.",
            "Extra annotation source collections and independent held-out overlap verification.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "imu_streams_checked": len(checked_streams),
                "annotation_metadata_files": annotation_review["metadata_files"],
                "candidate_provenance_fields": len(
                    annotation_review["candidate_provenance_fields"]
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
