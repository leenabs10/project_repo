"""Read-only IMU coverage and loader integration audit; no model evaluation."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingestion"))
from torwic_loader import SequenceImu, calibration, load_sample


def file_hash(path):
    """Return SHA-256 without retaining large file payloads in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_sequence(root, records, cal, *, sort, max_gap_seconds):
    """Audit every timestamp and decode representative samples with IMU attached.

    Sorting and gap limits are explicit inspection choices. Coverage failure is
    recorded rather than corrected by inventing data. October reads are solely
    integrity checks and must not be used for model or timing-policy selection.
    """
    reader = SequenceImu(root, records[0], sort=sort, max_gap_seconds=max_gap_seconds)
    source_times = np.loadtxt(reader.paths["left"].parent / "frame_times.txt", ndmin=1)
    if len(source_times) != len(records):
        raise ValueError("Manifest does not cover the complete sequence")
    for index, record in enumerate(records):
        if record["frame_id"] != f"{index:06d}":
            raise ValueError("Manifest frame order or coverage mismatch")
        if record["sample_id"] != f"{record['sequence_id']}/{record['frame_id']}":
            raise ValueError("Manifest sample ID disagrees with sequence/frame IDs")
        if not np.isclose(
            record["timestamp_seconds"], source_times[index], rtol=0, atol=1e-8
        ):
            raise ValueError("Manifest frame timestamp differs from raw frame_times")

    counts = {side: Counter() for side in reader.rows}
    gaps = {side: [] for side in reader.rows}
    invalid_records = []
    for record in records:
        for side, bracket in reader.bracket_record(record).items():
            counts[side][bracket["reason"]] += 1
            if np.isfinite(bracket["bracket_gap_seconds"]):
                gaps[side].append(bracket["bracket_gap_seconds"])
            if bracket["valid"]:
                # Compare all six columns; no units or axis meanings are assumed.
                np.testing.assert_array_equal(
                    bracket["measurement_rows"],
                    reader.rows[side][bracket["bracket_indices"], 1:],
                )
                before, after = bracket["bracket_times_seconds"]
                if not before <= record["timestamp_seconds"] <= after:
                    raise ValueError("Valid bracket does not contain the frame time")
            else:
                if not np.isnan(bracket["measurement_rows"]).all():
                    raise ValueError("Invalid bracket contains fabricated measurements")
                if not np.all(bracket["bracket_indices"] == -1):
                    raise ValueError("Invalid bracket exposes usable row indices")
                invalid_records.append(
                    {
                        "sample_id": record["sample_id"],
                        "side": side,
                        "reason": bracket["reason"],
                        "gap_seconds": (
                            bracket["bracket_gap_seconds"]
                            if np.isfinite(bracket["bracket_gap_seconds"])
                            else None
                        ),
                    }
                )

    representatives = sorted({0, len(records) // 2, len(records) - 1})
    loader_checks = []
    for index in representatives:
        sample = load_sample(root, records[index], cal, imu_sequence=reader)
        np.testing.assert_array_equal(
            np.isnan(sample["depth_m"]), ~sample["depth_valid"]
        )
        loader_checks.append(
            {
                "sample_id": sample["sample_id"],
                "imu_valid": {
                    side: sample["imu"][side]["valid"] for side in reader.rows
                },
                "native_rgb_shape": list(sample["rgb"].shape),
                "depth_mask_preserved": True,
            }
        )
    for side, path in reader.paths.items():
        if file_hash(path) != reader.reports[side]["source_sha256"]:
            raise ValueError("Raw IMU source changed during audit")
    return {
        "sequence_id": records[0]["sequence_id"],
        "split": records[0]["split"],
        "frames_checked": len(records),
        "frames_with_both_imu_brackets": len(records)
        - len({failure["sample_id"] for failure in invalid_records}),
        "coverage_status": "complete" if not invalid_records else "partial",
        "sides": {
            side: {
                "ordering": reader.reports[side],
                "coverage_counts": dict(counts[side]),
                "covered_frames": counts[side]["bracketed"]
                + counts[side]["exact_match"],
                "bracket_gap_seconds_p50_p99_max": (
                    np.quantile(gaps[side], [0.5, 0.99, 1]).tolist()
                    if gaps[side]
                    else None
                ),
            }
            for side in reader.rows
        },
        "invalid_frame_sides": invalid_records,
        "loader_checks": loader_checks,
        "raw_imu_hashes_unchanged": True,
        "measurement_rows_preserved": True,
    }


def main():
    """Write a fresh JSON report; uncovered frames are explicit limitations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sort-imu", action="store_true")
    parser.add_argument("--max-gap-seconds", type=float, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError("Choose a fresh report path outside the immutable raw root")
    manifest_payload = args.manifest.read_bytes()
    records = [
        json.loads(line)
        for line in manifest_payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not records or len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("Empty manifest or duplicate sample IDs")
    groups = {}
    for record in records:
        groups.setdefault(record["sequence_id"], []).append(record)
    cal = calibration(root)
    sequences, errors = [], []
    for sequence_id, group in sorted(groups.items()):
        try:
            sequence = review_sequence(
                root,
                group,
                cal,
                sort=args.sort_imu,
                max_gap_seconds=args.max_gap_seconds,
            )
            sequences.append(sequence)
            print(f"Reviewed {sequence_id}: {len(group)} frames", flush=True)
        except Exception as error:
            errors.append(
                {
                    "sequence_id": sequence_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    report = {
        "status": "passed" if not errors else "failed",
        "coverage_status": (
            "not_fully_audited"
            if errors
            else (
                "partial"
                if any(sequence["invalid_frame_sides"] for sequence in sequences)
                else "complete"
            )
        ),
        "scope": "Timestamp integrity and loader contracts; no fusion or model evaluation.",
        "max_gap_seconds": args.max_gap_seconds,
        "gap_limit_basis": "Explicit inspection tolerance; not certified for fusion.",
        "sorting_requested": args.sort_imu,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "calibration_sha256": file_hash(root / "calibrations.txt"),
        "source_hashes": {
            "torwic_loader.py": file_hash(
                Path(__file__).resolve().parents[1] / "ingestion" / "torwic_loader.py"
            ),
            "imu_timing.py": file_hash(
                Path(__file__).resolve().parents[1] / "ingestion" / "imu_timing.py"
            ),
            "imu_ingestion_review.py": file_hash(Path(__file__)),
        },
        "frames_checked": sum(sequence["frames_checked"] for sequence in sequences),
        "frame_side_queries_checked": 2
        * sum(sequence["frames_checked"] for sequence in sequences),
        "sequences": sequences,
        "errors": errors,
        "limitations": [
            "Invalid coverage remains masked; sorting does not supply missing readings.",
            "Measurement column meanings, units, sensor frames, and clock calibration remain unverified.",
            "Timestamp brackets do not establish interpolation or fusion readiness.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "frames_checked", "frame_side_queries_checked")
            }
        ),
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
