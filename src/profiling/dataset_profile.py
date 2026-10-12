"""Profile training sensors at native resolution without changing raw data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingestion"))
from torwic_loader import CLASSES, MODALITIES, calibration, read_xyz
from imu_timing import load_imu, bracket_frames


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest path escapes the dataset root")
    return path


def quantiles(values):
    """Describe a distribution without presenting frame percentiles as pooled pixels."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite distribution")
    return dict(
        zip(
            ["min", "p01", "p10", "p50", "p90", "p99", "max"],
            np.quantile(values, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]).tolist(),
        )
    )


def select_training(records):
    """Fail closed on unknown training trajectories or duplicate sample IDs."""
    selected = sorted(
        [r for r in records if r["split"] == "train"], key=lambda r: int(r["frame_id"])
    )
    if not selected or any(r["sequence_id"] != "Aisle_CCW_Run_1" for r in selected):
        raise ValueError("Expected the confirmed June 15 training trajectory only")
    if len({r["sample_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate training sample IDs")
    for row in selected:
        if row["sample_id"] != f"Aisle_CCW_Run_1/{row['frame_id']}":
            raise ValueError("Training sample ID and frame ID disagree")
        for key in list(MODALITIES) + ["imu_left_path", "imu_right_path"]:
            if Path(row[key]).parts[:2] != ("Aisle_CCW_Run_1", "Aisle_CCW_Run_1"):
                raise ValueError(
                    "Training record points outside the confirmed training sequence"
                )
    return selected


def profile_frame(root, record, cal):
    result = {k: record[k] for k in ["sample_id", "timestamp_seconds", "frame_id"]}
    result["pose_position_metres"] = record["pose_xyz_qxyzw"][:3]
    for side, camera in [("left", "Camera1"), ("right", "Camera2")]:
        keys = (
            ("rgb_path", "depth_path", "semantic_path")
            if side == "left"
            else ("rgb_right_path", "depth_right_path", "semantic_right_path")
        )
        expected = (int(cal[camera + ".height"]), int(cal[camera + ".width"]))
        with Image.open(safe_path(root, record[keys[0]])) as image:
            image.load()
            rgb = np.array(image)
            gray = np.array(image.convert("L"))
            # A reproducible sharpness proxy, not a calibrated blur/noise label.
            small = np.asarray(
                image.convert("L").resize((320, 180), Image.Resampling.BOX),
                dtype=np.float32,
            )
        if rgb.shape != (*expected, 3) or rgb.dtype != np.uint8:
            raise ValueError("Unexpected native RGB format")
        laplacian = (
            small[:-2, 1:-1]
            + small[2:, 1:-1]
            + small[1:-1, :-2]
            + small[1:-1, 2:]
            - 4 * small[1:-1, 1:-1]
        )
        with Image.open(safe_path(root, record[keys[1]])) as image:
            depth = np.array(image)
        with Image.open(safe_path(root, record[keys[2]])) as image:
            semantic = np.array(image)
        if depth.shape != expected or depth.dtype != np.uint16:
            raise ValueError("Unexpected native depth format")
        if (
            semantic.shape != expected
            or semantic.dtype != np.uint8
            or semantic.max() >= len(CLASSES)
        ):
            raise ValueError("Unexpected semantic format or class ID")
        values = depth[depth > 0].astype(np.float32) * 0.001
        if not len(values):
            raise ValueError("No usable depth")
        result[side] = {
            "luma_mean_8bit": float(gray.mean()),
            "luma_std_8bit": float(gray.std()),
            "dark_luma_fraction": float((gray <= 5).mean()),
            "bright_luma_fraction": float((gray >= 250).mean()),
            "laplacian_variance_320x180": float(laplacian.var()),
            "valid_depth_fraction": float((depth > 0).mean()),
            "depth_metres_p01_p50_p90_p99": np.quantile(
                values, [0.01, 0.5, 0.9, 0.99]
            ).tolist(),
            "depth_pixels_over_20m": int((depth > 20000).sum()),
            "semantic_pixel_counts": np.bincount(
                semantic.ravel(), minlength=len(CLASSES)
            ).tolist(),
        }
    xyz, valid = read_xyz(safe_path(root, record["lidar_path"]))
    ranges = np.linalg.norm(xyz, axis=1)
    result["lidar"] = {
        "stored_points": len(valid),
        "usable_points": len(xyz),
        "usable_fraction": float(valid.mean()),
        "range_metres_p01_p50_p90_p99": np.quantile(
            ranges, [0.01, 0.5, 0.9, 0.99]
        ).tolist(),
    }
    return result


def run(root, manifest, output, workers):
    payload = manifest.read_bytes()
    records = select_training(
        [
            json.loads(line)
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        ]
    )
    cal = calibration(root)
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Reports must be outside the raw dataset root")
    source_times = np.loadtxt(
        safe_path(root, records[0]["imu_left_path"]).parent / "frame_times.txt", ndmin=1
    )
    if len(source_times) != len(records) or not np.allclose(
        source_times, [r["timestamp_seconds"] for r in records], rtol=0, atol=1e-8
    ):
        raise ValueError(
            "Training manifest does not cover the complete sequence timestamps"
        )
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    profiles = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, row in enumerate(
            pool.map(lambda r: profile_frame(root, r, cal), records), 1
        ):
            profiles.append(row)
            if count % 100 == 0 or count == len(records):
                print(f"Profiled {count}/{len(records)} training frames", flush=True)
    (output / "frame_metrics.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in profiles),
        encoding="utf-8",
    )
    sides = {}
    for side in ["left", "right"]:
        counts = np.sum(
            [p[side]["semantic_pixel_counts"] for p in profiles], axis=0, dtype=np.int64
        )
        sides[side] = {
            key: quantiles([p[side][key] for p in profiles])
            for key in [
                "luma_mean_8bit",
                "luma_std_8bit",
                "dark_luma_fraction",
                "bright_luma_fraction",
                "laplacian_variance_320x180",
                "valid_depth_fraction",
            ]
        }
        sides[side]["per_frame_depth_p99_metres"] = quantiles(
            [p[side]["depth_metres_p01_p50_p90_p99"][3] for p in profiles]
        )
        sides[side]["pixels_over_20m"] = sum(
            p[side]["depth_pixels_over_20m"] for p in profiles
        )
        sides[side]["semantic_classes"] = {
            str(i): {
                "name": name,
                "pixel_fraction": float(counts[i] / counts.sum()),
                "frames_present": sum(
                    p[side]["semantic_pixel_counts"][i] > 0 for p in profiles
                ),
            }
            for i, name in enumerate(CLASSES)
        }
    times = np.array([r["timestamp_seconds"] for r in records])
    if not np.all(np.diff(times) > 0):
        raise ValueError("Training timestamps must increase strictly")
    positions = np.array([r["pose_xyz_qxyzw"][:3] for r in records], dtype=float)
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    imu = {}
    for side in ["left", "right"]:
        rows, info = load_imu(
            safe_path(root, records[0][f"imu_{side}_path"]), sort=True
        )
        _, _, covered, gaps = bracket_frames(rows[:, 0], times)
        info["frames_outside_span"] = int((~covered).sum())
        info["covered_frame_bracket_gap_seconds"] = quantiles(gaps[covered])
        imu[side] = info
    summary = {
        "status": "passed",
        "split": "train",
        "sequence": "Aisle_CCW_Run_1",
        "frames_profiled": len(profiles),
        "sensor_payloads_decoded": len(profiles) * 7,
        "test_sensor_files_opened": 0,
        "validation_sensor_files_opened": 0,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "calibration_sha256": hashlib.sha256(
            (root / "calibrations.txt").read_bytes()
        ).hexdigest(),
        "profile_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "cameras": sides,
        "lidar_usable_points": quantiles(
            [p["lidar"]["usable_points"] for p in profiles]
        ),
        "lidar_usable_fraction": quantiles(
            [p["lidar"]["usable_fraction"] for p in profiles]
        ),
        "per_frame_lidar_range_p99_metres": quantiles(
            [p["lidar"]["range_metres_p01_p50_p90_p99"][3] for p in profiles]
        ),
        "frame_interval_seconds": quantiles(np.diff(times)),
        "imu": imu,
        "trajectory_path_length_metres": float(distances.sum()),
        "pose_translation_speed_metres_per_second": quantiles(
            distances / np.diff(times)
        ),
        "runtime_seconds": time.perf_counter() - started,
        "workers": workers,
        "limitations": [
            "Luma is an image proxy, not physical illumination; Laplacian variance also depends on scene texture.",
            "Frame-level percentiles are not pooled pixel/point distributions.",
            "Trajectory semantic masks are model predictions, not human ground truth.",
            "No sensor-noise calibration, augmentation tuning or test evaluation is performed.",
            "IMU vectors retained without interpreting column units; bracketing does not interpolate or establish hardware timing accuracy.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    summary = run(args.root.resolve(), args.manifest, args.output, args.workers)
    print(
        json.dumps(
            {k: summary[k] for k in ["status", "frames_profiled", "runtime_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
