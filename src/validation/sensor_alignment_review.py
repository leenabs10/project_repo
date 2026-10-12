"""Review camera/LiDAR overlap on evenly spaced training frames; no calibration fitting."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingestion"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "profiling"))
from torwic_loader import calibration, read_xyz
from dataset_profile import safe_path, select_training
from sensor_projection import project_lidar_depth, stable_depth_pixels


def residual_metrics(projected, raw, mask):
    """Axial-depth agreement at overlapping pixels, not calibration accuracy."""
    if not mask.any():
        return {
            "overlap_pixels": 0,
            "absolute_error_metres_p50_p90_p99": None,
            "fraction_under_10cm": None,
        }
    error = np.abs(projected[mask] - raw[mask].astype(float) * 0.001)
    return {
        "overlap_pixels": int(mask.sum()),
        "absolute_error_metres_p50_p90_p99": np.quantile(
            error, [0.5, 0.9, 0.99]
        ).tolist(),
        "fraction_under_10cm": float((error < 0.1).mean()),
    }


def overlay(rgb, projected, raw, title, destination):
    """Export a diagnostic overlay only; never write back into source RGB."""
    base = np.array(rgb.resize((640, 360), Image.Resampling.BOX))
    overlap = np.isfinite(projected) & (raw > 0)
    ys, xs = np.nonzero(overlap)
    ys, xs = ys[::4], xs[::4]
    good = np.abs(projected[ys, xs] - raw[ys, xs].astype(float) * 0.001) < 0.1
    base[ys // 2, xs // 2] = np.where(
        good[:, None], np.array([0, 220, 255]), np.array([255, 130, 40])
    )
    canvas = Image.new("RGB", (640, 390), "white")
    canvas.paste(Image.fromarray(base), (0, 30))
    ImageDraw.Draw(canvas).text((8, 8), title, fill="black")
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("At least three evenly spaced training samples required")
    root = args.root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(root):
        parser.error("Reports must be outside the raw dataset root")
    payload = args.manifest.read_bytes()
    records = select_training(
        [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    )
    indices = np.unique(
        np.linspace(0, len(records) - 1, min(args.samples, len(records)), dtype=int)
    )
    cal = calibration(root)
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for position, index in enumerate(indices):
        row = records[index]
        xyz, _ = read_xyz(safe_path(root, row["lidar_path"]))
        for side in ["left", "right"]:
            depth_key = "depth_path" if side == "left" else "depth_right_path"
            rgb_key = "rgb_path" if side == "left" else "rgb_right_path"
            with Image.open(safe_path(root, row[depth_key])) as image:
                raw = np.array(image)
            stable = stable_depth_pixels(raw)
            for distorted in [False, True]:
                projected, count = project_lidar_depth(
                    xyz, cal, side, apply_distortion=distorted
                )
                valid = np.isfinite(projected) & (raw > 0)
                diagnostics.append(
                    {
                        "sample_id": row["sample_id"],
                        "camera": side,
                        "apply_distortion": distorted,
                        "points_in_image": count,
                        "all_overlap": residual_metrics(projected, raw, valid),
                        "stable_overlap": residual_metrics(
                            projected, raw, valid & stable
                        ),
                    }
                )
                if position in [0, len(indices) // 2, len(indices) - 1]:
                    with Image.open(safe_path(root, row[rgb_key])) as rgb:
                        name = f"{row['frame_id']}_{side}_{'distorted' if distorted else 'pinhole'}.png"
                        overlay(
                            rgb,
                            projected,
                            raw,
                            f"{row['sample_id']} {side} distortion={distorted}",
                            output / name,
                        )
        print(f"Reviewed {position+1}/{len(indices)} training samples", flush=True)
    summary = {}
    for side in ["left", "right"]:
        for distorted in [False, True]:
            selected = [
                d
                for d in diagnostics
                if d["camera"] == side and d["apply_distortion"] == distorted
            ]
            summary[f"{side}_{'distorted' if distorted else 'pinhole'}"] = {}
            for region in ["all_overlap", "stable_overlap"]:
                medians = [
                    d[region]["absolute_error_metres_p50_p90_p99"][0]
                    for d in selected
                    if d[region]["overlap_pixels"]
                ]
                summary[f"{side}_{'distorted' if distorted else 'pinhole'}"][region] = {
                    "per_frame_median_error_metres_min_median_max": (
                        np.quantile(medians, [0, 0.5, 1]).tolist() if medians else None
                    ),
                    "overlap_pixels": sum(
                        d[region]["overlap_pixels"] for d in selected
                    ),
                }
    report = {
        "status": "review_completed",
        "frames_reviewed": len(indices),
        "test_sensor_files_opened": 0,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "calibration_sha256": hashlib.sha256(
            (root / "calibrations.txt").read_bytes()
        ).hexdigest(),
        "transform": "p_camera = R.T @ (p_os - t)",
        "stable_depth_spread_metres": 0.1,
        "summary": summary,
        "diagnostics": diagnostics,
        "distortion_convention_resolved": False,
        "production_registration_certified": False,
        "limitations": [
            "Projection overlap is not independent calibration ground truth.",
            "Occlusion, finite LiDAR beams, depth failures, object motion and residual timing affect comparisons.",
            "Both distortion variants retained; no sensor parameters fitted or test data used.",
            "Publisher reports RGB-aligned depth and PTP-synchronized mounted cameras/LiDAR; per-sensor timing accuracy is not measured here.",
        ],
        "publisher_source": "https://github.com/Viky397/TorWICDataset#torwic-slam-faq",
    }
    (output / "alignment_review.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
