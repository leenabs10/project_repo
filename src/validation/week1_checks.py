"""Representative ingestion checks and calibration-direction diagnostics.
Requires numpy and Pillow. Raw files are read only; no model tuning is performed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Allow direct execution of this script without installing the project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingestion"))
from torwic_loader import calibration, load_sample, lidar_to_camera


def rotation(q):
    """Convert a quaternion in xyzw order to a 3x3 rotation matrix."""
    q = np.array(q, dtype=float)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def project(xyz, cal, camera, inverse, distort, width, height):
    """Return nearest axial depth per pixel and the number of points inside the image.

    Compare both interpretations of the supplied camera pose. Empty pixels contain
    infinity; the returned depth is camera-axis Z in metres, not Euclidean range.
    """
    prefix = "T_cam1_os" if camera == "Camera1" else "T_cam2_os"
    rotation_matrix = rotation([cal[f"{prefix}.q{a}"] for a in "xyzw"])
    translation = np.array([cal[f"{prefix}.p{a}"] for a in "xyz"])
    if inverse:
        side = "left" if camera == "Camera1" else "right"
        points = lidar_to_camera(xyz, cal, side)
    else:
        points = xyz @ rotation_matrix.T + translation
    # Reject nonfinite points and points behind or very close to the camera plane.
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0.05)]
    x, y = points[:, 0] / points[:, 2], points[:, 1] / points[:, 2]
    if distort:
        # Apply radial (k1, k2, k3) and tangential (p1, p2) lens distortion.
        k1, k2, p1, p2, k3 = cal[f"{camera}.D"]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        # Compute both coordinates together using the original, undistorted x and y.
        x, y = (
            x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
        )
    u = cal[f"{camera}.fx"] * x + cal[f"{camera}.cx"]
    v = cal[f"{camera}.fy"] * y + cal[f"{camera}.cy"]
    inside = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < width - 1)
        & (v >= 0)
        & (v < height - 1)
    )
    ui, vi = np.rint(u[inside]).astype(int), np.rint(v[inside]).astype(int)
    z = points[inside, 2]
    projected = np.full(width * height, np.inf)
    # Multiple points can round to one pixel; keep the closest surface (a Z-buffer).
    np.minimum.at(projected, vi * width + ui, z)
    return projected.reshape(height, width), int(inside.sum())


def main():
    """Check representative frames, compare training projections, and save a preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cal = calibration(root)
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped = {}
    for row in records:
        grouped.setdefault(row["sequence_id"], []).append(row)
    checks, diagnostics = [], []
    preview = None
    for sequence, rows in grouped.items():
        rows.sort(key=lambda r: int(r["frame_id"]))
        # Sample the start, midpoint, and end rather than decoding the full sequence.
        for index in (0, len(rows) // 2, len(rows) - 1):
            row = rows[index]
            sample = load_sample(root, row, cal)
            right = {}
            for key in ("rgb_right_path", "depth_right_path", "semantic_right_path"):
                with Image.open(root / row[key]) as im:
                    im.load()
                    right[key] = np.array(im)
            right_shape = (int(cal["Camera2.height"]), int(cal["Camera2.width"]))
            if (
                right["rgb_right_path"].shape != (*right_shape, 3)
                or right["depth_right_path"].shape != right_shape
                or right["semantic_right_path"].shape != right_shape
            ):
                raise ValueError(f'Right camera dimension mismatch: {row["sample_id"]}')
            if (
                right["depth_right_path"].dtype != np.uint16
                or right["semantic_right_path"].max() > 15
            ):
                raise ValueError(
                    f'Right depth dtype or semantic ID mismatch: {row["sample_id"]}'
                )
            if not np.array_equal(np.isnan(sample["depth_m"]), ~sample["depth_valid"]):
                raise ValueError("Depth validity mask mismatch")
            valid_depth = sample["depth_m"][sample["depth_valid"]]
            checks.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "status": "passed",
                    "both_camera_images_decoded": True,
                    "lidar_payload_decoded": True,
                    "left_positive_depth_fraction": float(sample["depth_valid"].mean()),
                    "right_positive_depth_fraction": float(
                        (right["depth_right_path"] > 0).mean()
                    ),
                    "left_depth_median_metres": float(np.median(valid_depth)),
                    "stored_lidar_points": len(sample["lidar_valid"]),
                    "finite_nonzero_lidar_points": len(sample["lidar_xyz"]),
                    "imu_time_span_contains_frame": row["imu_time_span_contains_frame"],
                }
            )
            # Spatial diagnostics use training frames only. Test/validation samples get decoding checks.
            if row["split"] != "train":
                continue
            xyz = sample["lidar_xyz"]
            for camera, depth in [
                ("Camera1", sample["depth_raw"]),
                ("Camera2", right["depth_right_path"]),
            ]:
                height, width = depth.shape
                for inverse in (False, True):
                    for distort in (False, True):
                        projected, point_count = project(
                            xyz, cal, camera, inverse, distort, width, height
                        )
                        overlap = np.isfinite(projected) & (depth > 0)
                        # Compare only pixels with valid depth from both sensors.
                        errors = np.abs(
                            projected[overlap] - depth[overlap].astype(float) * 0.001
                        )
                        entry = {
                            "sample_id": row["sample_id"],
                            "camera": camera,
                            "transform": (
                                "inverse: R.T @ (p_os - t)"
                                if inverse
                                else "direct: R @ p_os + t"
                            ),
                            "distortion_applied": distort,
                            "points_in_image": point_count,
                            "pixels_with_both_depths": int(overlap.sum()),
                            "absolute_depth_error_metres_p50_p90": (
                                np.percentile(errors, [50, 90]).tolist()
                                if len(errors)
                                else None
                            ),
                            "fraction_error_under_0_10m": (
                                float((errors < 0.1).mean()) if len(errors) else None
                            ),
                        }
                        diagnostics.append(entry)
                        if (
                            index == len(rows) // 2
                            and camera == "Camera1"
                            and inverse
                            and distort
                        ):
                            preview = (
                                sample["rgb"],
                                projected,
                                depth,
                                row["sample_id"],
                            )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "python_executable": sys.executable,
        "virtual_environment": sys.prefix != sys.base_prefix,
        "representative_samples_checked": len(checks),
        "sample_checks": checks,
        "projection_diagnostics_training_only": diagnostics,
        "interpretation": "Compare the supplied transform and its inverse; do not treat lowest depth error alone as proof of calibration accuracy. Occlusion, reflective surfaces, sensor range, and image distortion can cause residuals.",
        "coverage_limit": "First/middle/last samples decoded in each sequence; not a full-file integrity scan. No performance measurements or tuning on the test split.",
    }
    (args.output / "week1_checks.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if preview:
        rgb, projected, depth, label = preview
        image = Image.fromarray(rgb).resize((960, 540))
        canvas = Image.new("RGB", (960, 580), "#17202a")
        canvas.paste(image, (0, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 10),
            f"{label}: inverse transform + distortion; cyan=depth agreement <0.10m, orange=other",
            fill="white",
        )
        ys, xs = np.nonzero(np.isfinite(projected))
        # Limit marker density so the RGB background remains visible.
        step = max(1, len(xs) // 8000)
        for x, y in zip(xs[::step], ys[::step]):
            agree = (
                depth[y, x] > 0
                and abs(projected[y, x] - float(depth[y, x]) * 0.001) < 0.1
            )
            # The dataset's 1280x720 image is displayed at 960x540, below a 40px title.
            xx, yy = x * 0.75, y * 0.75 + 40
            draw.ellipse(
                (xx - 1, yy - 1, xx + 1, yy + 1), fill="#00dfff" if agree else "#ff9c38"
            )
        canvas.save(args.output / "lidar_rgb_overlay.png")
    print(
        json.dumps(
            {
                "checked": len(checks),
                "report": str(args.output / "week1_checks.json"),
                "projection_diagnostics": diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
