"""Camera-axis depth projection for calibrated training-data diagnostics."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingestion"))
from torwic_loader import lidar_to_camera


def project_lidar_depth(xyz, cal, side, *, apply_distortion=False):
    """Project inverse-transform camera Z in metres, retaining nearest surface.

    Empty pixels contain infinity. The OpenCV five-coefficient distortion model
    is optional because the exported image convention still needs validation.
    This function performs neither motion compensation nor temporal alignment.
    """
    camera = "Camera1" if side == "left" else "Camera2"
    width, height = int(cal[camera + ".width"]), int(cal[camera + ".height"])
    points = lidar_to_camera(xyz, cal, side)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0.05)]
    x, y = points[:, 0] / points[:, 2], points[:, 1] / points[:, 2]
    if apply_distortion:
        k1, k2, p1, p2, k3 = cal[camera + ".D"]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x, y = (
            x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
        )
    u = cal[camera + ".fx"] * x + cal[camera + ".cx"]
    v = cal[camera + ".fy"] * y + cal[camera + ".cy"]
    inside = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < width - 1)
        & (v >= 0)
        & (v < height - 1)
    )
    ui, vi = np.rint(u[inside]).astype(int), np.rint(v[inside]).astype(int)
    depth = np.full(height * width, np.inf)
    np.minimum.at(depth, vi * width + ui, points[inside, 2])
    return depth.reshape(height, width), int(inside.sum())


def stable_depth_pixels(raw, *, spread_metres=0.1):
    """Identify five-point neighborhoods with valid, locally consistent depth.

    This is a diagnostic edge filter, not a ground-truth validity rule. All valid
    overlapping pixels are also scored separately to avoid hiding hard regions.
    """
    raw = np.asarray(raw)
    stable = np.zeros(raw.shape, dtype=bool)
    neighbors = np.stack(
        [raw[1:-1, 1:-1], raw[:-2, 1:-1], raw[2:, 1:-1], raw[1:-1, :-2], raw[1:-1, 2:]]
    )
    stable[1:-1, 1:-1] = np.all(neighbors > 0, axis=0) & (
        (neighbors.max(axis=0).astype(float) - neighbors.min(axis=0)) * 0.001
        <= spread_metres
    )
    return stable
