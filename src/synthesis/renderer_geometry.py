"""Pure geometry contracts for clean pinhole synthetic depth and class labels."""

import numpy as np


def ray_lengths(camera):
    """Return the unit-axial ray length at each integer pixel center (H, W).

    ``camera`` uses OpenCV pixel indices: X right, Y down, Z forward; focal
    lengths/principal point are in pixels. This synthetic contract has zero
    distortion and does not resolve the real TorWIC image export convention.
    """
    width, height = camera["width"], camera["height"]
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or min(width, height) <= 0
    ):
        raise ValueError("Camera dimensions must be positive integers")
    values = np.array([camera[key] for key in ("fx", "fy", "cx", "cy")], dtype=float)
    if not np.isfinite(values).all() or min(values[:2]) <= 0:
        raise ValueError("Finite intrinsics and positive focal lengths are required")
    vertical, horizontal = np.indices((height, width), dtype=np.float64)
    return np.sqrt(
        1
        + ((horizontal - camera["cx"]) / camera["fx"]) ** 2
        + ((vertical - camera["cy"]) / camera["fy"]) ** 2
    )


def axial_depth(depth_pass, camera, convention, *, far_clip_metres):
    """Return float32 axial metres and a bool mask, preserving invalids as NaN.

    The renderer pass convention must be established with an analytic fixture.
    No interpolation, noise, denoising or depth-edge blending is performed.
    """
    lengths = ray_lengths(camera)
    raw = np.asarray(depth_pass, dtype=np.float64)
    if raw.shape != lengths.shape:
        raise ValueError("Depth shape differs from the camera contract")
    if not np.isfinite(far_clip_metres) or far_clip_metres <= 0:
        raise ValueError("Far clip must be positive and finite")
    if convention == "euclidean_ray_range":
        depth = raw / lengths
    elif convention == "axial_z":
        depth = raw.copy()
    else:
        raise ValueError("Unknown renderer depth convention")
    valid = np.isfinite(depth) & (depth > 0) & (depth < far_clip_metres)
    result = np.full(raw.shape, np.nan, dtype=np.float32)
    result[valid] = depth[valid]
    return result, valid


def semantic_ids(index_pass):
    """Return discrete uint8 classes 0..15; reject blended or unknown index values."""
    raw = np.asarray(index_pass)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("Expected a finite two-dimensional class-index pass")
    rounded = np.rint(raw)
    if np.max(np.abs(raw - rounded)) > 1e-5 or np.any((rounded < 0) | (rounded > 15)):
        raise ValueError("Renderer labels contain blended or unknown semantic IDs")
    return rounded.astype(np.uint8)
