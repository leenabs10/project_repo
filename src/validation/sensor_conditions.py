"""Shared quality gates for controlled sensor conditions, independent of backends."""

import numpy as np
from PIL import Image


def check_variant(
    pair, rgb: np.ndarray, depth: np.ndarray, condition: dict, bounds: dict
):
    """Measure requested effect and invariants; return explicit rejection reasons.

    Hidden fraction uses originally valid measurements as its denominator. The
    coverage/clipping envelope is an appearance screen, not a realism certificate.
    """
    if rgb.shape != pair.rgb.shape or rgb.dtype != np.uint8:
        raise ValueError("RGB geometry/dtype changed")
    if depth.shape != pair.depth_raw.shape or depth.dtype != np.uint16:
        raise ValueError("Depth geometry/dtype changed")
    clean_valid = (pair.depth_raw != pair.invalid_raw_value) & (pair.depth_raw > 0)
    observed = (depth != pair.invalid_raw_value) & (depth > 0)
    if np.any(observed & ~clean_valid):
        raise ValueError("Originally invalid depth was invented")
    if not np.array_equal(depth[observed], pair.depth_raw[observed]):
        raise ValueError("Retained depth measurements changed")
    if not clean_valid.any():
        raise ValueError("Clean reference has no valid depth measurements")
    luma = np.asarray(Image.fromarray(rgb).convert("L"))
    metrics = {
        "mean_luma": float(luma.mean()),
        "bright_luma_fraction": float((luma >= 250).mean()),
        "valid_depth_fraction": float(observed.mean()),
        "hidden_valid_fraction": float(
            (clean_valid & ~observed).sum() / clean_valid.sum()
        ),
        "rgb_mean_absolute_change": float(
            np.abs(rgb.astype(np.float32) - pair.rgb).mean()
        ),
    }
    reasons = []
    for metric, limits in (
        ("mean_luma", bounds["mean_luma_range"]),
        ("valid_depth_fraction", bounds["valid_depth_fraction_range"]),
    ):
        if not limits[0] <= metrics[metric] <= limits[1]:
            reasons.append(metric + "_outside_training_envelope")
    if metrics["bright_luma_fraction"] > bounds["bright_luma_fraction_max"]:
        reasons.append("bright_luma_fraction_exceeds_training_envelope")
    if condition["method"] == "rgb_gain":
        if not np.array_equal(depth, pair.depth_raw):
            raise ValueError("RGB-only condition changed depth")
        if metrics["rgb_mean_absolute_change"] < bounds["rgb_mean_absolute_change_min"]:
            reasons.append("rgb_condition_has_insufficient_effect")
    else:
        if not np.array_equal(rgb, pair.rgb):
            raise ValueError("Depth-only condition changed RGB")
        if (
            not bounds["hidden_valid_fraction_min"]
            <= metrics["hidden_valid_fraction"]
            <= bounds["hidden_valid_fraction_max"]
        ):
            reasons.append("depth_hidden_fraction_outside_requested_bounds")
    return metrics, reasons
