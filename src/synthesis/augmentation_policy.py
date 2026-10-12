"""Training-derived controlled augmentation proposals and paired-data checks."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def build_policy(profile):
    """Derive camera-specific bounds from verified training distributions only.

    Bounds describe dataset statistics, not calibrated sensor noise. Photometric
    gain is an 8-bit image stress transform, not a physical exposure model.
    """
    if (
        profile.get("status") != "passed"
        or profile.get("split") != "train"
        or profile.get("sequence") != "Aisle_CCW_Run_1"
        or profile.get("validation_sensor_files_opened") != 0
        or profile.get("test_sensor_files_opened") != 0
    ):
        raise ValueError("Policy requires confirmed training-only profiling evidence")
    cameras = {}
    for side in ["left", "right"]:
        camera = profile["cameras"][side]
        luma = camera["luma_mean_8bit"]
        if not (0 < luma["p10"] <= luma["p50"] <= luma["p90"] <= 255):
            raise ValueError("Invalid training luma quantiles")
        cameras[side] = {
            "rgb_gain_range": [luma["p10"] / luma["p50"], luma["p90"] / luma["p50"]],
            "accepted_mean_luma_range": [luma["p01"], luma["p99"]],
            "accepted_bright_luma_fraction_max": camera["bright_luma_fraction"]["p99"],
            "accepted_valid_depth_fraction_range": [
                camera["valid_depth_fraction"]["p10"],
                camera["valid_depth_fraction"]["p90"],
            ],
        }
        bounds = cameras[side]
        if not all(
            np.isfinite(v)
            for pair in [
                bounds["rgb_gain_range"],
                bounds["accepted_mean_luma_range"],
                bounds["accepted_valid_depth_fraction_range"],
            ]
            for v in pair
        ):
            raise ValueError("Training proposal bounds must be finite")
        low, high = bounds["accepted_valid_depth_fraction_range"]
        if (
            not 0 < low <= high <= 1
            or not 0 <= bounds["accepted_bright_luma_fraction_max"] <= 1
        ):
            raise ValueError("Invalid training coverage/clipping bounds")
    return {
        "schema_version": 1,
        "status": "controlled_pilot_proposal",
        "source_split": "train",
        "source_sequence": profile["sequence"],
        "manifest_sha256": profile["manifest_sha256"],
        "calibration_sha256": profile["calibration_sha256"],
        "frames_supporting_proposal": profile["frames_profiled"],
        "cameras": cameras,
        "methods": {
            "rgb_gain": "controlled photometric stress; not calibrated exposure",
            "depth_mask_transfer": "training donor missing-depth patterns; not a physical failure simulator",
        },
        "disabled_methods": {
            "gaussian_depth_noise": "requires sensor-noise calibration",
            "blur": "sharpness proxy does not identify a physical blur kernel",
            "lidar_thinning": "point-count variation alone does not identify sensor dropout",
            "geometric_warp": "image convention and joint label/intrinsics handling require resolution",
        },
        "preserve_native_resolution": True,
        "labels_unchanged": True,
        "physical_sensor_realism_certified": False,
    }


def validate_clean_reference(rgb, depth, semantic, expected_shape):
    """Reject malformed native reference arrays before generating any variants."""
    if rgb.shape != (*expected_shape, 3) or rgb.dtype != np.uint8:
        raise ValueError("Native RGB shape/dtype mismatch")
    if depth.shape != expected_shape or depth.dtype != np.uint16:
        raise ValueError("Native measured depth shape/dtype mismatch")
    if (
        semantic.shape != expected_shape
        or semantic.dtype != np.uint8
        or int(semantic.max()) >= 16
    ):
        raise ValueError("Native trajectory semantic class contract mismatch")


def apply_rgb_gain(rgb, gain):
    """Return a same-shape uint8 copy; use a documented, nongeometric transform."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected uint8 RGB")
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Gain must be finite and positive")
    return np.rint(np.clip(rgb.astype(np.float32) * gain, 0, 255)).astype(np.uint8)


def transfer_depth_missingness(depth, donor):
    """Mask measurements only; never copy donor depths or change retained values."""
    depth = np.asarray(depth)
    donor = np.asarray(donor)
    if (
        depth.dtype != np.uint16
        or donor.dtype != np.uint16
        or depth.ndim != 2
        or donor.shape != depth.shape
    ):
        raise ValueError("Require aligned-shape uint16 source and donor depth")
    result = depth.copy()
    result[donor == 0] = 0
    return result


def validate_pair(
    clean_rgb,
    clean_depth,
    clean_semantic,
    input_rgb,
    input_depth,
    semantic_target,
    bounds,
):
    """Return acceptance reasons; never relabel rejected variants as successes."""
    reasons = []
    if input_rgb.shape != clean_rgb.shape or input_rgb.dtype != np.uint8:
        reasons.append("RGB format changed")
    if input_depth.shape != clean_depth.shape or input_depth.dtype != np.uint16:
        reasons.append("Depth format changed")
    if (
        semantic_target.shape != clean_semantic.shape
        or semantic_target.dtype != clean_semantic.dtype
        or not np.array_equal(clean_semantic, semantic_target)
    ):
        reasons.append("Semantic target modified")
    if input_depth.shape == clean_depth.shape:
        observed = input_depth > 0
        if np.any(input_depth[clean_depth == 0] != 0):
            reasons.append("Originally invalid depths invented")
        if not np.array_equal(input_depth[observed], clean_depth[observed]):
            reasons.append("Retained depth measurements modified")
        fraction = float(observed.mean())
        low, high = bounds["accepted_valid_depth_fraction_range"]
        if not low <= fraction <= high:
            reasons.append("Input depth coverage outside proposed training envelope")
    if input_rgb.shape == clean_rgb.shape and input_rgb.dtype == np.uint8:
        gray = np.array(Image.fromarray(input_rgb).convert("L"))
        low, high = bounds["accepted_mean_luma_range"]
        if not low <= gray.mean() <= high:
            reasons.append("Mean image luma outside proposed training envelope")
        if (gray >= 250).mean() > bounds["accepted_bright_luma_fraction_max"]:
            reasons.append("Bright-luma fraction exceeds training envelope")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Raw TorWIC root, used only for identity checks",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile_bytes = args.profile.read_bytes()
    profile = json.loads(profile_bytes)
    policy = build_policy(profile)
    if (
        policy["manifest_sha256"]
        != hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    ):
        parser.error("Training profile does not match this manifest")
    if (
        policy["calibration_sha256"]
        != hashlib.sha256((args.root / "calibrations.txt").read_bytes()).hexdigest()
    ):
        parser.error("Training profile does not match current calibration")
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("Policy output must be outside raw data")
    if args.output.resolve() in [args.profile.resolve(), args.manifest.resolve()]:
        parser.error("Policy must not overwrite its evidence inputs")
    policy["profile_sha256"] = hashlib.sha256(profile_bytes).hexdigest()
    policy["policy_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    main()
