"""Export a bounded training-only augmentation pilot with untouched references."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "profiling"))
from dataset_profile import select_training, safe_path
from augmentation_policy import (
    apply_rgb_gain,
    transfer_depth_missingness,
    validate_pair,
    validate_clean_reference,
)
from torwic_loader import calibration


def read_array(path):
    """Read once for both byte-identity provenance and decoded quality checks."""
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        array = np.array(image)
    return payload, array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7331)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(root) or output == args.policy.resolve().parent:
        parser.error("Pilot output must be outside raw data and policy directory")
    if args.samples < 1:
        parser.error("Positive sample count required")
    if output.exists():
        parser.error(
            "Pilot output already exists; preserve it and choose a new run directory"
        )
    payload = args.manifest.read_bytes()
    policy_bytes = args.policy.read_bytes()
    policy = json.loads(policy_bytes)
    if policy["manifest_sha256"] != hashlib.sha256(payload).hexdigest():
        parser.error("Policy manifest identity mismatch")
    if (
        policy["calibration_sha256"]
        != hashlib.sha256((root / "calibrations.txt").read_bytes()).hexdigest()
    ):
        parser.error("Policy calibration identity mismatch")
    if (
        policy["status"] != "controlled_pilot_proposal"
        or policy["source_split"] != "train"
    ):
        parser.error("Training-only controlled proposal required")
    cal = calibration(root)
    implementation_hashes = {
        "pilot": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "transforms": hashlib.sha256(Path(__file__).with_name("augmentation_policy.py").read_bytes()).hexdigest(),
    }
    records = select_training(
        [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    )
    indices = np.unique(
        np.linspace(0, len(records) - 1, min(args.samples, len(records)), dtype=int)
    )
    output.mkdir(parents=True)
    audit = []
    for frame_index in indices:
        record = records[frame_index]
        for side in ["left", "right"]:
            keys = (
                ["rgb_path", "depth_path", "semantic_path"]
                if side == "left"
                else ["rgb_right_path", "depth_right_path", "semantic_right_path"]
            )
            originals = [read_array(safe_path(root, record[key])) for key in keys]
            rgb, depth, semantic = [pair[1] for pair in originals]
            camera = "Camera1" if side == "left" else "Camera2"
            expected_shape = (int(cal[camera + ".height"]), int(cal[camera + ".width"]))
            validate_clean_reference(rgb, depth, semantic, expected_shape)
            bounds = policy["cameras"][side]
            for method, parameter in [
                ("rgb_gain_low", bounds["rgb_gain_range"][0]),
                ("rgb_gain_high", bounds["rgb_gain_range"][1]),
                ("depth_mask_transfer", None),
            ]:
                seed = args.seed + int(frame_index) * 2 + int(side == "right")
                rng = np.random.default_rng(seed)
                donor_id = None
                input_rgb = rgb.copy()
                input_depth = depth.copy()
                attempts = 1
                if method.startswith("rgb_gain"):
                    input_rgb = apply_rgb_gain(rgb, parameter)
                    reasons = validate_pair(
                        rgb, depth, semantic, input_rgb, input_depth, semantic, bounds
                    )
                else:
                    for attempts in range(1, 25):
                        donor_index = int(rng.integers(0, len(records)))
                        if donor_index == frame_index:
                            donor_index = (donor_index + 1) % len(records)
                        donor_record = records[donor_index]
                        donor = read_array(safe_path(root, donor_record[keys[1]]))[1]
                        donor_id = donor_record["sample_id"]
                        input_depth = transfer_depth_missingness(depth, donor)
                        reasons = validate_pair(
                            rgb,
                            depth,
                            semantic,
                            input_rgb,
                            input_depth,
                            semantic,
                            bounds,
                        )
                        if not reasons or any("luma" in reason for reason in reasons):
                            break
                metadata = {
                    "sample_id": record["sample_id"],
                    "timestamp_seconds": record["timestamp_seconds"],
                    "pose_xyz_qxyzw": record["pose_xyz_qxyzw"],
                    "source_paths": {key: record[key] for key in keys},
                    "calibration_sha256": policy["calibration_sha256"],
                    "implementation_sha256": implementation_hashes,
                    "side": side,
                    "split": "train",
                    "method": method,
                    "gain": parameter,
                    "seed": seed,
                    "donor_sample_id": donor_id,
                    "donor_selection_attempts": attempts,
                    "accepted": not reasons,
                    "rejection_reasons": reasons,
                    "source_sha256": {
                        key: hashlib.sha256(original[0]).hexdigest()
                        for key, original in zip(keys, originals)
                    },
                    "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
                    "native_resolution_retained": True,
                    "geometric_transform": "none",
                    "target_provenance": "measured depth and provided semantic predictions",
                    "sensor_realism": "controlled augmentation; not calibrated physical simulation",
                }
                if not reasons:
                    destination = output / f"{record['frame_id']}_{side}_{method}"
                    destination.mkdir()
                    for name, original in zip(
                        ["clean_rgb.png", "clean_depth_mm.png", "clean_semantic.png"],
                        originals,
                    ):
                        target = destination / name
                        target.write_bytes(original[0])
                        if target.read_bytes() != original[0]:
                            raise ValueError("Clean reference byte identity failed")
                    for name, array in [
                        ("input_rgb.png", input_rgb),
                        ("input_depth_mm.png", input_depth),
                    ]:
                        target = destination / name
                        Image.fromarray(array).save(target)
                        with Image.open(target) as image:
                            if not np.array_equal(np.array(image), array):
                                raise ValueError(
                                    "Lossless augmented-input export failed"
                                )
                    metadata["export_round_trip"] = "passed"
                    (destination / "metadata.json").write_text(
                        json.dumps(metadata, indent=2), encoding="utf-8"
                    )
                audit.append(metadata)
    (output / "pilot_manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in audit), encoding="utf-8"
    )
    accepted = sum(r["accepted"] for r in audit)
    summary = {
        "status": (
            "integrity_passed_for_accepted_pilot"
            if accepted
            else "no_accepted_variants"
        ),
        "attempted_variants": len(audit),
        "accepted_variants": accepted,
        "rejected_variants": len(audit) - accepted,
        "references_byte_identical": True,
        "accepted_exports_lossless": True,
        "labels_unchanged": True,
        "test_sensor_files_opened": 0,
        "validation_sensor_files_opened": 0,
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "physical_sensor_realism_certified": False,
        "limitations": [
            "Acceptance envelope is a training-distribution filter, not proof of realistic sensor physics.",
            "Rejected variants are logged and never exported as accepted training examples.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return int(accepted == 0)


if __name__ == "__main__":
    raise SystemExit(main())
