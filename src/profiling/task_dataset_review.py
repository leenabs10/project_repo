"""Prepare training-only goods examples and freeze metadata-only evaluation splits."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingestion"))
from dataset_profile import safe_path, select_training
from torwic_loader import CONFIRMED_SPLITS, MODALITIES, calibration


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def goods_mask(semantic):
    """Derive a uint8 0/1 goods target from native predicted class ID 7.

    This is a deterministic pseudo-label derivative, not a human annotation or
    an aisle-blockage target. Never interpret binary target 1 as publisher class 1.
    """
    if semantic.dtype != np.uint8 or semantic.ndim != 2 or np.any(semantic > 15):
        raise ValueError("Expected native uint8 semantic IDs 0..15")
    return (semantic == 7).astype(np.uint8)


def evaluation_records(records):
    """Verify trajectory/path/date membership before freezing evaluation metadata."""
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("Duplicate dataset sample IDs")
    groups = {}
    for record in records:
        sequence = record["sequence_id"]
        expected = CONFIRMED_SPLITS.get(sequence)
        if expected is None or (record["split"], record["collection_date"]) != expected:
            raise ValueError("Unknown or inconsistent confirmed sequence split/date")
        if record["sample_id"] != f"{sequence}/{record['frame_id']}":
            raise ValueError("Inconsistent sample/sequence/frame IDs")
        for key in list(MODALITIES) + ["imu_left_path", "imu_right_path"]:
            if Path(record[key]).parts[:2] != (sequence, sequence):
                raise ValueError("Record modality path crosses its assigned trajectory")
        groups.setdefault(record["split"], []).append(
            {
                "sample_id": record["sample_id"],
                "sequence_id": sequence,
                "collection_date": record["collection_date"],
                "timestamp_seconds": record["timestamp_seconds"],
                "rgb_path": record["rgb_path"],
                "semantic_path": record["semantic_path"],
            }
        )
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("Expected complete train, validation and held-out test groups")
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError("Choose a fresh review directory outside immutable raw data")
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = evaluation_records(records)
    training = select_training(records)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if (
        profile.get("status") != "passed"
        or profile.get("split") != "train"
        or profile.get("manifest_sha256") != file_hash(args.manifest)
        or profile.get("calibration_sha256") != file_hash(root / "calibrations.txt")
    ):
        raise ValueError(
            "Training characterization does not match this manifest/calibration"
        )
    cal = calibration(root)
    shape = (int(cal["Camera1.height"]), int(cal["Camera1.width"]))
    chosen = np.linspace(0, len(training) - 1, 6, dtype=int)
    output.mkdir(parents=True)
    samples, previews = [], []
    for index in chosen:
        record = training[int(index)]
        rgb_path = safe_path(root, record["rgb_path"])
        semantic_path = safe_path(root, record["semantic_path"])
        rgb_hash, label_hash = file_hash(rgb_path), file_hash(semantic_path)
        with Image.open(rgb_path) as image:
            rgb = np.array(image)
        with Image.open(semantic_path) as image:
            semantic = np.array(image)
        if rgb.shape != (*shape, 3) or rgb.dtype != np.uint8 or semantic.shape != shape:
            raise ValueError("Unexpected native left-camera payload")
        target = goods_mask(semantic)
        folder = output / record["frame_id"]
        folder.mkdir()
        shutil.copyfile(rgb_path, folder / "rgb.png")
        shutil.copyfile(semantic_path, folder / "semantic_prediction.png")
        Image.fromarray(target).save(folder / "goods_target.png")
        with Image.open(folder / "goods_target.png") as image:
            np.testing.assert_array_equal(np.array(image), target)
        if (
            file_hash(folder / "rgb.png") != rgb_hash
            or file_hash(folder / "semantic_prediction.png") != label_hash
            or file_hash(rgb_path) != rgb_hash
            or file_hash(semantic_path) != label_hash
        ):
            raise ValueError("Source/export identity differs")
        overlay = rgb.copy()
        overlay[target > 0] = (
            rgb[target > 0].astype(float) * 0.55 + np.array([235, 70, 210]) * 0.45
        ).astype(np.uint8)
        Image.fromarray(overlay).save(folder / "goods_overlay.png")
        item = {
            "sample_id": record["sample_id"],
            "split": "train",
            "camera_side": "left",
            "timestamp_seconds": record["timestamp_seconds"],
            "rgb_source_path": record["rgb_path"],
            "semantic_source_path": record["semantic_path"],
            "rgb_sha256": rgb_hash,
            "semantic_sha256": label_hash,
            "target_sha256": file_hash(folder / "goods_target.png"),
            "native_shape": list(shape),
            "goods_pixel_fraction": float(target.mean()),
            "binary_mapping": {"0": "other", "1": "publisher goods/materials class 7"},
            "label_provenance": "provided_model_prediction; deterministic binary derivative",
            "independent_ground_truth": False,
        }
        (folder / "metadata.json").write_text(
            json.dumps(item, indent=2), encoding="utf-8"
        )
        samples.append(item)
        preview = Image.fromarray(overlay).resize((480, 270), Image.Resampling.LANCZOS)
        previews.append((record["sample_id"], preview))
    sheet = Image.new("RGB", (960, 900), "#202830")
    draw = ImageDraw.Draw(sheet)
    for index, (name, preview) in enumerate(previews):
        x, y = (index % 2) * 480, (index // 2) * 300
        draw.text((x + 10, y + 8), name + " | goods pseudo-label", fill="white")
        sheet.paste(preview, (x, y + 30))
    sheet.save(output / "training_goods_examples.png")
    protocol = {
        "task": "left_camera_goods_segmentation",
        "source_semantic_id": 7,
        "binary_target_id": 1,
        "camera": "left",
        "manifest_sha256": file_hash(args.manifest),
        "groups": groups,
        "test_use": "No model evaluation until candidate and protocol are frozen; no synthesis fitting on test",
        "metric_proposal": "Goods IoU and Dice; report pseudo-label agreement separately from human-reviewed accuracy",
        "human_reference_status": "pending; extra 237 annotation pairs remain quarantined",
        "blockage_ground_truth_status": "not established by semantic class masks",
    }
    (output / "evaluation_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    report = {
        "status": "task_examples_and_split_metadata_verified",
        "camera": "left",
        "task": "goods_segmentation",
        "source_semantic_id": 7,
        "native_shape": list(shape),
        "split_counts": {key: len(value) for key, value in groups.items()},
        "training_examples": samples,
        "trajectory_payloads_opened": 12,
        "validation_sensor_files_opened": 0,
        "test_sensor_files_opened": 0,
        "source_copies_byte_identical": True,
        "binary_masks_lossless": True,
        "manifest_sha256": file_hash(args.manifest),
        "profile_sha256": file_hash(args.profile),
        "evaluation_protocol_sha256": file_hash(output / "evaluation_protocol.json"),
        "code_sha256": file_hash(Path(__file__)),
        "independent_human_ground_truth_ready": False,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "split_counts",
                    "validation_sensor_files_opened",
                    "test_sensor_files_opened",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
