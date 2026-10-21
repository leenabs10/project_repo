"""Audit every manifest payload without changing source data.

Validate both-camera RGB, measured depth, semantic predictions and binary XYZ
LiDAR for decoding, native geometry, dtypes and usable measurements. The report
retains input hashes and failures; integrity does not certify label accuracy or
camera-LiDAR registration.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingestion"))
from torwic_loader import CLASSES, MODALITIES, calibration, read_xyz


def check_frame(root, record, cal):
    """Return modality-specific errors for a single manifest record.

    RGB is uint8 HxWx3; depth is uint16 millimetres with zero invalid; semantic
    predictions are uint8 HxW class IDs. XYZ decoding excludes nonfinite/zero
    points. One invalid modality does not hide failures in the other payloads.
    """
    errors = []
    for key in MODALITIES:
        try:
            path = (root / record[key]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Path escapes dataset root")
            if key == "lidar_path":
                xyz, mask = read_xyz(path)
                if not len(xyz):
                    raise ValueError("No finite nonzero LiDAR points")
                continue
            camera = "Camera2" if "right" in key else "Camera1"
            shape = (int(cal[f"{camera}.height"]), int(cal[f"{camera}.width"]))
            with Image.open(path) as image:
                image.load()
                array = np.array(image)
            if key.startswith("rgb"):
                if array.shape != (*shape, 3) or array.dtype != np.uint8:
                    raise ValueError("RGB shape or dtype mismatch")
            elif key.startswith("depth"):
                if array.shape != shape or array.dtype != np.uint16:
                    raise ValueError("Depth shape or dtype mismatch")
                # Zero is missing depth, never a measured surface.
                if not np.any(array > 0):
                    raise ValueError("No valid depth pixels")
            else:
                if array.shape != shape or array.dtype != np.uint8:
                    raise ValueError("Semantic shape or dtype mismatch")
                if int(array.max()) >= len(CLASSES):
                    raise ValueError("Unknown semantic class ID")
        except Exception as error:
            errors.append(
                {
                    "sample_id": record["sample_id"],
                    "modality": key,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return errors


def main():
    """Audit exact manifest bytes with bounded workers and write one report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    payload = args.manifest.read_bytes()
    records = [
        json.loads(line)
        for line in payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not records or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Empty manifest or duplicate sample IDs")
    cal = calibration(root)
    errors = []
    failed_frames = 0
    # Decode one frame per worker rather than retaining the dataset in memory.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, frame_errors in enumerate(
            pool.map(lambda r: check_frame(root, r, cal), records), 1
        ):
            errors.extend(frame_errors)
            failed_frames += bool(frame_errors)
            if count % 250 == 0 or count == len(records):
                print(
                    f"Checked {count}/{len(records)} frames; errors: {len(errors)}",
                    flush=True,
                )
    report = {
        "python_executable": sys.executable,
        "virtual_environment": sys.prefix != sys.base_prefix,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "frames_checked": len(records),
        "frames_passed": len(records) - failed_frames,
        "splits": dict(Counter(r["split"] for r in records)),
        "payloads_checked": len(records) * len(MODALITIES),
        "errors": errors,
        "status": "passed" if not errors else "failed",
        "scope": "Both-camera RGB/depth/semantic decoding, shape/dtype/class IDs, binary LiDAR header/payload and nonempty usable points. No model evaluation or parameter selection.",
        "limitations": [
            "Integrity does not prove sensor registration or semantic annotation accuracy.",
            "No IMU interpolation, LiDAR outlier calibration, map decoding, or extra annotated-set audit.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "frames_checked",
                    "frames_passed",
                    "payloads_checked",
                )
            }
        ),
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
