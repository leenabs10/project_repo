"""Decode only June training/validation RGB and measured depth; leave test untouched."""

import argparse
import hashlib
import json
import io
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.runtime import PROJECT_ROOT, require_output, require_cached_ids
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--height", type=int, default=90)
    a = p.parse_args()
    rows = [
        json.loads(x)
        for x in a.manifest.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    a.output = require_output(a.output, sources=[a.root, a.manifest])
    from src.profiling.task_dataset_review import evaluation_records

    evaluation_records(rows)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.width * 720 != a.height * 1280 or a.width > 1280 or a.width <= 0:
        raise ValueError("Resolution must preserve native aspect ratio and not upscale")
    provenance = []
    report = {
        "resolution": [a.width, a.height],
        "depth_units": "metres",
        "depth_valid_range_metres": [0, 20],
        "rgb_resampling": "BOX area averaging",
        "depth_resampling": "nearest; no invented depths",
        "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        "test_sensor_files_opened": 0,
        "splits": {},
        "limitations": [
            "RGB/depth decoded only; not a whole-dataset LiDAR or semantic integrity audit.",
            "Depth nearest-neighbour resampling; native-resolution details lost.",
            "Measured depth is sensor supervision, not independent ground truth.",
        ],
    }
    for split, sequence in [
        ("train", "Aisle_CCW_Run_1"),
        ("validation", "Aisle_CW_Run_1"),
    ]:
        selected = sorted(
            [r for r in rows if r["split"] == split], key=lambda r: int(r["frame_id"])
        )
        if not selected or any(r["sequence_id"] != sequence for r in selected):
            raise ValueError(f"Unexpected {split} sequence assignment")
        require_cached_ids([record["sample_id"] for record in selected], split)
        rgbs, depths, ids, fractions = [], [], [], []
        totals, valid_count, excluded = 0, 0, 0
        for i, r in enumerate(selected):
            paths = [(a.root / r[k]).resolve() for k in ("rgb_path", "depth_path")]
            if any(not x.is_relative_to(a.root.resolve()) for x in paths):
                raise ValueError("Path escapes root")
            payloads = [path.read_bytes() for path in paths]
            provenance.append(
                {
                    "sample_id": r["sample_id"],
                    "split": split,
                    "rgb_path": r["rgb_path"],
                    "depth_path": r["depth_path"],
                    "rgb_sha256": hashlib.sha256(payloads[0]).hexdigest(),
                    "depth_sha256": hashlib.sha256(payloads[1]).hexdigest(),
                }
            )
            with Image.open(io.BytesIO(payloads[0])) as im:
                im.load()
                if im.size != (1280, 720) or im.mode != "RGB":
                    raise ValueError(f"Unexpected RGB: {paths[0]}")
                rgb = np.array(im.resize((a.width, a.height), Image.Resampling.BOX))
            with Image.open(io.BytesIO(payloads[1])) as im:
                im.load()
                raw = np.array(im)
                if raw.shape != (720, 1280) or raw.dtype != np.uint16:
                    raise ValueError(f"Unexpected depth: {paths[1]}")
                valid = (raw > 0) & (raw <= 20000)
                totals += raw.size
                valid_count += int(valid.sum())
                excluded += int((raw > 20000).sum())
                fractions.append(float(valid.mean()))
                depth = np.array(
                    im.resize((a.width, a.height), Image.Resampling.NEAREST)
                ).astype(np.uint16)
            rgbs.append(rgb)
            depths.append(depth)
            ids.append(r["sample_id"])
            if (i + 1) % 100 == 0:
                print(f"{split}: decoded {i+1}/{len(selected)}", flush=True)
        np.savez(
            a.output / f"{split}.npz",
            rgb=np.stack(rgbs),
            depth_mm=np.stack(depths),
            sample_ids=np.array(ids),
        )
        with np.load(a.output / f"{split}.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["rgb"], np.stack(rgbs)) or not np.array_equal(
                saved["depth_mm"], np.stack(depths)
            ):
                raise ValueError("Lossless cache round-trip failed")
        report["splits"][split] = {
            "frames": len(ids),
            "sequence": sequence,
            "native_valid_depth_fraction": valid_count / totals,
            "native_pixels_over_20m": excluded,
            "native_valid_fraction_min_median_max": [
                min(fractions),
                float(np.median(fractions)),
                max(fractions),
            ],
        }
    report["cache_round_trip"] = "passed"
    report["cache_sha256"] = {
        s: hashlib.sha256((a.output / f"{s}.npz").read_bytes()).hexdigest()
        for s in ["train", "validation"]
    }
    (a.output / "source_provenance.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in provenance), encoding="utf-8"
    )
    (a.output / "profile.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
