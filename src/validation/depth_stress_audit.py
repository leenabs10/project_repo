"""Lossless paired examples with unchanged source RGB and explicit realism limits."""

import hashlib

import json

import os

from pathlib import Path

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.runtime import PROJECT_ROOT, require_output, require_cached_ids

import shutil

import numpy as np

from PIL import Image

from src.training.depth_quality import corrupt, SCENARIOS


def main():
    """Run the explicitly requested research audit; importing performs no I/O."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output = require_output(a.output, sources=[a.root, a.data, a.manifest])

    root = a.output

    root.mkdir(parents=True, exist_ok=True)

    raw = a.root.resolve()

    rows = {
        r["sample_id"]: r
        for r in [json.loads(s) for s in a.manifest.read_text().splitlines()]
    }

    with np.load(a.data / "train.npz", allow_pickle=False) as data:
        donors = data["depth_mm"]
        donor_ids = require_cached_ids(data["sample_ids"].tolist(), "train")

    records = []

    gallery = []

    for split in ["train", "validation"]:
        with np.load(a.data / f"{split}.npz", allow_pickle=False) as data:
            rgb = data["rgb"]
            depth = data["depth_mm"]
            ids = require_cached_ids(data["sample_ids"].tolist(), split)
        for i in [0, len(ids) // 2, len(ids) - 1]:
            sid = ids[i]
            directory = root / sid.replace("/", "_")
            directory.mkdir(exist_ok=True)
            original = (raw / rows[sid]["rgb_path"]).resolve()
            if not original.is_relative_to(raw) or Path(rows[sid]["rgb_path"]).parts[
                :2
            ] != (rows[sid]["sequence_id"], rows[sid]["sequence_id"]):
                raise ValueError("Audit source crosses its declared trajectory")
            payload = original.read_bytes()
            (directory / "rgb_native_original.png").write_bytes(payload)
            assert (
                hashlib.sha256(
                    (directory / "rgb_native_original.png").read_bytes()
                ).digest()
                == hashlib.sha256(payload).digest()
            )
            Image.fromarray(rgb[i]).save(directory / "rgb_model_320.png")
            Image.fromarray(depth[i]).save(directory / "depth_reference_mm.png")
            assert np.array_equal(
                np.array(Image.open(directory / "rgb_model_320.png")), rgb[i]
            )
            assert np.array_equal(
                np.array(Image.open(directory / "depth_reference_mm.png")), depth[i]
            )
            for scenario in SCENARIOS:
                x, mask, hidden, dist, metadata = corrupt(
                    rgb[i],
                    depth[i].astype(np.float32) * 0.001,
                    donors,
                    i,
                    4242,
                    scenario,
                )
                observed = x[4] > 0
                synthetic = np.rint(x[3] * 5000).astype(np.uint16)
                assert np.all(synthetic[~observed] == 0)
                assert np.all(synthetic[observed] > 0)
                assert not np.any(observed & (depth[i] == 0))
                if not metadata["noise_std_metres"]:
                    assert np.array_equal(synthetic[observed], depth[i][observed])
                path = directory / f"{scenario}_input_mm.png"
                Image.fromarray(synthetic).save(path)
                assert np.array_equal(np.array(Image.open(path)), synthetic)
                Image.fromarray((hidden * 255).astype(np.uint8)).save(
                    directory / f"{scenario}_hidden_target_mask.png"
                )
                if metadata["donor_train_index"] is not None:
                    metadata["donor_sample_id"] = donor_ids[
                        metadata["donor_train_index"]
                    ]
                    assert metadata["donor_sample_id"].startswith("Aisle_CCW_Run_1/")
                metadata.update(
                    sample_id=sid,
                    split=split,
                    output=str(path),
                    RGB_native_SHA256=hashlib.sha256(payload).hexdigest(),
                    reference_sha256=hashlib.sha256(
                        (directory / "depth_reference_mm.png").read_bytes()
                    ).hexdigest(),
                    input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    round_trip="passed",
                    RGB_modified=False,
                    input_depth_units="millimetres",
                    model_resolution=[320, 180],
                    native_RGB_resolution=[1280, 720],
                    labels="clean measured sensor depth; unknown pixels excluded",
                    noise_calibrated=False,
                )
                (directory / f"{scenario}_metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
                records.append(metadata)
                if split == "validation" and i == len(ids) // 2:
                    gallery.append((scenario, synthetic.copy(), metadata))

    summary = {
        "paired_examples": len(records),
        "native_RGB_bit_exact_copy": "passed",
        "PNG_round_trip": "passed",
        "clean_observed_depth_preserved": "passed",
        "original_invalid_depth_excluded": "passed",
        "donor_split_check": "passed",
        "noise_calibrated": False,
        "natural_RGB": "real original photographs preserved byte-for-byte",
        "sensor_realism": "not certified; mask transfer and hypothetical noise do not establish physical realism",
        "test_sensor_files_opened": 0,
        "records": records,
    }

    (root / "quality_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    middle_id = ids[len(ids) // 2]

    with Image.open(raw / rows[middle_id]["rgb_path"]) as im:
        axes.flat[0].imshow(im)

    axes.flat[0].set_title("Original real RGB (unaltered)")

    axes.flat[0].axis("off")

    for ax, (scenario, synthetic, metadata) in zip(list(axes.flat)[1:], gallery):
        panel = np.ma.masked_where(synthetic == 0, synthetic.astype(np.float32) * 0.001)
        im = ax.imshow(panel, cmap="viridis", vmin=0, vmax=8)
        ax.set_title(scenario + " depth input")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.045, label="metres")

    fig.suptitle(
        "Paired depth stress tests; real RGB preserved; sensor realism not certified"
    )

    fig.tight_layout()

    fig.savefig(root / "quality_gallery.png", dpi=150)

    plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
