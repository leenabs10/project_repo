"""Explicit June-validation diagnostics for the existing baseline checkpoint."""

import sys

import json

from pathlib import Path

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.runtime import PROJECT_ROOT, require_output, require_cached_ids

import torch

import numpy as np

from scipy.ndimage import distance_transform_edt

from src.training.depth_baseline import Frames, Refiner


def main():
    """Run the explicitly requested research audit; importing performs no I/O."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        raise ValueError("A diagnostic frame limit must be positive")
    a.output = require_output(a.output, sources=[a.data, a.checkpoint])
    a.output.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)

    data = Frames(a.data, 4242)
    require_cached_ids(data.ids, "validation")

    model = Refiner().cuda().eval()

    model.load_state_dict(
        torch.load(a.checkpoint, map_location="cuda", weights_only=True)["state_dict"]
    )

    errors = []

    corrections = []

    rgb_effect = []

    distances = []

    with torch.no_grad():
        for i in range(min(len(data), a.limit) if a.limit else len(data)):
            x, y, mask = data[i]
            mask = mask[0].numpy()
            batch = x[None].cuda()
            pred = model(batch)[0, 0].cpu().numpy() * 20
            no_rgb = batch.clone()
            no_rgb[:, :3] = 0
            altered = model(no_rgb)[0, 0].cpu().numpy() * 20
            nearest = x[5].numpy() * 20
            errors.append(np.abs(nearest - y[0].numpy() * 20)[mask])
            corrections.append(np.abs(pred - nearest)[mask])
            rgb_effect.append(np.abs(pred - altered)[mask])
            distances.append(distance_transform_edt(x[4].numpy() == 0)[mask])

    errors = np.concatenate(errors)

    corrections = np.concatenate(corrections)

    rgb_effect = np.concatenate(rgb_effect)

    distances = np.concatenate(distances)

    report = {
        "evaluated_frames": min(len(data), a.limit) if a.limit else len(data),
        "withheld_pixels": len(errors),
        "nearest_exact_within_1mm_fraction": float((errors <= 0.001).mean()),
        "nearest_error_metres_p50_p90_p99": np.quantile(
            errors, [0.5, 0.9, 0.99]
        ).tolist(),
        "nearest_observation_distance_pixels_p50_p90_p99": np.quantile(
            distances, [0.5, 0.9, 0.99]
        ).tolist(),
        "learned_correction_metres_mean_p99": [
            float(corrections.mean()),
            float(np.quantile(corrections, 0.99)),
        ],
        "zero_RGB_prediction_change_metres_mean_p99": [
            float(rgb_effect.mean()),
            float(np.quantile(rgb_effect, 0.99)),
        ],
        "original_model_receptive_field_pixels": 9,
        "test_evaluated": False,
        "interpretation": "RGB zeroing is a sensitivity diagnostic with out-of-distribution inputs, not an RGB usefulness estimate.",
    }

    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
