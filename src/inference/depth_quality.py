"""320x180 research inference. Unknown original depths remain unvalidated."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.runtime import PROJECT_ROOT, require_output, require_cached_ids
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from src.training.depth_quality import QualityModel, predictions


@torch.no_grad()
def restore(model, depth, noisy=False):
    """Restore a depth array in metres and return its observed validity mask.

    Clean mode preserves measured pixels. Originally invalid pixels may be
    inferred, but their values have not been quantitatively validated.
    The CLI must check checkpoint permission before enabling noisy mode."""
    observed = np.isfinite(depth) & (depth > 0) & (depth <= 20)
    if not observed.any():
        raise ValueError("No usable depth observation")
    sparse = np.where(observed, depth, 0).astype(np.float32)
    distance, indices = distance_transform_edt(~observed, return_indices=True)
    nearest = sparse[tuple(indices)]
    den = gaussian_filter(observed.astype(np.float32), 8)
    smooth = np.where(
        den > 1e-6, gaussian_filter(sparse, 8) / np.maximum(den, 1e-6), nearest
    )
    x = np.concatenate(
        [
            np.zeros((3, *depth.shape), np.float32),
            sparse[None] / 5,
            observed[None].astype(np.float32),
            nearest[None] / 5,
            smooth[None] / 5,
            np.minimum(distance[None] / 40, 1),
        ],
        axis=0,
    ).astype(np.float32)
    device = next(model.parameters()).device
    result = (
        predictions(
            model,
            torch.from_numpy(x)[None].to(device),
            torch.tensor([noisy], device=device),
            torch.from_numpy(distance.astype(np.float32))[None, None].to(device),
        )[0, 0]
        .cpu()
        .numpy()
        * 5
    )
    if not noisy:
        assert np.allclose(result[observed], depth[observed], atol=1e-5)
    return result, observed


def main():
    """Parse command-line options and run this module's documented workflow."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--depth", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--denoise",
        action="store_true",
        help="Uncalibrated denoising mode; may change observed measurements",
    )
    a = p.parse_args()
    if a.output.suffix.lower() != ".npz":
        raise ValueError("Prediction output must explicitly end in .npz")
    a.output = require_output(a.output, sources=[a.checkpoint, a.depth])
    device = (
        "cuda"
        if a.device == "auto" and torch.cuda.is_available()
        else ("cpu" if a.device == "auto" else a.device)
    )
    torch.set_num_threads(4)
    checkpoint = torch.load(a.checkpoint, map_location=device, weights_only=True)
    # Fail closed: an exported checkpoint must explicitly pass the observed-pixel quality gate.
    if a.denoise and not checkpoint.get("denoising_allowed", False):
        raise ValueError(
            "Denoising is blocked: observed-pixel accuracy did not pass the quality gate. Use clean observation-preserving mode."
        )
    model = QualityModel().to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    with Image.open(a.depth) as image:
        if np.array(image).dtype != np.uint16:
            raise ValueError("Depth must be uint16 millimetres")
        depth = (
            np.array(image.resize((320, 180), Image.Resampling.NEAREST)).astype(
                np.float32
            )
            * 0.001
        )
    output, observed = restore(model, depth, a.denoise)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        a.output,
        depth_metres=output,
        original_observed_mask=observed,
        unvalidated_inferred_mask=~observed,
    )
    print(
        "Saved 320x180 research depth output; inferred originally invalid pixels are unvalidated."
    )


if __name__ == "__main__":
    main()
