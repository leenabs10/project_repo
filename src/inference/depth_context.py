"""Inference for the saved low-resolution development candidate.

Output at original sensor-invalid pixels has not been quantitatively validated.
"""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.runtime import PROJECT_ROOT, require_output, require_cached_ids
import torch
import numpy as np
from PIL import Image
from src.training.depth_context import ContextModel, build_input


def load_model(path, device="cuda"):
    """Load the saved architecture and weights without unsafe pickle objects."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ContextModel(checkpoint["kind"] == "rgb").to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


@torch.no_grad()
def complete(model, rgb, depth_m, gate=3, device="cuda"):
    """Complete depth in metres, retaining nearest values around observations.

    The distance gate is in pixels at the model resolution. Return completed
    depth plus the mask of originally usable measurements."""
    observed = (depth_m > 0) & (depth_m <= 20) & np.isfinite(depth_m)
    if not observed.any():
        raise ValueError("At least one usable depth observation required")
    x, _ = build_input(rgb, depth_m, observed)
    x = torch.from_numpy(x)[None].to(device)
    pred = torch.where(x[:, 7:8] * 20 <= gate, x[:, 5:6], model(x)).clamp(0, 4)
    return pred[0, 0].cpu().numpy() * 5, observed


def main():
    """Parse command-line options and run this module's documented workflow."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rgb", type=Path, required=True)
    p.add_argument("--depth", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.suffix.lower() != ".npz":
        raise ValueError("Prediction output must explicitly end in .npz")
    a.output = require_output(a.output, sources=[a.checkpoint, a.depth, a.rgb])
    torch.set_num_threads(4)
    with Image.open(a.rgb) as im:
        rgb = np.array(im.convert("RGB").resize((160, 90), Image.Resampling.BILINEAR))
    with Image.open(a.depth) as im:
        if np.asarray(im).dtype != np.uint16:
            raise ValueError("Depth must be uint16 millimetres")
        depth = (
            np.array(im.resize((160, 90), Image.Resampling.NEAREST)).astype(np.float32)
            * 0.001
        )
    device = (
        "cuda"
        if a.device == "auto" and torch.cuda.is_available()
        else ("cpu" if a.device == "auto" else a.device)
    )
    model, checkpoint = load_model(a.checkpoint, device)
    pred, observed = complete(
        model, rgb, depth, checkpoint["distance_gate_pixels"], device
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, depth_metres=pred, original_observed_mask=observed)
    print(
        f"Saved {a.output}: 160x90 depth in metres; unknown sensor depths remain unvalidated."
    )


if __name__ == "__main__":
    main()
