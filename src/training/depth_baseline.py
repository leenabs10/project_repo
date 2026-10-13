"""Small RGB+sparse-depth completion pilot, supervised by measured depth.

Uses June train/validation cache only. Metrics cover held-out, originally valid
pixels; no target at those pixels enters the model input. No October evaluation.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.runtime import PROJECT_ROOT, require_output, require_cached_ids
import random
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw


class Refiner(nn.Module):
    """Small residual CNN initialized to reproduce nearest-depth filling."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(6, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 1, 3, padding=1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return x[:, 5:6] + self.layers(x)


def inputs(rgb, depth, observed):
    """Build RGB, sparse depth, validity, and nearest-fill input channels.

    Only observed depths are used; depth values are normalized by twenty metres."""
    sparse = np.where(observed, depth, 0).astype(np.float32)
    if not observed.any():
        raise ValueError("No observed depth")
    indices = distance_transform_edt(
        ~observed, return_distances=False, return_indices=True
    )
    nearest = sparse[tuple(indices)]
    x = np.concatenate(
        [
            rgb.astype(np.float32).transpose(2, 0, 1) / 255,
            sparse[None] / 20,
            observed[None].astype(np.float32),
            nearest[None] / 20,
        ],
        axis=0,
    )
    return x


class Frames(Dataset):
    """Load measured-depth references and generate reproducible pixel dropout."""

    def __init__(self, path, seed):
        with np.load(path, allow_pickle=False) as data:
            self.rgb = data["rgb"]
            self.depth = data["depth_mm"].astype(np.float32) * 0.001
            self.ids = data["sample_ids"].tolist()
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        depth = self.depth[i]
        valid = (depth > 0) & (depth <= 20)
        rng = np.random.default_rng(self.seed + self.epoch * 100000 + i)
        observed = valid & (rng.random(depth.shape) >= 0.6)
        withheld = valid & ~observed
        if not withheld.any():
            raise ValueError(f"No withheld pixels: {self.ids[i]}")
        return (
            torch.from_numpy(inputs(self.rgb[i], depth, observed)),
            torch.from_numpy(depth[None] / 20),
            torch.from_numpy(withheld[None]),
        )


def check_no_leakage():
    """Check that hidden target values cannot affect inputs or masked metrics."""
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    d = np.ones((4, 4), dtype=np.float32)
    observed = np.zeros((4, 4), dtype=bool)
    observed[0, 0] = True
    changed = d.copy()
    changed[~observed] = 19
    assert np.array_equal(
        inputs(rgb, d, observed), inputs(rgb, changed, observed)
    ), "Target leakage into model input"
    pred = torch.tensor([1.0, 100.0])
    ref = torch.tensor([2.0, 0.0])
    mask = torch.tensor([True, False])
    assert (pred - ref).abs()[mask].mean().item() == 1, "Invalid pixels affected metric"


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute pixel-weighted hidden-depth MAE and RMSE in metres."""
    model.eval()
    totals = {"model": [0.0, 0.0, 0], "nearest": [0.0, 0.0, 0]}
    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        for name, pred in [("model", model(x).clamp(0, 1)), ("nearest", x[:, 5:6])]:
            err = ((pred - y) * 20)[mask]
            totals[name][0] += err.abs().sum().item()
            totals[name][1] += (err * err).sum().item()
            totals[name][2] += err.numel()
    return {
        k: {
            "mae_metres": v[0] / v[2],
            "rmse_metres": (v[1] / v[2]) ** 0.5,
            "evaluated_pixels": v[2],
        }
        for k, v in totals.items()
    }


def preview(model, data, device, path):
    """Save a fixed validation example with consistent depth colors."""
    x, y, mask = data[len(data) // 2]
    with torch.no_grad():
        pred = model(x[None].to(device)).clamp(0, 1)[0, 0].cpu().numpy() * 20

    def color(d):
        """Map depth to display colors; unknown/nonpositive depth is black."""
        value = np.clip(d / 8, 0, 1)
        result = np.stack(
            [value * 255, (1 - value) * 160, (1 - value) * 255], axis=-1
        ).astype(np.uint8)
        result[d <= 0] = 0
        return result

    rgb = data.rgb[len(data) // 2]
    sparse = x[3].numpy() * 20
    panels = [
        rgb,
        color(y[0].numpy() * 20),
        color(sparse),
        color(x[5].numpy() * 20),
        color(pred),
    ]
    labels = [
        "RGB",
        "Measured reference",
        "Sparse input",
        "Nearest baseline",
        "Learned completion",
    ]
    canvas = Image.new("RGB", (800, 112), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(Image.fromarray(panel), (i * 160, 22))
        draw.text((i * 160 + 2, 4), label, fill="black")
    canvas.save(path)


def main():
    """Parse command-line options and run this module's documented workflow."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    if a.epochs < 1:
        raise ValueError("Training requires a positive epoch count")
    a.output = require_output(a.output, sources=[a.data])
    if not torch.cuda.is_available():
        raise RuntimeError("This research training configuration requires CUDA")
    a.output.mkdir(parents=True, exist_ok=True)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this configured run")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    check_no_leakage()
    train = Frames(a.data / "train.npz", 42)
    val = Frames(a.data / "validation.npz", 4242)
    if set(train.ids) & set(val.ids):
        raise ValueError("Train/validation overlap")
    if any(not i.startswith("Aisle_CCW_Run_1/") for i in train.ids) or any(
        not i.startswith("Aisle_CW_Run_1/") for i in val.ids
    ):
        raise ValueError("Unexpected sequence in training inputs")
    require_cached_ids(train.ids, "train")
    require_cached_ids(val.ids, "validation")
    g = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(
        train, batch_size=a.batch_size, shuffle=True, num_workers=0, generator=g
    )
    valloader = DataLoader(val, batch_size=a.batch_size, num_workers=0)
    device = torch.device("cuda")
    model = Refiner().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best = float("inf")
    started = time.time()
    baseline = evaluate(model, valloader, device)
    print(f'Nearest baseline validation: {baseline["nearest"]}', flush=True)
    config = {
        "task": "RGB + sparse-depth completion",
        "supervision": "measured sensor depth, not independent ground truth",
        "epochs": a.epochs,
        "batch_size": a.batch_size,
        "seed": seed,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "dropout_fraction": 0.6,
        "validation_dropout_seed": 4242,
        "resolution": [160, 90],
        "train_frames": len(train),
        "validation_frames": len(val),
        "test_sensor_files_opened": 0,
        "architecture": "4-layer residual convolutional refiner over RGB, sparse depth, validity, nearest fill",
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "metric_domain": "withheld originally valid pixels, 0 < depth <= 20m",
        "data_profile": json.loads((a.data / "profile.json").read_text()),
        "input_leakage_check": "passed",
        "split_overlap_check": "passed",
        "limitations": [
            "Low resolution pilot, not full-resolution production training.",
            "Random pixel dropout is a controlled corruption, not a calibrated model of real sensor failure.",
            "No LiDAR fusion, semantics, Isaac Sim or SLAM model is trained.",
            "Adjacent frames within each trajectory are correlated; this is one trajectory per split.",
            "Seeds are fixed; CUDA bitwise determinism across machines is not guaranteed.",
        ],
    }
    (a.output / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    for epoch in range(a.epochs):
        train.epoch = epoch
        model.train()
        total = 0.0
        count = 0
        for x, y, mask in trainloader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            opt.zero_grad(set_to_none=True)
            err = (model(x) - y).abs()[mask]
            loss = err.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            opt.step()
            total += err.detach().sum().item() * 20
            count += err.numel()
        metrics = evaluate(model, valloader, device)
        row = {
            "epoch": epoch + 1,
            "train_mae_metres": total / count,
            "validation_mae_metres": metrics["model"]["mae_metres"],
            "validation_rmse_metres": metrics["model"]["rmse_metres"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["validation_mae_metres"] < best:
            best = row["validation_mae_metres"]
            bestepoch = epoch + 1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config,
                    "epoch": bestepoch,
                },
                a.output / "best_model.pt",
            )
        with (a.output / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
    checkpoint = torch.load(
        a.output / "best_model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    final = evaluate(model, valloader, device)
    report = {
        "status": "completed",
        "best_epoch": bestepoch,
        "validation": final,
        "model_mae_relative_improvement_percent": 100
        * (final["nearest"]["mae_metres"] - final["model"]["mae_metres"])
        / final["nearest"]["mae_metres"],
        "runtime_seconds": time.time() - started,
        "checkpoint_sha256": hashlib.sha256(
            (a.output / "best_model.pt").read_bytes()
        ).hexdigest(),
        "test_evaluated": False,
        "claims": "Validation performance for controlled dropout only; no ground-truth SLAM or test-set result.",
    }
    (a.output / "results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    preview(model, val, device, a.output / "validation_preview.png")
    os.environ.setdefault("MPLCONFIGDIR", str(a.output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [r["epoch"] for r in history],
        [r["train_mae_metres"] for r in history],
        label="Train",
    )
    ax.plot(
        [r["epoch"] for r in history],
        [r["validation_mae_metres"] for r in history],
        label="Validation",
    )
    ax.axhline(
        final["nearest"]["mae_metres"],
        color="gray",
        linestyle="--",
        label="Nearest-fill validation",
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Withheld depth MAE (metres)",
        title="TorWIC depth completion pilot",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.output / "learning_curve.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
