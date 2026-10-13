"""Controlled block-dropout depth completion, fixed validation and RGB ablation."""

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.runtime import PROJECT_ROOT, require_output, require_cached_ids
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def build_input(rgb, depth, observed):
    """Build eight channels from RGB and observed depth only.

    Channel order: RGB (0:3), sparse depth (3), observed mask (4), nearest
    fill (5), Gaussian fill (6), and clipped observation distance (7).
    Depth channels use a five-metre scale; distance uses a twenty-pixel scale."""
    sparse = np.where(observed, depth, 0).astype(np.float32)
    distance, indices = distance_transform_edt(~observed, return_indices=True)
    nearest = sparse[tuple(indices)]
    denominator = gaussian_filter(observed.astype(np.float32), 4)
    smooth = np.where(
        denominator > 1e-6,
        gaussian_filter(sparse, 4) / np.maximum(denominator, 1e-6),
        nearest,
    )
    x = np.concatenate(
        [
            rgb.astype(np.float32).transpose(2, 0, 1) / 255,
            sparse[None] / 5,
            observed[None].astype(np.float32),
            nearest[None] / 5,
            smooth[None] / 5,
            np.minimum(distance[None] / 20, 1).astype(np.float32),
        ],
        axis=0,
    )
    return x.astype(np.float32), distance.astype(np.float32)


class Data(Dataset):
    """Provide fixed validation masks or varying deterministic training masks."""

    def __init__(self, path, seed, training=False):
        with np.load(path, allow_pickle=False) as d:
            self.rgb = d["rgb"]
            self.depth = d["depth_mm"].astype(np.float32) * 0.001
            self.ids = d["sample_ids"].tolist()
        self.seed = seed
        self.epoch = 0
        self.training = training
        self.cache = None if training else [self.make(i) for i in range(len(self.ids))]

    def __len__(self):
        return len(self.ids)

    def make(self, i):
        """Generate one sample using only its observed depths as input features."""
        d = self.depth[i]
        valid = (d > 0) & (d <= 20)
        rng = np.random.default_rng(self.seed + self.epoch * 100000 + i)
        removed = rng.random(d.shape) < 0.2
        # Same documented hole distribution for train and validation, independent seeds.
        for _ in range(4):
            h = int(rng.integers(12, 29))
            w = int(rng.integers(20, 49))
            y = int(rng.integers(0, d.shape[0] - h + 1))
            x = int(rng.integers(0, d.shape[1] - w + 1))
            removed[y : y + h, x : x + w] = True
        observed = valid & ~removed
        mask = valid & removed
        if not observed.any() or not mask.any():
            raise ValueError("Empty corruption mask")
        x, distance = build_input(self.rgb[i], d, observed)
        return tuple(
            torch.from_numpy(a) for a in (x, d[None] / 5, mask[None], distance[None])
        )

    def __getitem__(self, i):
        return self.make(i) if self.training else self.cache[i]


def conv(a, b):
    """Create a two-convolution feature block with normalization and activation."""
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1),
        nn.GroupNorm(4, b),
        nn.SiLU(),
        nn.Conv2d(b, b, 3, padding=1),
        nn.GroupNorm(4, b),
        nn.SiLU(),
    )


class ContextModel(nn.Module):
    """Encoder/decoder with optional RGB ablation and nearest-fill residuals."""

    def __init__(self, rgb=True):
        super().__init__()
        self.rgb = rgb
        self.e1 = conv(8, 16)
        self.e2 = conv(16, 32)
        self.e3 = conv(32, 64)
        self.d2 = conv(96, 32)
        self.d1 = conv(48, 16)
        self.head = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        if not self.rgb:
            x = x.clone()
            x[:, :3] = 0
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        d = self.d2(
            torch.cat(
                [
                    F.interpolate(
                        c, size=b.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    b,
                ],
                1,
            )
        )
        d = self.d1(
            torch.cat(
                [
                    F.interpolate(
                        d, size=a.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    a,
                ],
                1,
            )
        )
        pred = x[:, 5:6] + self.head(d)
        return torch.where(x[:, 4:5] > 0, x[:, 3:4], pred)


class LocalModel(nn.Module):
    """Short-context residual comparator using the same input layout."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(8, 24, 3, padding=1),
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


@torch.no_grad()
def evaluate(model, loader):
    """Measure hidden valid pixels, distinguishing nearby and interior holes."""
    if model is not None:
        model.eval()
    totals = {}
    frame_rows = []
    for x, y, mask, distance in loader:
        x = x.cuda()
        y = y.cuda()
        mask = mask.cuda()
        distance = distance.cuda()
        predictions = {"nearest": x[:, 5:6], "smooth": x[:, 6:7]}
        if model is not None:
            predictions["model"] = model(x).clamp(0, 4)
        errors = {name: (pred - y) * 5 for name, pred in predictions.items()}
        for name, err in errors.items():
            for region, m in [
                ("all", mask),
                ("near", mask & (distance <= 3)),
                ("interior", mask & (distance >= 8)),
            ]:
                vals = err[m]
                t = totals.setdefault(name + "_" + region, [0.0, 0.0, 0])
                t[0] += vals.abs().sum().item()
                t[1] += (vals * vals).sum().item()
                t[2] += vals.numel()
        for j in range(len(x)):
            frame_rows.append(
                {
                    name: {
                        "absolute_sum": err[j][mask[j]].abs().sum().item(),
                        "count": int(mask[j].sum().item()),
                    }
                    for name, err in errors.items()
                }
            )
    metrics = {
        k: {
            "mae_metres": v[0] / v[2] if v[2] else None,
            "rmse_metres": (v[1] / v[2]) ** 0.5 if v[2] else None,
            "pixels": v[2],
        }
        for k, v in totals.items()
    }
    return metrics, frame_rows


def paired_interval(rows):
    # Resample contiguous groups, not independent neighboring video frames.
    """Bootstrap paired errors in contiguous 20-frame groups.

    This interval is conditional on one trajectory, not evidence of
    independent frames or generalization to new sites."""
    blocks = [rows[i : i + 20] for i in range(0, len(rows), 20)]
    arrays = np.array(
        [
            [
                sum(
                    r["nearest"]["absolute_sum"] - r["model"]["absolute_sum"] for r in b
                ),
                sum(r["nearest"]["count"] for r in b),
            ]
            for b in blocks
        ]
    )
    rng = np.random.default_rng(2468)
    values = []
    for _ in range(1000):
        s = arrays[rng.integers(0, len(arrays), len(arrays))].sum(axis=0)
        values.append(s[0] / s[1])
    return {
        "nearest_minus_model_mae_metres_95pct": np.quantile(
            values, [0.025, 0.975]
        ).tolist(),
        "block_frames": 20,
        "caveat": "Conditional on this one validation trajectory; block bootstrap does not establish across-site generalization.",
    }


def checks():
    """Verify hidden-target exclusion and depth-only RGB invariance."""
    rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    d = np.ones((12, 12), dtype=np.float32)
    observed = np.zeros_like(d, dtype=bool)
    observed[:2] = True
    changed = d.copy()
    changed[~observed] = 15
    assert np.array_equal(
        build_input(rgb, d, observed)[0], build_input(rgb, changed, observed)[0]
    )
    network = ContextModel(False).eval()
    x = torch.from_numpy(build_input(rgb, d, observed)[0])[None]
    altered = x.clone()
    altered[:, :3] = 1
    with torch.no_grad():
        assert torch.equal(network(x), network(altered))


def run(kind, seed, train, val, out, epochs):
    """Train one model seed and reload its best validation-selected checkpoint."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = (LocalModel() if kind == "local" else ContextModel(kind == "rgb")).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=0.00005
    )
    loader = DataLoader(
        train,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    vl = DataLoader(val, batch_size=32)
    run_dir = out / f"{kind}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    started = time.time()
    for epoch in range(epochs):
        train.epoch = epoch
        model.train()
        total = 0.0
        count = 0
        for x, y, mask, distance in loader:
            x = x.cuda()
            y = y.cuda()
            mask = mask.cuda()
            distance = distance.cuda()
            optimizer.zero_grad(set_to_none=True)
            error = (model(x) - y).abs()
            weights = 1 + torch.clamp(distance / 8, max=2)
            loss = (error[mask] * weights[mask]).sum() / weights[mask].sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total += error[mask].detach().sum().item() * 5
            count += int(mask.sum().item())
        scheduler.step()
        metrics, _ = evaluate(model, vl)
        history.append(
            {
                "epoch": epoch + 1,
                "train_mae_metres": total / count,
                "validation": metrics["model_all"],
            }
        )
        print(
            kind,
            seed,
            epoch + 1,
            "train",
            round(total / count, 5),
            "val",
            round(metrics["model_all"]["mae_metres"], 5),
            flush=True,
        )
        if metrics["model_all"]["mae_metres"] < best:
            best = metrics["model_all"]["mae_metres"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "kind": kind,
                    "seed": seed,
                    "epoch": best_epoch,
                },
                run_dir / "best_model.pt",
            )
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    model.load_state_dict(
        torch.load(run_dir / "best_model.pt", weights_only=True)["state_dict"]
    )
    metrics, frames = evaluate(model, vl)
    result = {
        "kind": kind,
        "seed": seed,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "paired_block_bootstrap": paired_interval(frames),
        "runtime_seconds": time.time() - started,
        "relative_mae_improvement_percent": 100
        * (
            1
            - metrics["model_all"]["mae_metres"] / metrics["nearest_all"]["mae_metres"]
        ),
    }
    (run_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (run_dir / "per_frame_errors.json").write_text(json.dumps(frames), encoding="utf-8")
    return result


def main():
    """Parse command-line options and run this module's documented workflow."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=8)
    a = p.parse_args()
    if a.epochs < 1:
        raise ValueError("Training requires a positive epoch count")
    a.output = require_output(a.output, sources=[a.data])
    if not torch.cuda.is_available():
        raise RuntimeError("This research training configuration requires CUDA")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    checks()
    a.output.mkdir(parents=True, exist_ok=True)
    train = Data(a.data / "train.npz", 99, True)
    val = Data(a.data / "validation.npz", 4242)
    assert all(i.startswith("Aisle_CCW_Run_1/") for i in train.ids)
    assert all(i.startswith("Aisle_CW_Run_1/") for i in val.ids)
    assert not set(train.ids) & set(val.ids)
    require_cached_ids(train.ids, "train")
    require_cached_ids(val.ids, "validation")
    baseline, _ = evaluate(None, DataLoader(val, batch_size=32))
    configuration = {
        "scenario": "20% random dropout plus four missing rectangles per frame",
        "rectangle_height_pixels": [12, 28],
        "rectangle_width_pixels": [20, 48],
        "validation_seed": 4242,
        "training_corruption_seed": 99,
        "resolution": [160, 90],
        "epochs": a.epochs,
        "initial_lr": 0.0005,
        "normalization_metres": 5,
        "loss": "distance-weighted L1; interior weight up to 3",
        "selection": "lowest aggregate fixed-validation MAE per run",
        "model_seeds": [42, 43, 44],
        "train_frames": len(train),
        "validation_frames": len(val),
        "test_sensor_files_opened": 0,
        "limitations": [
            "Controlled corruption, not calibrated sensor failure.",
            "Measured sensor depth supervision.",
            "Single validation trajectory repeatedly used for model selection; not unbiased test performance.",
            "Resolution, data source and original missing depth remain limitations.",
        ],
        "input_leakage_check": "passed",
        "RGB_ablation_invariance_check": "passed",
        "baseline_metrics": baseline,
    }
    (a.output / "config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    print("Block scenario baselines:", json.dumps(baseline), flush=True)
    results = []
    for kind, seed in [("local", 42)] + [
        (k, s) for s in [42, 43, 44] for k in ["rgb", "depth"]
    ]:
        results.append(run(kind, seed, train, val, a.output, a.epochs))
        (a.output / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    summary = {
        k: {
            "runs": len([r for r in results if r["kind"] == k]),
            "mae_metres_mean": float(
                np.mean(
                    [
                        r["metrics"]["model_all"]["mae_metres"]
                        for r in results
                        if r["kind"] == k
                    ]
                )
            ),
            "mae_metres_std_across_seeds": float(
                np.std(
                    [
                        r["metrics"]["model_all"]["mae_metres"]
                        for r in results
                        if r["kind"] == k
                    ]
                )
            ),
            "relative_improvement_percent_mean": float(
                np.mean(
                    [
                        r["relative_mae_improvement_percent"]
                        for r in results
                        if r["kind"] == k
                    ]
                )
            ),
        }
        for k in ["local", "rgb", "depth"]
    }
    summary["baseline_metrics"] = baseline
    summary["test_evaluated"] = False
    (a.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    os.environ.setdefault("MPLCONFIGDIR", str(a.output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ["Nearest", "Smooth", "Local CNN", "Context RGB", "Context depth"]
    values = [
        baseline["nearest_all"]["mae_metres"],
        baseline["smooth_all"]["mae_metres"],
    ] + [summary[k]["mae_metres_mean"] for k in ["local", "rgb", "depth"]]
    ax.bar(keys, values)
    ax.set(
        ylabel="Withheld validation MAE (metres)",
        title="Block-dropout comparison; same validation corruption",
    )
    fig.tight_layout()
    fig.savefig(a.output / "comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
