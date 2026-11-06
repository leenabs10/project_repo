"""Higher-resolution measured-depth restoration with provenance and scenario metrics."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.paths import workspace_root
from src.training.runtime import PROJECT_ROOT, require_output, require_cached_ids
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from src.training.depth_context import ContextModel

# Noise levels are hypothetical stress tests, not calibrated sensor simulations.
SCENARIOS = ["random", "blocks", "empirical", "noise10", "noise20"]


class QualityModel(ContextModel):
    """Depth-only context model. Inputs retain the shared eight-channel layout."""

    def __init__(self):
        super().__init__(False)

    def forward(self, x):
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
        return x[:, 5:6] + self.head(d)


def corrupt(rgb, depth, donors, i, seed, scenario):
    """Create a reproducible stress input without changing the clean depth target.

    Depth is in metres; zero and values outside (0, 20] are not valid targets.
    Donor masks must come from the training split. Return normalized channels,
    metric and hidden masks, observation distance in pixels, and provenance."""
    rng = np.random.default_rng(seed + i)
    valid = (depth > 0) & (depth <= 20)
    donor_index = int(rng.integers(0, len(donors)))
    if donor_index == i % len(donors):
        donor_index = (donor_index + 1) % len(donors)
    if scenario == "random":
        removed = rng.random(depth.shape) < 0.6
    elif scenario == "blocks":
        removed = rng.random(depth.shape) < 0.2
        for _ in range(4):
            h = int(rng.integers(24, 57))
            w = int(rng.integers(40, 97))
            y = int(rng.integers(0, depth.shape[0] - h + 1))
            x = int(rng.integers(0, depth.shape[1] - w + 1))
            removed[y : y + h, x : x + w] = True
    else:
        attempts = 1
        removed = donors[donor_index] == 0
        while (
            (valid & ~removed).sum() < 100 or (valid & removed).sum() < 100
        ) and attempts < 16:
            donor_index = int(rng.integers(0, len(donors)))
            if donor_index == i % len(donors):
                donor_index = (donor_index + 1) % len(donors)
            removed = donors[donor_index] == 0
            attempts += 1
    observed = valid & ~removed
    hidden = valid & removed
    if observed.sum() < 100 or hidden.sum() < 100:
        raise ValueError(
            "Insufficient observed/hidden depth for quality-controlled augmentation"
        )
    sigma = {"noise10": 0.01, "noise20": 0.02}.get(scenario, 0)
    noisy = np.where(observed, depth, 0).astype(np.float32)
    if sigma:
        noisy[observed] = np.clip(
            noisy[observed] + rng.normal(0, sigma, int(observed.sum())), 0.001, 20
        )
        noisy = np.rint(noisy * 1000) / 1000
    distance, indices = distance_transform_edt(~observed, return_indices=True)
    nearest = noisy[tuple(indices)]
    denominator = gaussian_filter(observed.astype(np.float32), 8)
    smooth = np.where(
        denominator > 1e-6,
        gaussian_filter(noisy, 8) / np.maximum(denominator, 1e-6),
        nearest,
    )
    # Layout: RGB, sparse depth, observation mask, nearest fill, smooth fill, distance.
    x = np.concatenate(
        [
            rgb.astype(np.float32).transpose(2, 0, 1) / 255,
            noisy[None] / 5,
            observed[None].astype(np.float32),
            nearest[None] / 5,
            smooth[None] / 5,
            np.minimum(distance[None] / 40, 1),
        ],
        0,
    ).astype(np.float32)
    # Noise tasks score all valid targets; dropout tasks score only withheld targets.
    metric = valid if sigma else hidden
    metadata = {
        "scenario": scenario,
        "seed": seed + i,
        "donor_train_index": (
            donor_index if scenario not in ["blocks", "random"] else None
        ),
        "donor_selection_attempts": (
            attempts if scenario not in ["blocks", "random"] else 0
        ),
        "noise_std_metres": sigma,
        "observed_fraction_of_valid": float(observed.sum() / valid.sum()),
        "metric_domain": (
            "all originally valid pixels" if sigma else "hidden originally valid pixels"
        ),
        "realism_status": (
            "uncalibrated noise stress test"
            if sigma
            else (
                "empirical train-mask transfer; not physical simulation"
                if scenario == "empirical"
                else "controlled stress test"
            )
        ),
    }
    return x, metric, hidden, distance, metadata


class Samples(Dataset):
    """Load lossless cached frames and select deterministic per-epoch corruptions."""

    def __init__(self, path, donors, scenario="mixed", seed=99):
        with np.load(path, allow_pickle=False) as d:
            self.rgb = d["rgb"]
            self.depth = d["depth_mm"].astype(np.float32) * 0.001
            self.ids = d["sample_ids"].tolist()
        self.donors = donors
        self.scenario = scenario
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        scenario = (
            SCENARIOS[(i + self.epoch) % len(SCENARIOS)]
            if self.scenario == "mixed"
            else self.scenario
        )
        x, mask, hidden, distance, meta = corrupt(
            self.rgb[i],
            self.depth[i],
            self.donors,
            i,
            self.seed + self.epoch * 100000,
            scenario,
        )
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.depth[i][None] / 5),
            torch.from_numpy(mask[None]),
            torch.from_numpy(hidden[None]),
            torch.from_numpy(distance[None].astype(np.float32)),
            bool(meta["noise_std_metres"]),
        )


def predictions(model, x, noisy, distance):
    """Apply the clean/noisy policy to predictions normalized by five metres.

    Clean mode keeps nearest observations within six pixels. Research noisy
    mode can change observations; the exported CLI separately enforces its gate."""
    raw = model(x).clamp(0, 4)
    # Clean observations are preserved. Noise scenarios permit denoising them.
    clean = torch.where(distance <= 6, x[:, 5:6], raw)
    return torch.where(noisy[:, None, None, None], raw, clean)


@torch.no_grad()
def evaluate(model, loader):
    """Accumulate pixel-weighted MAE/RMSE in metres for each metric region."""
    model.eval()
    sums = {}
    elapsed = time.time()
    for x, y, mask, hidden, distance, noisy in loader:
        x = x.cuda()
        y = y.cuda()
        mask = mask.cuda()
        hidden = hidden.cuda()
        distance = distance.cuda()
        noisy = noisy.cuda()
        pred = predictions(model, x, noisy, distance)
        for name, p in [("model", pred), ("nearest", x[:, 5:6]), ("smooth", x[:, 6:7])]:
            for region, m in [
                ("primary", mask),
                ("hidden", hidden),
                ("interior", hidden & (distance >= 16)),
            ]:
                e = ((p - y) * 5)[m]
                t = sums.setdefault(name + "_" + region, [0.0, 0.0, 0])
                t[0] += e.abs().sum().item()
                t[1] += (e * e).sum().item()
                t[2] += e.numel()
    return {
        k: {
            "mae_metres": v[0] / v[2] if v[2] else None,
            "rmse_metres": (v[1] / v[2]) ** 0.5 if v[2] else None,
            "pixels": v[2],
        }
        for k, v in sums.items()
    }


def checks(donors):
    """Verify shapes, finite inputs, and absence of hidden-target leakage."""
    rgb = np.zeros((180, 320, 3), dtype=np.uint8)
    d = np.ones((180, 320), dtype=np.float32)
    x, mask, hidden, _, m = corrupt(rgb, d, donors, 3, 99, "empirical")
    changed = d.copy()
    changed[hidden] = 10
    other = corrupt(rgb, changed, donors, 3, 99, "empirical")[0]
    assert np.array_equal(x, other), "Hidden targets leaked into input"
    assert np.allclose(x[:3], 0)
    assert x.shape == (8, 180, 320) and np.isfinite(x).all()


def main():
    """Parse command-line options and run this module's documented workflow."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--warm-start-root",
        type=Path,
        default=workspace_root(PROJECT_ROOT) / "models/depth_context",
    )
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--resume-completed-training", action="store_true")
    a = p.parse_args()
    if a.epochs < 1:
        raise ValueError("Training requires a positive epoch count")
    a.output = require_output(
        a.output,
        sources=[a.data, a.warm_start_root],
        allow_existing=a.resume_completed_training,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("This research training configuration requires CUDA")
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    with np.load(a.data / "train.npz", allow_pickle=False) as d:
        donors = d["depth_mm"]
    checks(donors)
    train = Samples(a.data / "train.npz", donors)
    val = Samples(a.data / "validation.npz", donors, scenario="empirical", seed=4242)
    # Keep whole trajectories separate; never use October for tuning.
    assert not set(train.ids) & set(val.ids)
    assert all(i.startswith("Aisle_CCW_Run_1/") for i in train.ids) and all(
        i.startswith("Aisle_CW_Run_1/") for i in val.ids
    )
    require_cached_ids(train.ids, "train")
    require_cached_ids(val.ids, "validation")
    # Use a fixed selection subset for efficiency; final evaluation still uses every frame.
    subset = Subset(val, list(range(0, len(val), 4)))
    cfg = {
        "resolution": [320, 180],
        "native_resolution": [1280, 720],
        "epochs": a.epochs,
        "seeds": [42, 43, 44],
        "train_frames": len(train),
        "validation_frames": len(val),
        "selection_frames": len(subset),
        "selection": "fixed every-fourth validation frame, empirical-mask task MAE; full suite evaluated after selection",
        "scenarios": SCENARIOS,
        "donor_source": "training measured zero-depth masks only",
        "test_sensor_files_opened": 0,
        "noise_levels_metres": [0.01, 0.02],
        "noise_calibrated": False,
        "RGB_generation": "none; unchanged real RGB source",
        "derivatives": "BOX-downsampled RGB; nearest-downsampled depth; lossless uint8/uint16 caches",
        "warm_start": "matching prior depth-only seed checkpoint, optimizer reset",
        "loss": "weighted L1; large-hole pixels upweighted up to 3; noise tasks include observed pixels",
        "clean_observation_preservation": True,
        "input_leakage_check": "passed",
        "donor_rejection_rule": "Require at least 100 observed and 100 hidden valid pixels; deterministically redraw donor up to 16 attempts",
        "limitations": [
            "Single training and development-validation trajectory; repeated validation model selection.",
            "Noise is hypothetical and empirical mask transfer is not physical sensor simulation.",
            "Sensor supervision at valid original pixels only; unknown-depth completion not certified.",
            "Training derivatives are below native sensor resolution.",
        ],
    }
    (a.output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    results = []
    for seed in cfg["seeds"]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = QualityModel().cuda()
        checkpoint = torch.load(
            a.warm_start_root / f"depth_seed{seed}" / "best_model.pt",
            map_location="cuda",
            weights_only=True,
        )
        model.load_state_dict(checkpoint["state_dict"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.0002, weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, a.epochs, eta_min=0.00003
        )
        loader = DataLoader(
            train,
            batch_size=16,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        best = float("inf")
        history = []
        started = time.time()
        out = a.output / f"seed{seed}"
        out.mkdir(exist_ok=True)
        completed = False
        if (
            a.resume_completed_training
            and (out / "history.json").is_file()
            and (out / "best_model.pt").is_file()
        ):
            old = json.loads((out / "history.json").read_text())
            if len(old) == a.epochs:
                history = old
                completed = True
                print(
                    seed, "resuming full assessment from completed training", flush=True
                )
        for epoch in ([] if completed else range(a.epochs)):
            train.epoch = epoch
            model.train()
            total = 0
            count = 0
            for x, y, mask, hidden, distance, noisy in loader:
                x = x.cuda()
                y = y.cuda()
                mask = mask.cuda()
                distance = distance.cuda()
                noisy = noisy.cuda()
                optimizer.zero_grad(set_to_none=True)
                # No output clipping while optimizing, to retain gradients.
                raw = model(x)
                clean = torch.where(distance <= 6, x[:, 5:6], raw)
                pred = torch.where(noisy[:, None, None, None], raw, clean)
                error = (pred - y).abs()
                weight = 1 + torch.clamp(distance / 16, max=2)
                loss = (error[mask] * weight[mask]).sum() / weight[mask].sum()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                total += error[mask].detach().sum().item() * 5
                count += int(mask.sum().item())
            scheduler.step()
            metric = evaluate(model, DataLoader(subset, batch_size=16))
            score = metric["model_primary"]["mae_metres"]
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_mae_metres": total / count,
                    "selection_mae_metres": score,
                }
            )
            print(
                seed,
                epoch + 1,
                "train",
                round(total / count, 5),
                "selection",
                round(score, 5),
                flush=True,
            )
            if score < best:
                best = score
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "seed": seed,
                        "epoch": epoch + 1,
                        "config": cfg,
                    },
                    out / "best_model.pt",
                )
            (out / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
        checkpoint = torch.load(
            out / "best_model.pt", map_location="cuda", weights_only=True
        )
        model.load_state_dict(checkpoint["state_dict"])
        metrics = {}
        for scenario in SCENARIOS:
            val.scenario = scenario
            metrics[scenario] = evaluate(model, DataLoader(val, batch_size=16))
            print(
                seed,
                scenario,
                "MAE",
                metrics[scenario]["model_primary"]["mae_metres"],
                flush=True,
            )
        val.scenario = "empirical"
        result = {
            "seed": seed,
            "best_epoch": checkpoint["epoch"],
            "scenarios": metrics,
            "runtime_seconds": time.time() - started,
            "checkpoint_sha256": hashlib.sha256(
                (out / "best_model.pt").read_bytes()
            ).hexdigest(),
        }
        results.append(result)
        (out / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        (a.output / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    summary = {
        s: {
            "model_mae_metres_mean": float(
                np.mean(
                    [r["scenarios"][s]["model_primary"]["mae_metres"] for r in results]
                )
            ),
            "model_mae_metres_sample_std": float(
                np.std(
                    [r["scenarios"][s]["model_primary"]["mae_metres"] for r in results],
                    ddof=1,
                )
            ),
            "nearest_mae_metres": results[0]["scenarios"][s]["nearest_primary"][
                "mae_metres"
            ],
            "relative_improvement_percent": float(
                np.mean(
                    [
                        100
                        * (
                            1
                            - r["scenarios"][s]["model_primary"]["mae_metres"]
                            / r["scenarios"][s]["nearest_primary"]["mae_metres"]
                        )
                        for r in results
                    ]
                )
            ),
        }
        for s in SCENARIOS
    }
    (a.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
