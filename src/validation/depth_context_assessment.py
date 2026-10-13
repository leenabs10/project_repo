"""Cross-corruption validation of saved checkpoints; no test data."""

import json

import os

from pathlib import Path

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.runtime import PROJECT_ROOT, require_output, require_cached_ids

import torch

from torch.utils.data import Dataset, DataLoader

import numpy as np

from src.training.depth_context import (
    Data,
    ContextModel,
    LocalModel,
    build_input,
    evaluate,
)


class RandomData(Data):
    """Recreate the original random-dropout validation task at a fixed seed."""

    def make(self, i):
        """Build the fixed 60-percent-dropout sample without exposing hidden depths."""
        d = self.depth[i]
        valid = (d > 0) & (d <= 20)
        rng = np.random.default_rng(4242 + i)
        observed = valid & (rng.random(d.shape) >= 0.6)
        mask = valid & ~observed
        x, distance = build_input(self.rgb[i], d, observed)
        return tuple(
            torch.from_numpy(a) for a in (x, d[None] / 5, mask[None], distance[None])
        )


class Gated(torch.nn.Module):
    """Keep nearest filling within three pixels and use the model farther away."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Fixed rule from the original diagnosis: nearby observations already
        # reconstruct random-dropout targets well. Learn only larger gaps.
        return torch.where(x[:, 7:8] * 20 <= 3, x[:, 5:6], self.model(x))


def main():
    """Run the explicitly requested research audit; importing performs no I/O."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output = require_output(a.output, sources=[a.data, a.checkpoints])
    a.output.mkdir(parents=True)

    torch.set_num_threads(4)

    root = a.output

    random = RandomData(a.data, 4242)

    loader = DataLoader(random, batch_size=32)

    val = Data(a.data, 4242)
    require_cached_ids(val.ids, "validation")
    require_cached_ids(random.ids, "validation")

    blockloader = DataLoader(val, batch_size=32)

    reports = []

    for run in json.loads((a.checkpoints / "results.json").read_text()):
        kind = run["kind"]
        seed = run["seed"]
        model = (
            (LocalModel() if kind == "local" else ContextModel(kind == "rgb"))
            .cuda()
            .eval()
        )
        checkpoint = torch.load(
            a.checkpoints / f"{kind}_seed{seed}" / "best_model.pt",
            map_location="cuda",
            weights_only=True,
        )
        model.load_state_dict(checkpoint["state_dict"])
        metrics, _ = evaluate(model, loader)
        random_gated, _ = evaluate(Gated(model), loader)
        block_gated, _ = evaluate(Gated(model), blockloader)
        reports.append(
            {
                "kind": kind,
                "seed": seed,
                "random_dropout_validation": metrics,
                "random_dropout_gated": random_gated,
                "block_dropout_gated": block_gated,
            }
        )
        print(kind, seed, metrics["model_all"]["mae_metres"], flush=True)

    (root / "cross_corruption_results.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )

    summary = {
        kind: {
            scenario: float(
                np.mean(
                    [
                        r[scenario]["model_all"]["mae_metres"]
                        for r in reports
                        if r["kind"] == kind
                    ]
                )
            )
            for scenario in [
                "random_dropout_validation",
                "random_dropout_gated",
                "block_dropout_gated",
            ]
        }
        for kind in ["rgb", "depth"]
    }

    chosen = min(summary, key=lambda k: summary[k]["block_dropout_gated"])

    candidates = sorted(
        [r for r in reports if r["kind"] == chosen],
        key=lambda r: r["block_dropout_gated"]["model_all"]["mae_metres"],
    )

    run = candidates[len(candidates) // 2]

    raw_model = ContextModel(chosen == "rgb").cuda().eval()

    checkpoint = torch.load(
        a.checkpoints / f"{chosen}_seed{run['seed']}" / "best_model.pt",
        map_location="cuda",
        weights_only=True,
    )

    raw_model.load_state_dict(checkpoint["state_dict"])

    model = Gated(raw_model)

    checkpoint.update(
        distance_gate_pixels=3,
        depth_normalization_metres=5,
        selection="lowest family mean block-dropout validation MAE; median seed within family",
        test_evaluated=False,
    )

    torch.save(checkpoint, root / "recommended_model.pt")

    summary.update(
        recommended_kind=chosen,
        representative_seed=run["seed"],
        distance_gate_pixels=3,
        test_evaluated=False,
        selection_caveat="Development validation comparison; an untouched test assessment is still required.",
    )

    (root / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    i = len(val) // 2

    x, y, mask, dist = val[i]

    with torch.no_grad():
        pred = model(x[None].cuda()).clamp(0, 4)[0, 0].cpu().numpy() * 5

    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    panels = [
        val.rgb[i],
        y[0].numpy() * 5,
        x[3].numpy() * 5,
        x[5].numpy() * 5,
        pred,
        np.abs(pred - y[0].numpy() * 5),
    ]

    titles = [
        "RGB",
        "Measured depth (unknown masked)",
        "Observed depth after dropout",
        "Nearest fill",
        "Context-model completion",
        "Model absolute error (hidden valid only)",
    ]

    for j, (ax, panel, title) in enumerate(zip(axes.flat, panels, titles)):
        if j == 0:
            ax.imshow(panel)
        else:
            valid = (
                (y[0].numpy() > 0)
                if j == 1
                else ((x[4].numpy() > 0) if j == 2 else np.ones_like(mask[0].numpy()))
            )
            if j == 5:
                valid = mask[0].numpy()
            masked = np.ma.masked_where(~valid, panel)
            im = ax.imshow(
                masked,
                cmap="viridis" if j < 5 else "magma",
                vmin=0,
                vmax=8 if j < 5 else 1,
            )
            fig.colorbar(im, ax=ax, fraction=0.045, label="metres")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f'Fixed validation sample {val.ids[i]}; {chosen} seed {run["seed"]}, distance gate 3 px'
    )

    fig.tight_layout()

    fig.savefig(root / "block_completion_preview.png", dpi=150)

    plt.close(fig)

    print("Cross-corruption validation and preview completed.", flush=True)


if __name__ == "__main__":
    main()
