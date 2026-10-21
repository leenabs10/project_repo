"""Reject model regressions without confusing execution success with data quality.

These development checks compare matched runs. Passing them does not certify
annotation accuracy, production fitness, or permission to use predictions as labels.
"""

import math
from statistics import mean

import numpy as np

MATCHED_CONTRACT_FIELDS = (
    "reference_profile_sha256",
    "protocol_sha256",
    "initial_checkpoint_sha256",
    "model_seeds",
    "epochs_per_seed",
    "batch_size",
    "training_resolution",
    "evaluation_resolution",
    "synthetic_frames",
    "test_access",
    "selection",
)


def background_false_positive_rate(prediction, target):
    """Measure false goods on reviewed background only; ignore goods and ID 255.

    Inputs are matching native or explicitly cropped HxW integer masks. Return
    None if the region contains no scored background; absence is never a pass.
    """
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or not np.issubdtype(prediction.dtype, np.integer)
        or not np.issubdtype(target.dtype, np.integer)
        or not np.isin(prediction, [0, 1]).all()
        or not np.isin(target, [0, 1, 255]).all()
    ):
        raise ValueError("Expected matching binary predictions and 0/1/255 targets")
    background = target == 0
    return float((prediction[background] == 1).mean()) if background.any() else None


def assess_relative_metrics(baseline_contract, candidate_contract, baseline, candidate):
    """Require matched seeds/budgets and nonregressing mean IoU, precision, recall.

    Rows contain a unique integer seed and finite development metrics in [0, 1].
    Different configurations/implementations are allowed; the comparison contract
    binds the reference version, initialization, geometry and training budget.
    """
    for key in MATCHED_CONTRACT_FIELDS:
        if key not in baseline_contract or baseline_contract[
            key
        ] != candidate_contract.get(key):
            raise ValueError("Unmatched comparison contract: " + key)
    seeds = baseline_contract["model_seeds"]
    if (
        not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(seeds) != len(set(seeds))
        or baseline_contract["test_access"] != "disabled"
    ):
        raise ValueError("Expected unique seeds and disabled held-out access")
    names = ("goods_iou", "precision", "recall")
    means = []
    for rows in (baseline, candidate):
        if len(rows) != len(seeds) or {row["seed"] for row in rows} != set(seeds):
            raise ValueError("Missing, duplicated or unmatched model seeds")
        for row in rows:
            if type(row["seed"]) is not int:
                raise ValueError("Seed must be an integer")
            for name in names:
                value = row["metrics"].get(name)
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError("Missing or invalid development metric: " + name)
        means.append(
            {name: mean(row["metrics"][name] for row in rows) for name in names}
        )
    checks = {
        name: {
            "baseline_mean": means[0][name],
            "candidate_mean": means[1][name],
            "delta": means[1][name] - means[0][name],
            "passed": means[1][name] >= means[0][name],
        }
        for name in names
    }
    return {
        "matched_contract": True,
        "relative_metric_checks": checks,
        "relative_metrics_passed": all(check["passed"] for check in checks.values()),
        "absolute_quality_certified": False,
        "model_predictions_may_be_used_as_reference_labels": False,
        "scope": "Reused development cohort; no independent generalization estimate",
    }
