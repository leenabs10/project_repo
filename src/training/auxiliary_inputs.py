"""Preserve primary real exposure while adding a bounded, class-balanced loss.

Replay and scene arms use the same auxiliary schedule, weight and foreground
target. Every auxiliary RGB is paired with its own target. No loss weight changes
the stored labels or turns a prediction into a supervised reference.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image

from src.adapters.storage.local import source_path


def validate_auxiliary_settings(settings):
    """Reject unspecified budgets and unbounded auxiliary influence."""
    if (
        not isinstance(settings, dict)
        or set(settings) - {"missing_class_policy"}
        != {"strategy", "interval_updates", "loss_weight", "foreground_fraction"}
        or settings.get("missing_class_policy", "reject")
        not in ("reject", "background_only_zero_foreground_mass")
        or settings["strategy"] != "preserve_real_exposure"
        or type(settings["interval_updates"]) is not int
        or not 2 <= settings["interval_updates"] <= 16
        or type(settings["loss_weight"]) not in (int, float)
        or not 0 < settings["loss_weight"] <= 0.25
        or type(settings["foreground_fraction"]) not in (int, float)
        or not 0 < settings["foreground_fraction"] < 1
    ):
        raise ValueError("Unsupported auxiliary exposure/loss policy")


def validate_auxiliary_sampling(config):
    """Validate legacy scene cycles or matched real/scene class-presence cycles.

    Omission preserves the original independent-hash sampler. Cadence, loss
    settings and primary real exposure belong to separate unchanged contracts.
    """
    if "auxiliary_sampling" not in config:
        return
    presence_cycle = config["auxiliary_sampling"] == {
        "strategy": "balanced_presence_cycle"
    }
    if (
        config["auxiliary_sampling"]
        not in (
            {"strategy": "balanced_scene_cycle"},
            {"strategy": "balanced_presence_cycle"},
        )
        or config.get("experiment_kind")
        not in (
            ("real_only", "real_plus_scene") if presence_cycle else ("real_plus_scene",)
        )
        or config.get("auxiliary_training") is None
        or (
            config.get("experiment_kind") == "real_plus_scene"
            and config.get("synthetic_mixing") != {"strategy": "auxiliary_only"}
        )
        or type(config.get("generation_seed")) is not int
        or config["generation_seed"] < 0
    ):
        raise ValueError("Balanced donor sampling requires scene auxiliary inputs")
    validate_auxiliary_settings(config["auxiliary_training"])
    if (
        presence_cycle
        and config["auxiliary_training"].get("missing_class_policy")
        != "background_only_zero_foreground_mass"
    ):
        raise ValueError(
            "Presence-balanced cycles require explicit background-only supervision"
        )


def prepare_auxiliary_donors(references, real_rows, scene_root, scenes, config):
    """Validate the loss target against real training pixels and qualify donors.

    Count masks at the exact nearest-neighbour training resolution.
    Background-only donors require the explicit missing-class policy. All original
    real frames still participate in the unchanged primary training stream.
    """
    settings = config["auxiliary_training"]
    validate_auxiliary_settings(settings)
    allow_negative = (
        settings.get("missing_class_policy") == "background_only_zero_foreground_mass"
    )
    foreground, background = 0, 0
    real_candidates = []
    for row in real_rows:
        if row["split"] != "train":
            continue
        with Image.open(source_path(Path(references), row["target_path"])) as image:
            target = np.array(
                image.resize(
                    tuple(config["training_resolution"]), Image.Resampling.NEAREST
                )
            )
        fg, bg = int((target == 1).sum()), int((target == 0).sum())
        foreground += fg
        background += bg
        if bg and (fg or allow_negative):
            donor_row = dict(row)
            if allow_negative:
                donor_row["auxiliary_target_kind"] = (
                    "mixed" if fg else "background_only"
                )
            real_candidates.append((Path(references), donor_row))
    fraction = (
        foreground / (foreground + background) if foreground + background else None
    )
    if fraction is None or not math.isclose(
        fraction, settings["foreground_fraction"], rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(
            "Auxiliary foreground balance differs from real training references"
        )
    if config["experiment_kind"] == "real_only":
        donors = real_candidates
    else:
        donors = []
        for row in scenes:
            if row["split"] != "train":
                raise ValueError("Auxiliary scenes must be training-only")
            with Image.open(source_path(Path(scene_root), row["target_path"])) as image:
                target = np.array(
                    image.resize(
                        tuple(config["training_resolution"]), Image.Resampling.NEAREST
                    )
                )
            if np.any(target == 0) and (np.any(target == 1) or allow_negative):
                donor_row = dict(row)
                if allow_negative:
                    donor_row["auxiliary_target_kind"] = (
                        "mixed" if np.any(target == 1) else "background_only"
                    )
                donors.append((Path(scene_root), donor_row))
        if len(donors) != len(scenes):
            raise ValueError(
                "An auxiliary scene lacks foreground or background supervision"
            )
    if not donors:
        raise ValueError("No eligible auxiliary RGB/target donors")
    if config.get("auxiliary_sampling") == {"strategy": "balanced_presence_cycle"} and {
        row.get("auxiliary_target_kind") for _, row in donors
    } != {"mixed", "background_only"}:
        raise ValueError("Presence-balanced cycles need both negative and mixed donors")
    return sorted(donors, key=lambda value: value[1]["sample_id"])


def select_auxiliary_donor(donors, config, update):
    """Select one auxiliary input without consuming primary/model random state.

    The scene policy visits each donor once per cycle. The presence policy
    alternates background-only and mixed inputs, balancing within each group.
    A stable hash
    permutation changes order between cycles, so every prefix differs by at most
    one visit across donors. The legacy policy remains byte-for-byte equivalent
    in its hash inputs and indexing when no sampling policy is declared.
    """
    if type(update) is not int or update < 1 or not donors:
        raise ValueError(
            "Auxiliary scheduling requires donors and a positive update index"
        )
    validate_auxiliary_sampling(config)
    interval = config["auxiliary_training"]["interval_updates"]
    if update % interval:
        return None
    if "auxiliary_sampling" in config:
        identifiers = [row.get("sample_id") for _, row in donors]
        if (
            any(
                not isinstance(identifier, str) or not identifier
                for identifier in identifiers
            )
            or len(set(identifiers)) != len(identifiers)
            or any(row.get("split") != "train" for _, row in donors)
        ):
            raise ValueError("Balanced cycles require unique training-only donors")
        # Updates are one-based; the first scheduled auxiliary input is index 0.
        ordinal = update // interval - 1
        if config["auxiliary_sampling"]["strategy"] == "balanced_presence_cycle":
            if any(
                row.get("auxiliary_target_kind") not in ("mixed", "background_only")
                for _, row in donors
            ):
                raise ValueError("Presence cycle donors require verified target kinds")
            kind = "background_only" if ordinal % 2 == 0 else "mixed"
            donors = [
                donor for donor in donors if donor[1]["auxiliary_target_kind"] == kind
            ]
            if not donors:
                raise ValueError("Presence cycle is missing a required donor kind")
            ordinal //= 2
        cycle, position = divmod(ordinal, len(donors))
        ordered = sorted(
            donors,
            key=lambda donor: (
                hashlib.sha256(
                    f"{config['generation_seed']}:{cycle}:{donor[1]['sample_id']}:auxiliary-balanced".encode()
                ).digest(),
                donor[1]["sample_id"],
            ),
        )
        return ordered[position]
    digest = hashlib.sha256(
        f"{config['generation_seed']}:{update}:auxiliary".encode()
    ).digest()
    return donors[int.from_bytes(digest[:8], "big") % len(donors)]


def balanced_auxiliary_loss(
    logits, targets, foreground_fraction, *, missing_class_policy="reject"
):
    """Apply fixed training-derived class mass; never renormalize missing goods.

    Logits are N,2,H,W floats; targets are N,H,W categorical IDs 0,1,255.
    The opt-in negative loss is (1-f)*mean(background CE), with zero foreground
    mass. Ignored pixels have zero gradient. Foreground-only/all-ignore inputs,
    unsupported classes/policies and nonfinite logits or losses are rejected.
    """
    import torch
    from torch.nn import functional

    if (
        missing_class_policy not in ("reject", "background_only_zero_foreground_mass")
        or not math.isfinite(foreground_fraction)
        or not 0 < foreground_fraction < 1
    ):
        raise ValueError("Unsupported auxiliary missing-class or foreground policy")
    if (
        logits.ndim != 4
        or logits.shape[1] != 2
        or targets.shape != (logits.shape[0], *logits.shape[2:])
        or not bool(torch.isfinite(logits).all())
        or not bool(((targets == 0) | (targets == 1) | (targets == 255)).all())
    ):
        raise ValueError("Invalid auxiliary logits/targets")
    foreground, background = targets == 1, targets == 0
    if not bool(background.any()) or (
        not bool(foreground.any()) and missing_class_policy == "reject"
    ):
        raise ValueError("Auxiliary balancing requires both scorable classes")
    pixels = functional.cross_entropy(
        logits, targets, ignore_index=255, reduction="none"
    )
    loss = (1 - foreground_fraction) * pixels[background].mean()
    if bool(foreground.any()):
        loss = loss + foreground_fraction * pixels[foreground].mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite auxiliary loss")
    return loss
