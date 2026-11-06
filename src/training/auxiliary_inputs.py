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
    """Validate legacy cycles or an explicitly frozen auxiliary update sequence.

    Omission preserves the original independent-hash sampler. Cadence, loss
    settings and primary real exposure belong to separate unchanged contracts.
    """
    if "auxiliary_sampling" not in config:
        return
    sampling = config["auxiliary_sampling"]
    if (
        isinstance(sampling, dict)
        and sampling.get("strategy")
        in {"explicit_update_sequence", "explicit_mixed_source_sequence"}
    ):
        _validate_explicit_sampling(config)
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


def _validate_explicit_sampling(config):
    """Check sequence syntax without reading data or consuming random state."""
    sampling = config["auxiliary_sampling"]
    mixed_sources = sampling["strategy"] == "explicit_mixed_source_sequence"
    if (
        set(sampling) != {"strategy", "events"}
        or not isinstance(sampling["events"], list)
        or config.get("experiment_kind") not in ("real_only", "real_plus_scene")
        or (mixed_sources and config.get("experiment_kind") != "real_plus_scene")
        or config.get("auxiliary_training") is None
        or (
            config.get("experiment_kind") == "real_plus_scene"
            and config.get("synthetic_mixing") != {"strategy": "auxiliary_only"}
        )
        or type(config.get("generation_seed")) is not int
        or config["generation_seed"] < 0
    ):
        raise ValueError(
            "Explicit auxiliary sequences require bounded auxiliary inputs"
        )
    validate_auxiliary_settings(config["auxiliary_training"])
    interval = config["auxiliary_training"]["interval_updates"]
    previous_update = 0
    event_fields = {"update", "sample_id", "target_kind"}
    if mixed_sources:
        event_fields.add("input_source")
    for event in sampling["events"]:
        if (
            not isinstance(event, dict)
            or set(event) != event_fields
            or type(event["update"]) is not int
            or event["update"] <= previous_update
            or event["update"] % interval
            or not isinstance(event["sample_id"], str)
            or not event["sample_id"]
            or event["target_kind"] not in ("mixed", "background_only")
            or (
                mixed_sources
                and event["input_source"]
                not in ("real_training_reference", "admitted_scene")
            )
        ):
            raise ValueError(
                "Explicit auxiliary events must be ordered, unique and on cadence"
            )
        previous_update = event["update"]
    if mixed_sources and {
        event["input_source"] for event in sampling["events"]
    } != {"real_training_reference", "admitted_scene"}:
        raise ValueError("Mixed-source sequences must explicitly use both input sources")


def _validate_explicit_target(image):
    """Reject invalid native categorical masks before nearest-neighbour resizing."""
    values = np.asarray(image)
    if (
        values.ndim != 2
        or values.dtype != np.uint8
        or not values.size
        or not np.isin(values, (0, 1, 255)).all()
    ):
        raise ValueError("Explicit auxiliary donors require uint8 categorical masks")


def _explicit_donors_by_id(donors, events):
    """Bind every declared event to a unique training donor and verified mask kind."""
    by_id = {}
    for donor in donors:
        row = donor[1]
        identifier = row.get("sample_id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in by_id
            or row.get("split") != "train"
            or row.get("auxiliary_target_kind") not in ("mixed", "background_only")
        ):
            raise ValueError(
                "Explicit sequences require unique training-only donors and verified kinds"
            )
        by_id[identifier] = donor
    for event in events:
        donor = by_id.get(event["sample_id"])
        if donor is None or donor[1]["auxiliary_target_kind"] != event["target_kind"]:
            raise ValueError(
                "Explicit auxiliary event has an unknown donor or wrong target kind"
            )
        if (
            "input_source" in event
            and donor[1].get("auxiliary_input_source") != event["input_source"]
        ):
            raise ValueError("Explicit auxiliary event selects the wrong input source")
    return by_id


def _validate_explicit_budget(donors, config, training_count):
    """Require one event for every auxiliary update in the entire real-data budget."""
    if training_count < 1 or any(
        type(config.get(field)) is not int or config[field] < 1
        for field in ("epochs", "batch_size")
    ):
        raise ValueError(
            "Explicit sequences require a positive real-training update budget"
        )
    total_updates = config["epochs"] * math.ceil(training_count / config["batch_size"])
    interval = config["auxiliary_training"]["interval_updates"]
    events = config["auxiliary_sampling"]["events"]
    if [event["update"] for event in events] != list(
        range(interval, total_updates + 1, interval)
    ):
        raise ValueError(
            "Explicit auxiliary sequence does not cover the complete update budget"
        )
    _explicit_donors_by_id(donors, events)


def prepare_auxiliary_donors(references, real_rows, scene_root, scenes, config):
    """Validate the loss target against real training pixels and qualify donors.

    Count masks at the exact nearest-neighbour training resolution.
    Background-only donors require the explicit missing-class policy. All original
    real frames still participate in the unchanged primary training stream.
    Explicit sequences additionally verify native uint8 IDs (0, 1, 255), donor
    identity and kind, and complete epoch/batch coverage before any model is fit.
    Mixed-source sequences bind each event to either a qualified real training
    reference or an already admitted scene. Each returned pair retains its own
    root and RGB/target paths; validation/development rows cannot be selected.
    """
    settings = config["auxiliary_training"]
    validate_auxiliary_settings(settings)
    sampling = config.get("auxiliary_sampling")
    explicit = (
        isinstance(sampling, dict)
        and sampling.get("strategy")
        in {"explicit_update_sequence", "explicit_mixed_source_sequence"}
    )
    mixed_sources = (
        explicit and sampling["strategy"] == "explicit_mixed_source_sequence"
    )
    if explicit:
        validate_auxiliary_sampling(config)
    if mixed_sources and (
        scene_root is None or Path(scene_root).resolve() == Path(references).resolve()
    ):
        raise ValueError(
            "Mixed-source donors require distinct real and admitted-scene roots"
        )
    allow_negative = (
        settings.get("missing_class_policy") == "background_only_zero_foreground_mass"
    )
    foreground, background = 0, 0
    real_candidates = []
    for row in real_rows:
        if row["split"] != "train":
            continue
        with Image.open(source_path(Path(references), row["target_path"])) as image:
            if explicit:
                _validate_explicit_target(image)
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
            if mixed_sources:
                donor_row["auxiliary_input_source"] = "real_training_reference"
            if allow_negative or explicit:
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
                if explicit:
                    _validate_explicit_target(image)
                target = np.array(
                    image.resize(
                        tuple(config["training_resolution"]), Image.Resampling.NEAREST
                    )
                )
            if np.any(target == 0) and (np.any(target == 1) or allow_negative):
                donor_row = dict(row)
                if mixed_sources:
                    donor_row["auxiliary_input_source"] = "admitted_scene"
                if allow_negative or explicit:
                    donor_row["auxiliary_target_kind"] = (
                        "mixed" if np.any(target == 1) else "background_only"
                    )
                donors.append((Path(scene_root), donor_row))
        if len(donors) != len(scenes):
            raise ValueError(
                "An auxiliary scene lacks foreground or background supervision"
            )
        if mixed_sources:
            donors.extend(real_candidates)
    if not donors:
        raise ValueError("No eligible auxiliary RGB/target donors")
    if config.get("auxiliary_sampling") == {"strategy": "balanced_presence_cycle"} and {
        row.get("auxiliary_target_kind") for _, row in donors
    } != {"mixed", "background_only"}:
        raise ValueError("Presence-balanced cycles need both negative and mixed donors")
    if explicit:
        _validate_explicit_budget(
            donors, config, sum(row["split"] == "train" for row in real_rows)
        )
    return sorted(donors, key=lambda value: value[1]["sample_id"])


def select_auxiliary_donor(donors, config, update):
    """Select one auxiliary input without consuming primary/model random state.

    The scene policy visits each donor once per cycle. The presence policy
    alternates background-only and mixed inputs, balancing within each group.
    A stable hash
    permutation changes order between cycles, so every prefix differs by at most
    one visit across donors. The legacy policy remains byte-for-byte equivalent
    in its hash inputs and indexing when no sampling policy is declared.
    Explicit sequences use the supplied update/identity/kind binding directly;
    changing a job-derived sample ID therefore cannot silently reshuffle exposure.
    """
    if type(update) is not int or update < 1 or not donors:
        raise ValueError(
            "Auxiliary scheduling requires donors and a positive update index"
        )
    validate_auxiliary_sampling(config)
    interval = config["auxiliary_training"]["interval_updates"]
    if update % interval:
        return None
    if (
        config.get("auxiliary_sampling", {}).get("strategy")
        in {"explicit_update_sequence", "explicit_mixed_source_sequence"}
    ):
        events = config["auxiliary_sampling"]["events"]
        by_id = _explicit_donors_by_id(donors, events)
        ordinal = update // interval - 1
        if ordinal >= len(events) or events[ordinal]["update"] != update:
            raise ValueError("No explicit auxiliary event for the requested update")
        return by_id[events[ordinal]["sample_id"]]
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
