"""Visible-goods target rules and metrics, with explicit uncertainty and provenance."""

import json
from pathlib import Path

import numpy as np


def load_definition(path: Path) -> dict:
    """Load this task's supported binary IDs; reject accidental semantic remapping."""
    definition = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        definition.get("schema_version") != "dubaidata/1.0"
        or definition.get("task_id") != "warehouse_visible_goods"
        or definition.get("target_ids") != {"background": 0, "goods": 1, "ignore": 255}
        or definition.get("reference_origin") != "human_reviewed_model_assisted"
        or not 0 <= definition.get("maximum_ignore_fraction", -1) < 1
        or set(definition.get("minimum_reference_frames", {}))
        != {"train", "validation"}
        or any(
            type(count) is not int or count < 1
            for count in definition["minimum_reference_frames"].values()
        )
        or type(definition.get("source_candidate_class_id")) is not int
        or not 0 <= definition["source_candidate_class_id"] <= 255
    ):
        raise ValueError("Unsupported or malformed goods task definition")
    return definition


def candidate_target(semantic: np.ndarray, class_id: int) -> np.ndarray:
    """Derive a proposal only; this operation never establishes reference quality."""
    if semantic.dtype != np.uint8 or semantic.ndim != 2:
        raise ValueError("Candidate semantics must be a uint8 class-ID image")
    return (semantic == class_id).astype(np.uint8)


def validate_target(
    mask: np.ndarray, shape: tuple[int, int], maximum_ignore_fraction: float
) -> None:
    """Require native uint8 0/1/255 labels and sufficient nonignored supervision."""
    if (
        mask.shape != shape
        or mask.dtype != np.uint8
        or not np.isin(mask, [0, 1, 255]).all()
    ):
        raise ValueError("Goods target must match native geometry and IDs 0/1/255")
    if float((mask == 255).mean()) > maximum_ignore_fraction or np.all(mask == 255):
        raise ValueError("Goods reference exceeds the ignored-pixel limit")


def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Count binary predictions over valid targets only; never score ignored pixels."""
    if prediction.shape != target.shape or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Expected matching binary predictions")
    validate_target(target, target.shape, 1.0)
    valid = target != 255
    return {
        "true_positive": int(((prediction == 1) & (target == 1) & valid).sum()),
        "false_positive": int(((prediction == 1) & (target == 0) & valid).sum()),
        "false_negative": int(((prediction == 0) & (target == 1) & valid).sum()),
        "true_negative": int(((prediction == 0) & (target == 0) & valid).sum()),
        "valid_pixels": int(valid.sum()),
    }


def segmentation_metrics(counts: dict) -> dict:
    """Return pixel-weighted foreground metrics; an empty denominator yields null."""
    tp, fp, fn = (
        counts[name] for name in ("true_positive", "false_positive", "false_negative")
    )
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("Confusion counts must be nonnegative integers")

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "goods_iou": ratio(tp, tp + fp + fn),
        "goods_dice": ratio(2 * tp, 2 * tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "valid_pixels": counts["valid_pixels"],
    }
