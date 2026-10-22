"""Native goods predictions and error galleries from verified model snapshots."""

from html import escape
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.training.goods_benchmark import build_model, input_tensor, load_references
from src.adapters.tasks.goods import confusion_counts, segmentation_metrics
from src.validation.artifacts import artifact_hashes, verify_snapshot, write_json


def error_colours(prediction, target):
    """Return RGB error classes at native geometry, excluding ignored supervision.

    Green is correct goods, magenta false goods, yellow missed goods, orange
    ignored target and dark grey correct background. Ignored pixels are never
    counted as model errors, regardless of their prediction.
    """
    confusion_counts(prediction, target)
    colours = np.full((*target.shape, 3), 35, dtype=np.uint8)
    colours[(prediction == 1) & (target == 1)] = [40, 200, 90]
    colours[(prediction == 1) & (target == 0)] = [235, 70, 210]
    colours[(prediction == 0) & (target == 1)] = [255, 225, 0]
    colours[target == 255] = [255, 160, 0]
    return colours


def label_overlay(rgb, labels):
    """Blend class colours into an RGB display copy; never alter label values."""
    if (
        rgb.dtype != np.uint8
        or rgb.shape != (*labels.shape, 3)
        or labels.dtype != np.uint8
        or not np.isin(labels, [0, 1, 255]).all()
    ):
        raise ValueError("Expected matching native RGB and class-ID arrays")
    colour = np.empty_like(rgb)
    colour[labels == 0] = [0, 180, 210]
    colour[labels == 1] = [235, 70, 210]
    colour[labels == 255] = [255, 160, 0]
    return np.rint(rgb.astype(np.float32) * 0.55 + colour * 0.45).astype(np.uint8)


def export_predictions(run, run_sha, references, references_sha, output, device="cuda"):
    """Reload one verified goods model and export lossless native predictions.

    All train/development predictions and metrics are retained. The gallery
    shows all development frames and six evenly spaced training diagnostics.
    Train metrics describe fitting; development metrics were used for selection.
    This function does not read the raw dataset or held-out test trajectory.
    """
    import torch
    from torch.nn import functional

    run, references = Path(run), Path(references)
    profile = verify_snapshot(run, run_sha)
    if (
        profile["purpose"] not in {"real_only_benchmark", "hybrid_benchmark"}
        or profile["status"] != "passed"
    ):
        raise ValueError("Prediction export requires a completed goods benchmark run")
    if profile["reference_profile_sha256"] != references_sha:
        raise ValueError("Prediction references differ from model-selection references")
    rows = load_references(references, references_sha, run / "task.json")
    config = json.loads((run / "configuration.json").read_text())
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    for name in [
        "seed",
        "reference_profile_sha256",
        "protocol_sha256",
        "configuration_sha256",
    ]:
        if saved[name] != profile[name]:
            raise ValueError("Prediction checkpoint metadata differs from its run")
    if config["experiment_kind"] == "real_plus_synthetic" and (
        saved.get("synthetic_profile_sha256") != profile.get("synthetic_profile_sha256")
        or not saved.get("synthetic_profile_sha256")
    ):
        raise ValueError(
            "Prediction checkpoint synthetic provenance differs from its run"
        )
    model = build_model(config).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    output = reserve_output(
        output, (run, references, Path(__file__).parents[2] / "data/raw")
    )
    training = [row for row in rows if row["split"] == "train"]
    selected = {
        training[index]["sample_id"]
        for index in np.linspace(0, len(training) - 1, min(6, len(training)), dtype=int)
    }
    counts = {
        split: {
            name: 0
            for name in [
                "true_positive",
                "false_positive",
                "false_negative",
                "true_negative",
                "valid_pixels",
            ]
        }
        for split in ["train", "validation"]
    }
    frame_rows, cards = [], []
    with torch.no_grad():
        for row in rows:
            logits = model(input_tensor(references, row, config)[None].to(device))[
                "out"
            ]
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite prediction logits")
            logits = functional.interpolate(
                logits,
                size=tuple(reversed(row["image_size"])),
                mode="bilinear",
                align_corners=False,
            )
            prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            with Image.open(source_path(references, row["target_path"])) as image:
                target = np.array(image)
            frame_counts = confusion_counts(prediction, target)
            for name, value in frame_counts.items():
                counts[row["split"]][name] += value
            metrics = segmentation_metrics(frame_counts)
            identity = hashlib.sha256(row["sample_id"].encode()).hexdigest()[:24]
            folder = output / "samples" / identity
            folder.mkdir(parents=True)
            prediction_path = folder / "prediction.png"
            Image.fromarray(prediction).save(prediction_path)
            with Image.open(prediction_path) as image:
                if not np.array_equal(np.array(image), prediction):
                    raise ValueError("Prediction PNG is not lossless")
            frame_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "prediction_path": prediction_path.relative_to(output).as_posix(),
                    "prediction_sha256": file_hash(prediction_path),
                    "target_sha256": row["target_sha256"],
                    "ignore_fraction": float((target == 255).mean()),
                    **metrics,
                }
            )
            if row["split"] == "validation" or row["sample_id"] in selected:
                with Image.open(source_path(references, row["rgb_path"])) as image:
                    rgb = np.array(image)
                images = {
                    "RGB": rgb,
                    "Reviewed reference": label_overlay(rgb, target),
                    "Prediction": label_overlay(rgb, prediction),
                    "Errors": error_colours(prediction, target),
                }
                panels = []
                for title, values in images.items():
                    # Display panels must never overwrite the binary prediction.
                    filename = "display_" + title.lower().replace(" ", "_") + ".png"
                    Image.fromarray(values).save(folder / filename)
                    relative = (folder / filename).relative_to(output).as_posix()
                    panels.append(
                        f'<figure><figcaption>{escape(title)}</figcaption><a href="{relative}"><img src="{relative}" loading="lazy"></a></figure>'
                    )
                score = (
                    "undefined"
                    if metrics["goods_iou"] is None
                    else f'{metrics["goods_iou"]:.4f}'
                )
                cards.append(
                    f'<article><h2>{escape(row["sample_id"])} ({row["split"]})</h2><p>Goods IoU: {score}; ignored target: {100 * float((target == 255).mean()):.1f}%</p><div class="panels">{"".join(panels)}</div></article>'
                )
    split_metrics = {
        split: {**segmentation_metrics(value), "confusion": value}
        for split, value in counts.items()
    }
    for row in frame_rows:
        if file_hash(output / row["prediction_path"]) != row["prediction_sha256"]:
            raise ValueError("Native prediction changed during gallery export")
    # Independently exported pixels must reproduce the recorded selected metrics.
    selected_metrics = json.loads((run / "validation_metrics.json").read_text())
    if split_metrics["validation"]["confusion"] != selected_metrics["confusion"]:
        raise ValueError(
            "Exported predictions differ from selected-checkpoint evaluation"
        )
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in frame_rows), encoding="utf-8"
    )
    write_json(output / "split_metrics.json", split_metrics)
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Goods prediction diagnostics</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#182230}article{background:white;padding:16px;margin:20px 0}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:auto}h2{font-size:18px}</style>
<a href="../../index.html">Baseline summary</a><h1>Goods prediction diagnostics</h1>
<p>Reference/prediction overlays: magenta goods, cyan background, orange ignored reference. Prediction overlays cover the complete image; only valid reference pixels are scored.</p>
<p>Error colours: green correct goods; magenta false goods; yellow missed goods; orange ignored target; dark grey correct background. Click a panel for its native image.</p>
<p>Development frames were used for checkpoint selection. Training frames diagnose fitting. Neither is a held-out test result. References are model-assisted corrections.</p>"""
    (output / "index.html").write_text(
        html + "".join(cards) + "</html>", encoding="utf-8"
    )
    verify_snapshot(references, references_sha)
    verify_snapshot(run, run_sha)
    report = {
        "status": "passed",
        "seed": profile["seed"],
        "run_profile_sha256": run_sha,
        "reference_profile_sha256": references_sha,
        "prediction_count": len(frame_rows),
        "gallery_frames": len(cards),
        "native_prediction_roundtrip": "passed",
        "recorded_confusion_reproduced": True,
        "test_sensor_files_opened": 0,
        "artifact_sha256": artifact_hashes(output),
    }
    write_json(output / "profile.json", report)
    return report
