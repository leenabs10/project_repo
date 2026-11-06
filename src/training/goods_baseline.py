"""Freeze and execute matched real-only goods seeds with native diagnostics."""

import argparse
from collections import Counter
import gc
import json
import logging
from pathlib import Path
import shutil
import statistics

import numpy as np
from PIL import Image

from src.core.paths import workspace_root
from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.adapters.tasks.goods import confusion_counts, segmentation_metrics
from src.training.goods_benchmark import (
    evaluate_run,
    load_references,
    train_run,
    validate_configuration,
)
from src.validation.artifacts import (
    artifact_hashes,
    verify_snapshot,
    write_implementation_snapshot,
    write_json,
)
from src.validation.goods_predictions import export_predictions


def metric_text(value):
    """Keep undefined foreground metrics visible rather than formatting them as zero."""
    return "undefined" if value is None else f"{value:.4f}"


def summarize_seeds(runs, configuration, references_sha):
    """Reject partial or mismatched runs before summarizing seed variability.

    Standard deviation measures variation across model seeds on the same small
    development trajectory. It is not an independent-sample confidence interval.
    """
    seeds = configuration["model_seeds"]
    if len(seeds) != len(set(seeds)) or sorted(row["seed"] for row in runs) != sorted(
        seeds
    ):
        raise ValueError("Every configured seed must complete exactly once")
    for row in runs:
        if (
            row["status"] != "passed"
            or row["reference_profile_sha256"] != references_sha
            or row["synthetic_training_frames"] != 0
            or row["optimizer_updates"] != row["expected_optimizer_updates"]
            or row["strict_reload"] != "passed"
        ):
            raise ValueError("Baseline seed failed provenance, budget or reload checks")
    budget_keys = [
        "optimizer_updates",
        "real_training_frames",
        "development_frames",
        "initial_checkpoint_sha256",
        "configuration_sha256",
        "protocol_sha256",
    ]
    if any(len({row[key] for row in runs}) != 1 for key in budget_keys):
        raise ValueError("Seed budgets or input snapshots are not matched")
    scores = [row["selected_validation_goods_iou"] for row in runs]
    if not all(np.isfinite(score) for score in scores):
        raise ValueError("Seed metrics must be finite")
    return {
        "model_seeds": seeds,
        "seed_count": len(runs),
        "development_goods_iou_mean": statistics.mean(scores),
        "development_goods_iou_sample_std": (
            statistics.stdev(scores) if len(scores) > 1 else None
        ),
        "development_goods_iou_min": min(scores),
        "development_goods_iou_max": max(scores),
        "optimizer_updates_per_seed": runs[0]["optimizer_updates"],
        "uncertainty_scope": "Model-seed variability on a reused development trajectory; not an independent-sample confidence interval",
    }


def reference_statistics(references, rows):
    """Record native class coverage and an always-background control per split."""
    totals = {split: Counter() for split in ["train", "validation"]}
    controls = {split: Counter() for split in totals}
    for row in rows:
        with Image.open(source_path(references, row["target_path"])) as image:
            target = np.array(image)
        values, counts = np.unique(target, return_counts=True)
        totals[row["split"]].update(
            {int(value): int(count) for value, count in zip(values, counts)}
        )
        controls[row["split"]].update(confusion_counts(np.zeros_like(target), target))
    return {
        split: {
            "background_pixels": count[0],
            "goods_pixels": count[1],
            "ignored_pixels": count[255],
            "valid_fraction": (count[0] + count[1]) / sum(count.values()),
            "goods_fraction_among_valid": count[1] / (count[0] + count[1]),
            "always_background": segmentation_metrics(dict(controls[split])),
        }
        for split, count in totals.items()
    }


def run_baseline(
    references, references_sha, task, configuration, checkpoint, output, device="cuda"
):
    """Run all frozen seeds sequentially; preserve each run and reject partial success."""
    references, task, configuration, checkpoint = map(
        Path, [references, task, configuration, checkpoint]
    )
    config = json.loads(configuration.read_text())
    validate_configuration(config)
    rows = load_references(references, references_sha, task)
    if file_hash(checkpoint) != config["initial_checkpoint_sha256"]:
        raise ValueError("Pinned initialization checksum mismatch")
    if len(set(config["model_seeds"])) != len(config["model_seeds"]):
        raise ValueError("Baseline seeds must be unique")
    output = reserve_output(
        output,
        (
            references,
            task,
            configuration,
            checkpoint,
            workspace_root() / "data/raw",
        ),
    )
    write_json(
        output / "profile.json", {"status": "running", "baseline_complete": False}
    )
    shutil.copyfile(configuration, output / "configuration.json")
    shutil.copyfile(task, output / "task.json")
    write_implementation_snapshot(output)
    frozen_implementation = json.loads((output / "implementation.json").read_text())[
        "files"
    ]
    contract = {
        "reference_profile_sha256": references_sha,
        "configuration_sha256": file_hash(configuration),
        "protocol_sha256": file_hash(task),
        "initial_checkpoint_sha256": file_hash(checkpoint),
        "model_seeds": config["model_seeds"],
        "epochs_per_seed": config["epochs"],
        "batch_size": config["batch_size"],
        "training_resolution": config["training_resolution"],
        "evaluation_resolution": config["native_reference_resolution"],
        "synthetic_frames": 0,
        "test_access": "disabled",
        "selection": "maximum development goods IoU; development reused for selection",
        "implementation_sha256": file_hash(output / "implementation.json"),
    }
    write_json(output / "contract.json", contract)
    write_json(
        output / "reference_statistics.json", reference_statistics(references, rows)
    )
    runs, cards = [], []
    try:
        for seed in config["model_seeds"]:
            logging.info("Starting frozen real-only seed %s", seed)
            run = output / f"seed_{seed}"
            report = train_run(
                references,
                references_sha,
                output / "task.json",
                output / "configuration.json",
                checkpoint,
                run,
                seed,
                device,
            )
            run_sha = file_hash(run / "profile.json")
            evaluation = output / f"evaluation_{seed}"
            evaluate_run(run, run_sha, references, references_sha, evaluation, device)
            selected = json.loads((run / "validation_metrics.json").read_text())
            if selected != json.loads((evaluation / "metrics.json").read_text()):
                raise ValueError(
                    "Separate evaluation differs from the selected checkpoint"
                )
            prediction = output / "predictions" / f"seed_{seed}"
            export_predictions(
                run, run_sha, references, references_sha, prediction, device
            )
            runs.append(report)
            cards.append(
                {
                    "seed": seed,
                    "run_profile_sha256": run_sha,
                    "evaluation_profile_sha256": file_hash(evaluation / "profile.json"),
                    "prediction_profile_sha256": file_hash(prediction / "profile.json"),
                    "best_epoch": report["best_epoch"],
                    "metrics": selected,
                    "elapsed_seconds": report["elapsed_seconds"],
                    "peak_cuda_allocated_bytes": report["peak_cuda_allocated_bytes"],
                }
            )
            write_json(
                output / "progress.json",
                {
                    "completed_seeds": [row["seed"] for row in runs],
                    "configured_seeds": config["model_seeds"],
                },
            )
            # Release allocator caches between seeds; the batch never runs GPU jobs concurrently.
            gc.collect()
            if device == "cuda":
                import torch

                torch.cuda.empty_cache()
        summary = summarize_seeds(runs, config, references_sha)
        for name, digest in frozen_implementation.items():
            if file_hash(Path(__file__).parents[2] / name) != digest:
                raise ValueError("Implementation changed during the frozen experiment")
        verify_snapshot(references, references_sha)
        if (
            file_hash(configuration) != contract["configuration_sha256"]
            or file_hash(task) != contract["protocol_sha256"]
            or file_hash(checkpoint) != contract["initial_checkpoint_sha256"]
        ):
            raise ValueError("Frozen inputs changed during the experiment")
        write_json(output / "seed_results.json", cards)
        write_json(output / "summary.json", summary)
        table = "".join(
            f'<tr><td>{row["seed"]}</td><td>{row["best_epoch"]}</td><td>{metric_text(row["metrics"]["goods_iou"])}</td><td>{metric_text(row["metrics"]["goods_dice"])}</td><td>{metric_text(row["metrics"]["precision"])}</td><td>{metric_text(row["metrics"]["recall"])}</td><td><a href="predictions/seed_{row["seed"]}/index.html">Native predictions</a></td></tr>'
            for row in cards
        )
        html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Goods real-only baseline</title><style>body{{font:17px system-ui;margin:30px;max-width:1200px;color:#182230}}table{{border-collapse:collapse}}td,th{{padding:12px;border:1px solid #ccd3dc}}a{{color:#164ca0}}</style><h1>Goods real-only baseline</h1><p>{runs[0]["real_training_frames"]} training / {runs[0]["development_frames"]} development references; source-bound reviewed annotations; RGB only. Training {config["training_resolution"]} float32; native evaluation {config["native_reference_resolution"]}. No held-out evaluation.</p><p>Development goods IoU: mean {metric_text(summary["development_goods_iou_mean"])}; sample standard deviation {metric_text(summary["development_goods_iou_sample_std"])} across the configured seeds. This is seed variability, not a generalization confidence interval.</p><table><tr><th>Seed</th><th>Selected epoch</th><th>IoU</th><th>Dice</th><th>Precision</th><th>Recall</th><th>Evidence</th></tr>{table}</table><p>Each checkpoint was selected on development IoU and independently reloaded for evaluation. The small, dependent development cohort and potential annotation anchoring bias limit interpretation. Synthetic-data utility has not been measured.</p></html>"""
        (output / "index.html").write_text(html, encoding="utf-8")
        result = {
            "status": "passed",
            "baseline_complete": True,
            "matched_synthetic_comparison_complete": False,
            "held_out_test_evaluated": False,
            "reference_profile_sha256": references_sha,
            "seeds": config["model_seeds"],
            "summary": summary,
            "separate_evaluation_and_native_export": "passed",
            "test_sensor_files_opened": 0,
            "artifact_sha256": artifact_hashes(output),
        }
        write_json(output / "profile.json", result)
        return result
    except Exception as error:
        write_json(
            output / "profile.json",
            {
                "status": "failed",
                "baseline_complete": False,
                "completed_seeds": [row["seed"] for row in runs],
                "error": str(error),
            },
        )
        raise


def main():
    """Run one fresh frozen baseline batch with explicit source and output paths."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["references", "task", "configuration", "checkpoint", "output"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--references-sha256", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    result = run_baseline(
        args.references,
        args.references_sha256,
        args.task,
        args.configuration,
        args.checkpoint,
        args.output,
        args.device,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "artifact_sha256"},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
