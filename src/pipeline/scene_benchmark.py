"""Run and compare fixed-final real-only and admitted-scene goods experiments."""

import argparse
import json
import logging
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from src.core.paths import workspace_root
from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.training.goods_benchmark import (
    checkpoint_selection_metric,
    evaluate_run,
    load_references,
    train_run,
    validate_configuration,
)
from src.training.scene_inputs import load_scene_inputs
from src.validation.artifacts import artifact_hashes, verify_snapshot, write_json
from src.validation.goods_predictions import export_predictions


def run_arm(
    references,
    references_sha256,
    task,
    configuration,
    checkpoint,
    output,
    *,
    synthetic=None,
    synthetic_sha256=None,
    device="cuda",
    fixture=False,
):
    """Run one frozen arm; a scene arm cannot start with pending admission.

    Every seed uses the same source-slot/update budget. Final-epoch selection
    avoids earlier development forwards. No candidate is automatically promoted.
    """
    references, task, configuration = map(Path, (references, task, configuration))
    config = json.loads(configuration.read_text(encoding="utf-8"))
    validate_configuration(config, fixture=fixture)
    if config.get("checkpoint_policy") != "final_epoch" or config[
        "experiment_kind"
    ] not in {"real_only", "real_plus_scene"}:
        raise ValueError(
            "Scene comparison requires an explicit fixed-final supported arm"
        )
    if not fixture and config["model_seeds"] != [17, 29, 43]:
        raise ValueError("Customer comparison requires all three declared seeds")
    rows = load_references(references, references_sha256, task, fixture=fixture)
    scene_arm = config["experiment_kind"] == "real_plus_scene"
    if scene_arm:
        if synthetic is None or synthetic_sha256 is None:
            raise ValueError("Scene arm requires completed admission and its checksum")
        load_scene_inputs(
            synthetic,
            synthetic_sha256,
            references,
            references_sha256,
            rows,
            config,
            fixture=fixture,
        )
    elif synthetic is not None or synthetic_sha256 is not None:
        raise ValueError("Real-only arm cannot receive synthetic inputs")
    if not fixture and (
        checkpoint is None
        or file_hash(Path(checkpoint)) != config["initial_checkpoint_sha256"]
    ):
        raise ValueError("Exact local initialization checkpoint is required")
    configuration_sha256 = file_hash(configuration)
    output = reserve_output(
        output,
        tuple(
            Path(value)
            for value in (
                references,
                task,
                configuration,
                synthetic or references,
                checkpoint or references,
                workspace_root() / "data/raw",
            )
        ),
    )
    try:
        shutil.copyfile(configuration, output / "configuration.json")
        shutil.copyfile(task, output / "task.json")
        results = []
        for seed in config["model_seeds"]:
            folder = output / f"seed_{seed}"
            trained = train_run(
                references,
                references_sha256,
                task,
                configuration,
                checkpoint,
                folder,
                seed,
                device,
                fixture=fixture,
                synthetic=synthetic,
                synthetic_sha=synthetic_sha256,
            )
            run_sha = file_hash(folder / "profile.json")
            evaluate_run(
                folder,
                run_sha,
                references,
                references_sha256,
                output / f"evaluation_{seed}",
                device,
                fixture=fixture,
            )
            if not fixture:
                export_predictions(
                    folder,
                    run_sha,
                    references,
                    references_sha256,
                    output / f"predictions_{seed}",
                    device,
                )
            metrics = json.loads(
                (folder / "validation_metrics.json").read_text(encoding="utf-8")
            )
            results.append(
                dict(
                    seed=seed,
                    metrics=metrics,
                    optimizer_updates=trained["optimizer_updates"],
                    presentations=trained["total_training_presentations"],
                    real_presentations=trained["real_presentations"],
                    scene_presentations=trained["synthetic_presentations"],
                    run_profile_sha256=run_sha,
                    checkpoint_sha256=trained["best_checkpoint_sha256"],
                    **(
                        {
                            "primary_real_presentations": trained[
                                "primary_real_presentations"
                            ],
                            "auxiliary_presentations": trained[
                                "auxiliary_presentations"
                            ],
                            **(
                                {
                                    "auxiliary_target_presentations": trained[
                                        "auxiliary_target_presentations"
                                    ]
                                }
                                if "auxiliary_target_presentations" in trained
                                else {}
                            ),
                        }
                        if "primary_real_presentations" in trained
                        else {}
                    ),
                )
            )
        if file_hash(configuration) != configuration_sha256:
            raise ValueError("Frozen comparison configuration changed during training")
        write_json(output / "seed_results.json", results)
        report = dict(
            status="passed",
            purpose="known_fixture_only" if fixture else "fixed_final_goods_arm",
            experiment_kind=config["experiment_kind"],
            reference_profile_sha256=references_sha256,
            protocol_sha256=file_hash(task),
            configuration_sha256=configuration_sha256,
            synthetic_profile_sha256=synthetic_sha256,
            model_seeds=config["model_seeds"],
            checkpoint_policy="final_epoch",
            held_out_test_evaluated=False,
            candidate_promoted=False,
            matched_comparison_complete=False,
            artifact_sha256=artifact_hashes(output),
        )
        write_json(output / "profile.json", report)
        return report
    except Exception as error:
        write_json(
            output / "profile.json",
            dict(status="failed", candidate_promoted=False, error=str(error)),
        )
        raise


def recount_predictions(root, references, rows, recorded):
    """Recount saved native predictions with an independent joint histogram."""
    root = Path(root)
    predictions = [
        json.loads(line)
        for line in (root / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected = {row["sample_id"]: row for row in rows if row["split"] == "validation"}
    selected = [row for row in predictions if row["split"] == "validation"]
    if len(selected) != len(expected) or {row["sample_id"] for row in selected} != set(
        expected
    ):
        raise ValueError(
            "Prediction membership differs from the frozen development cohort"
        )
    totals = np.zeros(4, dtype=np.int64)
    for prediction in selected:
        row = expected[prediction["sample_id"]]
        path = source_path(root, prediction["prediction_path"])
        if (
            file_hash(path) != prediction["prediction_sha256"]
            or prediction["target_sha256"] != row["target_sha256"]
        ):
            raise ValueError("Saved prediction or target binding changed")
        with Image.open(path) as image:
            values = np.array(image)
        with Image.open(source_path(references, row["target_path"])) as image:
            target = np.array(image)
        if (
            values.shape != target.shape
            or values.dtype != np.uint8
            or not np.isin(values, [0, 1]).all()
        ):
            raise ValueError(
                "Saved prediction has invalid native shape or categorical values"
            )
        valid = target != 255
        totals += np.bincount(
            (target[valid].astype(np.int64) * 2 + values[valid]).ravel(), minlength=4
        )
    tn, fp, fn, tp = map(int, totals)
    counts = dict(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        valid_pixels=int(totals.sum()),
    )
    if counts != recorded["confusion"]:
        raise ValueError("Independent native prediction confusion counts disagree")
    return counts


def compare_arms(
    control,
    control_sha256,
    scene,
    scene_sha256,
    references,
    references_sha256,
    output,
    *,
    fixture=False,
):
    """Save paired seed deltas and regressions without changing the retained model."""
    control, scene, references = map(Path, (control, scene, references))
    profiles = [
        verify_snapshot(root, sha)
        for root, sha in ((control, control_sha256), (scene, scene_sha256))
    ]
    expected_purpose = "known_fixture_only" if fixture else "fixed_final_goods_arm"
    if any(
        profile.get("status") != "passed"
        or profile.get("purpose") != expected_purpose
        or profile.get("reference_profile_sha256") != references_sha256
        for profile in profiles
    ):
        raise ValueError("Comparison requires completed arms on one exact real cohort")
    configs = [
        json.loads((root / "configuration.json").read_text(encoding="utf-8"))
        for root in (control, scene)
    ]
    for config in configs:
        validate_configuration(config, fixture=fixture)
    presence_policy = {"strategy": "balanced_presence_cycle"}
    if any(
        config.get("auxiliary_sampling") == presence_policy for config in configs
    ) and not all(
        config.get("auxiliary_sampling") == presence_policy for config in configs
    ):
        raise ValueError(
            "Matched negative/mixed exposure requires the same presence cycle in both arms"
        )
    permitted_differences = {
        "experiment_kind",
        "synthetic_mixing",
        "experiment_name",
        "hypothesis",
        "comparison",
        "budget_matching",
        "auxiliary_sampling",
    }
    for field in set(configs[0]) | set(configs[1]):
        # Normalize only redundant selection metadata. The frozen control files,
        # weights, data bindings and every actual training parameter stay intact.
        if field == "selection_metric" and all(
            config.get("checkpoint_policy") == "final_epoch" for config in configs
        ):
            if checkpoint_selection_metric(configs[0]) != checkpoint_selection_metric(
                configs[1]
            ):
                raise ValueError("Unmatched effective checkpoint selection")
            continue
        if field not in permitted_differences and configs[0].get(field) != configs[
            1
        ].get(field):
            raise ValueError("Unmatched comparison configuration: " + field)
    if (
        configs[0]["experiment_kind"] != "real_only"
        or configs[1]["experiment_kind"] != "real_plus_scene"
        or any(config.get("checkpoint_policy") != "final_epoch" for config in configs)
        or profiles[0]["protocol_sha256"] != profiles[1]["protocol_sha256"]
    ):
        raise ValueError("Expected matched fixed-final real-only and scene arms")
    rows = load_references(
        references, references_sha256, control / "task.json", fixture=fixture
    )
    arm_results = [
        json.loads((root / "seed_results.json").read_text(encoding="utf-8"))
        for root in (control, scene)
    ]
    if any(
        [row["seed"] for row in arm] != configs[0]["model_seeds"] for arm in arm_results
    ):
        raise ValueError("Comparison seed membership mismatch")
    deltas, recounts = [], []
    for real, synthetic in zip(*arm_results, strict=True):
        if (
            real["optimizer_updates"] != synthetic["optimizer_updates"]
            or real["presentations"] != synthetic["presentations"]
        ):
            raise ValueError(
                "Scene comparison changed its optimizer/presentation budget"
            )
        if configs[0].get("auxiliary_training") is not None and (
            real.get("primary_real_presentations")
            != synthetic.get("primary_real_presentations")
            or real.get("auxiliary_presentations")
            != synthetic.get("auxiliary_presentations")
            or not real.get("primary_real_presentations")
        ):
            raise ValueError("Auxiliary comparison changed primary real exposure")
        if configs[0].get("auxiliary_sampling") == {
            "strategy": "balanced_presence_cycle"
        }:
            for result in (real, synthetic):
                count = result["auxiliary_presentations"]
                expected = {"background_only": (count + 1) // 2, "mixed": count // 2}
                measured = result.get("auxiliary_target_presentations", {})
                if set(measured) - set(expected) or any(
                    measured.get(key, 0) != value for key, value in expected.items()
                ):
                    raise ValueError(
                        "Executed negative/mixed exposure differs from its matched schedule"
                    )
        if real["scene_presentations"] or not synthetic["scene_presentations"]:
            raise ValueError(
                "Comparison did not exercise distinct real-only and scene arms"
            )
        if not fixture:
            for root, result in ((control, real), (scene, synthetic)):
                recounts.append(
                    dict(
                        arm=root.name,
                        seed=result["seed"],
                        confusion=recount_predictions(
                            root / f'predictions_{result["seed"]}',
                            references,
                            rows,
                            result["metrics"],
                        ),
                    )
                )
        metrics = ("goods_iou", "precision", "recall")
        if any(
            real["metrics"][key] is None or synthetic["metrics"][key] is None
            for key in metrics
        ):
            raise ValueError("Required comparison metric is undefined")
        deltas.append(
            dict(
                seed=real["seed"],
                **{
                    key: synthetic["metrics"][key] - real["metrics"][key]
                    for key in metrics
                },
            )
        )
    means = [
        {
            key: float(np.mean([row["metrics"][key] for row in arm]))
            for key in ("goods_iou", "precision", "recall")
        }
        for arm in arm_results
    ]
    changes = {key: means[1][key] - means[0][key] for key in means[0]}
    passes = (
        changes["goods_iou"] > 0
        and changes["precision"] >= 0
        and changes["recall"] >= 0
    )
    output = reserve_output(
        output, (control, scene, references, workspace_root() / "data/raw")
    )
    write_json(output / "paired_deltas.json", deltas)
    write_json(output / "native_recounts.json", recounts)
    report = dict(
        status="passed",
        purpose="known_fixture_only" if fixture else "scene_utility_comparison",
        matched_comparison_complete=True,
        scene_utility_measured=not fixture,
        held_out_test_evaluated=False,
        control_profile_sha256=control_sha256,
        scene_profile_sha256=scene_sha256,
        donor_sampling={
            "control": configs[0].get(
                "auxiliary_sampling", {"strategy": "independent_hash"}
            ),
            "scene": configs[1].get(
                "auxiliary_sampling", {"strategy": "independent_hash"}
            ),
        },
        reference_profile_sha256=references_sha256,
        model_seeds=configs[0]["model_seeds"],
        real_only=means[0],
        real_plus_scene=means[1],
        mean_deltas=changes,
        acceptance_rule="Mean goods IoU improves with no mean precision or recall regression",
        quality_rule_passed=passes,
        candidate_promoted=False,
        native_predictions_independently_recounted=(
            0
            if fixture
            else len([row for row in rows if row["split"] == "validation"])
            * len(deltas)
            * 2
        ),
        limitations="One authored scene and one train/development trajectory; dependent frames and repeated development use. Three seeds do not establish statistical significance or held-out generalization.",
        artifact_sha256=artifact_hashes(output),
    )
    write_json(output / "profile.json", report)
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    for name in ("references", "task", "configuration", "checkpoint", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--references-sha256", required=True)
    run.add_argument("--synthetic", type=Path)
    run.add_argument("--synthetic-sha256")
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    compare = commands.add_parser("compare")
    for name in ("control", "scene", "references", "output"):
        compare.add_argument("--" + name, type=Path, required=True)
    for name in ("control", "scene", "references"):
        compare.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_arm(
            args.references,
            args.references_sha256,
            args.task,
            args.configuration,
            args.checkpoint,
            args.output,
            synthetic=args.synthetic,
            synthetic_sha256=args.synthetic_sha256,
            device=args.device,
        )
    else:
        result = compare_arms(
            args.control,
            args.control_sha256,
            args.scene,
            args.scene_sha256,
            args.references,
            args.references_sha256,
            args.output,
        )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "artifact_sha256"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
