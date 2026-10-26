"""Audit saved segmentation errors and executed training exposure without fitting.

Coverage statistics use training inputs only. Development predictions are inspected
separately at native resolution, excluding ignored targets. Luminance strata are
fixed display-intensity ranges; they are not object labels or sensor calibration.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.training.goods_benchmark import load_references
from src.training.scene_inputs import load_scene_inputs
from src.validation.artifacts import artifact_hashes, verify_snapshot, write_json
from src.validation.scene_admission import require_snapshot_digest

LUMA_EDGES = (0, 50, 100, 150, 200, 256)
CLASSES = {"background": 0, "goods": 1, "ignored": 255}


def validate_pair(rgb, target):
    """Require aligned uint8 RGB (H, W, 3) and categorical targets (H, W)."""
    if (
        not isinstance(rgb, np.ndarray)
        or not isinstance(target, np.ndarray)
        or target.ndim != 2
        or not target.size
        or rgb.shape != (*target.shape, 3)
        or rgb.dtype != np.uint8
        or target.dtype != np.uint8
        or not np.isin(target, list(CLASSES.values())).all()
    ):
        raise ValueError("Expected nonempty aligned uint8 RGB and 0/1/255 targets")


def image_profile(rgb, target):
    """Count target classes and luminance bins; never fill ignored pixels.

    RGB and targets must already use the intended resolution. Luminance follows
    Pillow's RGB-to-L conversion. Histogram bins are half-open 8-bit intervals.
    """
    validate_pair(rgb, target)
    luma = np.array(Image.fromarray(rgb).convert("L"))
    counts = {name: int((target == value).sum()) for name, value in CLASSES.items()}
    histograms = {
        name: np.histogram(luma[target == value], bins=LUMA_EDGES)[0].tolist()
        for name, value in CLASSES.items()
    }
    return dict(
        pixels=int(target.size),
        class_pixels=counts,
        luma_histograms=histograms,
        foreground_fraction_of_image=counts["goods"] / target.size,
        dark_goods_fraction=(
            histograms["goods"][0] / counts["goods"] if counts["goods"] else None
        ),
    )


def rates(counts):
    """Return FPR and FNR with explicit nulls for absent reference classes."""
    tn, fp, fn, tp = (counts[key] for key in ("tn", "fp", "fn", "tp"))
    return dict(
        false_positive_rate=fp / (tn + fp) if tn + fp else None,
        false_negative_rate=fn / (tp + fn) if tp + fn else None,
    )


def luma_confusion(rgb, target, prediction):
    """Partition native scorable errors into fixed luminance strata.

    Predictions must be uint8 binary IDs with the native target shape. Ignored
    pixels contribute to no confusion count, including at intensity boundaries.
    """
    validate_pair(rgb, target)
    if (
        not isinstance(prediction, np.ndarray)
        or prediction.shape != target.shape
        or prediction.dtype != np.uint8
        or not np.isin(prediction, [0, 1]).all()
    ):
        raise ValueError("Predictions must be aligned uint8 binary class IDs")
    luma = np.array(Image.fromarray(rgb).convert("L"))
    valid = target != 255
    strata = []
    for low, high in zip(LUMA_EDGES[:-1], LUMA_EDGES[1:]):
        selected = valid & (luma >= low) & (luma < high)
        histogram = np.bincount(
            target[selected].astype(np.int64) * 2 + prediction[selected], minlength=4
        )
        counts = dict(zip(("tn", "fp", "fn", "tp"), map(int, histogram)))
        strata.append(dict(low=low, high=high, **counts, **rates(counts)))
    total = {key: sum(row[key] for row in strata) for key in ("tn", "fp", "fn", "tp")}
    return dict(
        **total, **rates(total), strata=strata, ignored_pixels=int((~valid).sum())
    )


def weighted_exposure(profiles, schedule):
    """Sum class and intensity counts over actual integer input presentations.

    This measures pixel exposure, not gradient contribution. The auxiliary loss
    averages classes separately and cannot be inferred from raw pixel counts.
    """
    if not schedule or any(
        name not in profiles or type(count) is not int or count < 1
        for name, count in schedule.items()
    ):
        raise ValueError("Exposure requires known inputs and positive integer counts")
    counts = {
        label: sum(
            profiles[name]["class_pixels"][label] * count
            for name, count in schedule.items()
        )
        for label in CLASSES
    }
    histograms = {
        label: [
            sum(
                profiles[name]["luma_histograms"][label][index] * count
                for name, count in schedule.items()
            )
            for index in range(len(LUMA_EDGES) - 1)
        ]
        for label in CLASSES
    }
    return dict(
        presentations=sum(schedule.values()),
        class_pixels=counts,
        luma_histograms=histograms,
        dark_goods_fraction=(
            histograms["goods"][0] / counts["goods"] if counts["goods"] else None
        ),
        foreground_fraction_of_supervised=(
            counts["goods"] / (counts["goods"] + counts["background"])
            if counts["goods"] + counts["background"]
            else None
        ),
    )


def read_pair(root, row, resolution=None):
    """Read a bound RGB/target pair, using the declared training resamplers only."""
    for key, digest in (
        ("rgb_path", "source_rgb_sha256"),
        ("target_path", "target_sha256"),
    ):
        if file_hash(source_path(root, row[key])) != row[digest]:
            raise ValueError("Bound RGB/target payload changed")
    with Image.open(source_path(root, row["rgb_path"])) as image:
        rgb = image.copy()
    with Image.open(source_path(root, row["target_path"])) as image:
        target = image.copy()
    if resolution is not None:
        rgb = rgb.resize(tuple(resolution), Image.Resampling.BOX)
        target = target.resize(tuple(resolution), Image.Resampling.NEAREST)
    return np.array(rgb), np.array(target)


def scene_family_profile(rgb, target, instances, objects):
    """Count visible goods pixels by declared mesh asset, including background stock.

    Renderer instance roles define these families. They are not transferred to
    real pixels, whose binary references carry no object-family annotation.
    """
    validate_pair(rgb, target)
    if instances.shape != target.shape or instances.dtype != np.uint16:
        raise ValueError("Instance IDs must be aligned uint16 labels")
    known = {obj["instance_id"]: obj for obj in objects}
    visible_goods = set(map(int, np.unique(instances[target == 1])))
    if not visible_goods <= known.keys():
        raise ValueError("A visible goods instance has no source object")
    luma = np.array(Image.fromarray(rgb).convert("L"))
    families = {}
    for identifier in visible_goods:
        obj = known[identifier]
        if obj["role"] != "goods" or obj["class_id"] != 1:
            raise ValueError("Goods pixels conflict with declared instance roles")
        asset = obj.get("mesh", {}).get("asset_id", "authored_goods_geometry")
        pixels = (target == 1) & (instances == identifier)
        histogram = np.histogram(luma[pixels], bins=LUMA_EDGES)[0]
        record = families.setdefault(asset, dict(goods_pixels=0, dark_goods_pixels=0))
        record["goods_pixels"] += int(pixels.sum())
        record["dark_goods_pixels"] += int(histogram[0])
    if sum(row["goods_pixels"] for row in families.values()) != int(
        (target == 1).sum()
    ):
        raise ValueError("Source-family partition lost foreground pixels")
    return families


def run_audit(
    references,
    reference_sha,
    task,
    control,
    control_sha,
    scene,
    scene_sha,
    admission,
    admission_sha,
    output,
):
    """Write a fresh, source-bound audit without loading a model or raw test data."""
    references, task, control, scene, admission = map(
        Path, (references, task, control, scene, admission)
    )
    roots = {
        "references": (references, reference_sha),
        "control": (control, control_sha),
        "scene": (scene, scene_sha),
        "admission": (admission, admission_sha),
    }
    for _, digest in roots.values():
        require_snapshot_digest(digest)
    snapshots = {
        name: verify_snapshot(root, digest) for name, (root, digest) in roots.items()
    }
    for name, kind in (("control", "real_only"), ("scene", "real_plus_scene")):
        profile = snapshots[name]
        if (
            profile.get("status") != "passed"
            or profile.get("purpose") != "fixed_final_goods_arm"
            or profile.get("experiment_kind") != kind
            or profile.get("checkpoint_policy") != "final_epoch"
            or profile.get("reference_profile_sha256") != reference_sha
        ):
            raise ValueError(
                "Completed fixed-final arms on the same real cohort are required"
            )
    if snapshots["scene"].get("synthetic_profile_sha256") != admission_sha:
        raise ValueError("The scene run used a different admitted batch")
    rows = load_references(references, reference_sha, task)
    config = json.loads((scene / "configuration.json").read_text(encoding="utf-8"))
    scenes = load_scene_inputs(
        admission, admission_sha, references, reference_sha, rows, config
    )
    job = json.loads((admission / "dataset/generation_job.json").read_text())
    captures = {capture["sample_id"]: capture for capture in job["scenes"]}
    profiles, families = {}, {}
    training = [row for row in rows if row["split"] == "train"]
    for root, members in ((references, training), (admission, scenes)):
        for row in members:
            rgb, target = read_pair(root, row, config["training_resolution"])
            profiles[row["sample_id"]] = image_profile(rgb, target)
            if root == admission:
                capture = captures[row["scene_sample_id"]]
                with Image.open(
                    admission / "dataset" / row["scene_sample_id"] / "instance_ids.png"
                ) as image:
                    instances = np.array(
                        image.resize(
                            tuple(config["training_resolution"]),
                            Image.Resampling.NEAREST,
                        )
                    )
                families[row["sample_id"]] = scene_family_profile(
                    rgb, target, instances, capture["objects"]
                )

    exposure, errors = [], []
    development = {
        row["sample_id"]: row for row in rows if row["split"] == "validation"
    }
    # Cache twelve native development pairs for diagnosis. These arrays never
    # enter the training-coverage profile or generation-conditioning statistics.
    native = {
        identity: read_pair(references, row) for identity, row in development.items()
    }
    for arm, root in (("real_replay", control), ("real_plus_scene", scene)):
        seeds = json.loads((root / "seed_results.json").read_text())
        if [row["seed"] for row in seeds] != config["model_seeds"]:
            raise ValueError(
                "Arm seed membership differs from its frozen configuration"
            )
        for result in seeds:
            seed = result["seed"]
            run = root / f"seed_{seed}"
            run_profile = verify_snapshot(run, result["run_profile_sha256"])
            if run_profile["reference_profile_sha256"] != reference_sha:
                raise ValueError("Run uses a different reference cohort")
            primary = Counter(json.loads((run / "sampling.json").read_text()))
            auxiliary = Counter(
                json.loads((run / "auxiliary_sampling.json").read_text())
            )
            actual = Counter(json.loads((run / "input_sampling.json").read_text()))
            if actual != primary + auxiliary or set(primary) != {
                row["sample_id"] for row in training
            }:
                raise ValueError(
                    "Executed inputs do not preserve the primary training cohort"
                )
            if (
                set(primary.values()) != {config["epochs"]}
                or sum(auxiliary.values()) != 60
            ):
                raise ValueError(
                    "Executed presentation budgets differ from the fixed protocol"
                )
            if (
                run_profile["primary_real_presentations"] != sum(primary.values())
                or run_profile["auxiliary_presentations"] != sum(auxiliary.values())
                or run_profile["total_training_presentations"] != sum(actual.values())
                or run_profile["optimizer_updates"] != 120
            ):
                raise ValueError(
                    "Recorded exposure/update counts differ from executed inputs"
                )
            if arm == "real_plus_scene" and set(auxiliary) - set(families):
                raise ValueError("Scene donors are not members of the admitted batch")
            if arm == "real_replay" and set(auxiliary) - set(primary):
                raise ValueError("Replay donors are not training references")
            family_exposure = {}
            for identity, count in auxiliary.items():
                for asset, values in families.get(identity, {}).items():
                    aggregate = family_exposure.setdefault(
                        asset, dict(goods_pixels=0, dark_goods_pixels=0)
                    )
                    for key in aggregate:
                        aggregate[key] += values[key] * count
            exposure.append(
                dict(
                    arm=arm,
                    seed=seed,
                    primary=weighted_exposure(profiles, primary),
                    auxiliary=weighted_exposure(profiles, auxiliary),
                    auxiliary_sampling=dict(auxiliary),
                    visible_scene_family_exposure=family_exposure,
                )
            )
            prediction_root = root / f"predictions_{seed}"
            verify_snapshot(
                prediction_root, file_hash(prediction_root / "profile.json")
            )
            predictions = [
                json.loads(line)
                for line in (prediction_root / "predictions.jsonl")
                .read_text()
                .splitlines()
            ]
            selected = [row for row in predictions if row["split"] == "validation"]
            if len(selected) != len(development) or {
                row["sample_id"] for row in selected
            } != set(development):
                raise ValueError(
                    "Native prediction membership differs from the development cohort"
                )
            totals = Counter()
            for row in selected:
                identity = row["sample_id"]
                path = source_path(prediction_root, row["prediction_path"])
                if (
                    file_hash(path) != row["prediction_sha256"]
                    or row["target_sha256"] != development[identity]["target_sha256"]
                ):
                    raise ValueError("Saved prediction or target binding changed")
                with Image.open(path) as image:
                    prediction = np.array(image)
                measured = luma_confusion(*native[identity], prediction)
                totals.update({key: measured[key] for key in ("tn", "fp", "fn", "tp")})
                errors.append(dict(arm=arm, seed=seed, sample_id=identity, **measured))
            expected = result["metrics"]["confusion"]
            if any(
                totals[short] != expected[long]
                for short, long in (
                    ("tn", "true_negative"),
                    ("fp", "false_positive"),
                    ("fn", "false_negative"),
                    ("tp", "true_positive"),
                )
            ):
                raise ValueError(
                    "Luminance partitions do not reproduce the saved confusion counts"
                )
    # Recheck every immutable source after the audit to detect concurrent edits.
    for root, digest in roots.values():
        verify_snapshot(root, digest)
    output = reserve_output(
        output,
        tuple(root for root, _ in roots.values())
        + (Path(__file__).parents[2] / "data/raw",),
    )
    write_json(output / "training_profiles.json", profiles)
    write_json(output / "rendered_family_profiles.json", families)
    write_json(output / "executed_exposure.json", exposure)
    write_json(output / "development_error_strata.json", errors)
    summary = dict(
        status="passed",
        purpose="post_experiment_failure_analysis",
        source_profiles={name: digest for name, (_, digest) in roots.items()},
        training_references_profiled=len(training),
        admitted_scenes_profiled=len(scenes),
        native_development_predictions_partitioned=len(errors),
        luma_edges=list(LUMA_EDGES),
        training_resolution=config["training_resolution"],
        development_resolution=config["native_reference_resolution"],
        conditioning_statistics_split="train",
        development_use="post-experiment error diagnosis only",
        reference_labels_changed=False,
        model_loaded=False,
        training_performed=False,
        held_out_test_evaluated=False,
        limitations="Display-intensity strata and binary reference counts do not identify object families in real images or establish causes of model changes. Auxiliary pixel exposure is not gradient contribution.",
        artifact_sha256=artifact_hashes(output),
    )
    write_json(output / "profile.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("references", "task", "control", "scene", "admission", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("references", "control", "scene", "admission"):
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    result = run_audit(
        args.references,
        args.references_sha256,
        args.task,
        args.control,
        args.control_sha256,
        args.scene,
        args.scene_sha256,
        args.admission,
        args.admission_sha256,
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
