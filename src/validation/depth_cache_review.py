"""Verify depth caches against native June sources and reproduce fixed baselines.

Cached depth is uint16 millimetres; model depth is float32 metres. This audit
does not retrain, evaluate October sensors, or certify unknown-depth predictions.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.profiling.task_dataset_review import evaluation_records
from src.training import depth_baseline, depth_context, depth_quality
from src.training.runtime import PROJECT_ROOT, require_cached_ids, require_output

# These are two historical cache formats, not interchangeable preparation modes.
# The legacy RGB mode was absent from its profile and is verified for every frame.
RGB_RESAMPLING = {160: Image.Resampling.BILINEAR, 320: Image.Resampling.BOX}


def file_hash(path):
    """Hash file bytes in bounded blocks, including large lossless NPZ caches."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def validate_cache_arrays(rgb, depth_mm, sample_ids, expected_ids, resolution, split):
    """Reject missing, reordered, duplicate or cross-split frames and bad payloads.

    RGB must be uint8 NHWC. Depth must be uint16 NHW millimetres; (0, 20000]
    is the valid supervised range. No interpolation or repairs occur here.
    """
    ids = require_cached_ids(sample_ids, split)
    if ids != expected_ids:
        raise ValueError("Cache membership/order differs from the ingestion manifest")
    width, height = resolution
    if rgb.dtype != np.uint8 or rgb.shape != (len(ids), height, width, 3):
        raise ValueError("Incorrect RGB cache shape or dtype")
    if depth_mm.dtype != np.uint16 or depth_mm.shape != (len(ids), height, width):
        raise ValueError("Incorrect millimetre depth cache shape or dtype")
    if np.any(depth_mm > 20000):
        raise ValueError("Unexpected out-of-range cached depths; do not silently clip")
    return ids


def compare_metrics(actual, expected, tolerance=1e-7):
    """Require identical metric regions/counts and numerical reproduction in metres.

    The absolute tolerance allows GPU reduction rounding, not changed masks,
    seeds, sources or scenario settings. Frame/pixel counts must match exactly.
    """
    deltas = {}
    for key, historical in expected.items():
        current = actual[key]
        differences = {}
        for metric, value in historical.items():
            if value is None:
                if current[metric] is not None:
                    raise ValueError(f"Changed empty metric region: {key}/{metric}")
            elif metric in ("pixels", "evaluated_pixels"):
                if current[metric] != value:
                    raise ValueError(f"Changed evaluated pixel count: {key}")
            else:
                differences[metric] = abs(current[metric] - value)
                if differences[metric] > tolerance:
                    raise ValueError(
                        f"Historical metric not reproduced: {key}/{metric}"
                    )
        deltas[key] = differences
    return deltas


class NearestOnly(nn.Module):
    """Return the existing nearest input channel without learned parameters."""

    def forward(self, inputs):
        return inputs[:, 5:6]


def audit_caches(project, raw_root, manifest, output):
    """Decode each June RGB/depth source once and compare both cache resolutions."""
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    groups = evaluation_records(rows)
    expected_counts = {"train": 1136, "validation": 1037, "test": 915}
    if {split: len(items) for split, items in groups.items()} != expected_counts:
        raise ValueError("Manifest does not contain the complete expected trajectories")
    if any(row["depth_scale_to_metres"] != 0.001 for row in rows):
        raise ValueError("Unexpected measured-depth unit scale")
    manifest_hash = file_hash(manifest)
    provenance_path = project / "data/caches/depth_320/source_provenance.jsonl"
    provenance_records = [
        json.loads(line) for line in provenance_path.read_text().splitlines() if line
    ]
    if len({r["sample_id"] for r in provenance_records}) != len(provenance_records):
        raise ValueError("Duplicate source-provenance records")
    provenance = {r["sample_id"]: r for r in provenance_records}
    expected_source_ids = {r["sample_id"] for r in rows if r["split"] != "test"}
    if set(provenance) != expected_source_ids:
        raise ValueError("Source provenance does not cover exactly the June caches")
    summaries = {}
    profiles = {}
    for width, height in ((160, 90), (320, 180)):
        directory = project / f"data/caches/depth_{width}"
        profile = read_json(directory / "profile.json")
        if profile["manifest_sha256"] != manifest_hash:
            raise ValueError("Cache profile refers to another ingestion manifest")
        if profile["resolution"] != [width, height]:
            raise ValueError("Cache profile has an incorrect resolution")
        profiles[width] = profile
    with (output / "source_verification.jsonl").open("w", encoding="utf-8") as evidence:
        for split in ("train", "validation"):
            selected = sorted(
                (row for row in rows if row["split"] == split),
                key=lambda row: int(row["frame_id"]),
            )
            expected_ids = [row["sample_id"] for row in selected]
            caches = {}
            for width, height in ((160, 90), (320, 180)):
                path = project / f"data/caches/depth_{width}/{split}.npz"
                with np.load(path, allow_pickle=False) as payload:
                    if set(payload.files) != {"rgb", "depth_mm", "sample_ids"}:
                        raise ValueError("Unexpected cache array schema")
                    rgb, depth_mm = payload["rgb"], payload["depth_mm"]
                    validate_cache_arrays(
                        rgb,
                        depth_mm,
                        payload["sample_ids"].tolist(),
                        expected_ids,
                        (width, height),
                        split,
                    )
                cache_hash = file_hash(path)
                historical_hash = profiles[width].get("cache_sha256", {}).get(split)
                if historical_hash and historical_hash != cache_hash:
                    raise ValueError("Cached bytes differ from recorded cache hash")
                caches[width] = (rgb, depth_mm)
                valid = (depth_mm > 0) & (depth_mm <= 20000)
                summaries[f"depth_{width}/{split}"] = {
                    "frames": len(expected_ids),
                    "resolution": [width, height],
                    "sha256": cache_hash,
                    "depth_storage_units": "millimetres",
                    "depth_storage_dtype": "uint16",
                    "depth_scale_to_metres": 0.001,
                    "valid_fraction": float(valid.mean()),
                    "invalid_zero_pixels": int((depth_mm == 0).sum()),
                    "pixels_over_20m": int((depth_mm > 20000).sum()),
                    "original_cache_hash_available": historical_hash is not None,
                    "rgb_resampling": RGB_RESAMPLING[width].name,
                    "depth_resampling": "NEAREST",
                }
            native_valid, native_pixels = 0, 0
            for index, row in enumerate(selected):
                payloads = []
                source_hashes = {}
                record = provenance[row["sample_id"]]
                if record["split"] != split:
                    raise ValueError("Wrong split in source provenance")
                for modality in ("rgb", "depth"):
                    key = modality + "_path"
                    source = (raw_root / row[key]).resolve()
                    if not source.is_relative_to(raw_root):
                        raise ValueError("Manifest source escapes the raw root")
                    if row[key] != record[key]:
                        raise ValueError("Source provenance path differs from manifest")
                    payload = source.read_bytes()
                    digest = hashlib.sha256(payload).hexdigest()
                    if digest != record[modality + "_sha256"]:
                        raise ValueError(f"Raw source hash changed: {row['sample_id']}")
                    source_hashes[modality + "_sha256"] = digest
                    payloads.append(payload)
                with Image.open(io.BytesIO(payloads[0])) as rgb_image, Image.open(
                    io.BytesIO(payloads[1])
                ) as depth_image:
                    native_rgb = np.array(rgb_image)
                    native_depth = np.array(depth_image)
                    if native_rgb.dtype != np.uint8 or native_rgb.shape != (
                        720,
                        1280,
                        3,
                    ):
                        raise ValueError("Unexpected native RGB shape/dtype")
                    if native_depth.dtype != np.uint16 or native_depth.shape != (
                        720,
                        1280,
                    ):
                        raise ValueError("Unexpected native depth shape/dtype")
                    native_valid += int(
                        ((native_depth > 0) & (native_depth <= 20000)).sum()
                    )
                    native_pixels += native_depth.size
                    for width, height in ((160, 90), (320, 180)):
                        expected_rgb = np.array(
                            rgb_image.resize((width, height), RGB_RESAMPLING[width])
                        )
                        expected_depth = np.array(
                            depth_image.resize(
                                (width, height), Image.Resampling.NEAREST
                            )
                        )
                        if not np.array_equal(caches[width][0][index], expected_rgb):
                            raise ValueError(
                                f"Cached RGB differs from native {RGB_RESAMPLING[width].name} derivative ({width}px): {row['sample_id']}"
                            )
                        if not np.array_equal(caches[width][1][index], expected_depth):
                            raise ValueError(
                                f"Cached depth differs from native NEAREST derivative: {row['sample_id']}"
                            )
                evidence.write(
                    json.dumps(
                        {
                            "sample_id": row["sample_id"],
                            "split": split,
                            **source_hashes,
                            "both_cache_derivatives": "bit_exact",
                        }
                    )
                    + "\n"
                )
                if (index + 1) % 100 == 0:
                    print(
                        f"Cache sources: {split} {index+1}/{len(selected)}", flush=True
                    )
            for profile in profiles.values():
                if (
                    abs(
                        profile["splits"][split]["native_valid_depth_fraction"]
                        - native_valid / native_pixels
                    )
                    > 1e-12
                ):
                    raise ValueError(
                        "Native valid-depth statistics no longer match the profile"
                    )
            del caches
    summary = {
        "status": "passed",
        "manifest_sha256": manifest_hash,
        "manifest_split_counts": expected_counts,
        "cached_unique_frames": 2173,
        "native_source_files_verified": 4346,
        "cached_frame_derivatives_verified": 4346,
        "source_provenance_sha256": file_hash(provenance_path),
        "cache_files": summaries,
        "whole_trajectory_splits": "passed",
        "native_derivative_equality": "passed",
        "raw_source_hashes": "passed",
        "test_sensor_files_opened": 0,
        "notes": [
            "The old 160px profile lacks cache hashes; current hashes and full source equality are now recorded.",
            "profile depth_units refers to model metres; NPZ depth_mm is uint16 millimetres.",
            "Nearest resampling loses native detail without blending depths.",
            "Legacy 160px RGB is BILINEAR, newer 320px RGB is BOX; the legacy profile omitted that distinction.",
        ],
    }
    write_json(output / "cache_integrity.json", summary)
    return summary


class ReviewedQualitySamples(depth_quality.Samples):
    """Validate all scenario masks without altering existing dataset outputs."""

    def __getitem__(self, index):
        result = super().__getitem__(index)
        inputs, target, metric, hidden, distance, noisy = result
        valid = (self.depth[index] > 0) & (self.depth[index] <= 20)
        observed = inputs[4].numpy() > 0
        hidden_mask = hidden[0].numpy()
        metric_mask = metric[0].numpy()
        np.testing.assert_array_equal(observed | hidden_mask, valid)
        if np.any(observed & hidden_mask):
            raise ValueError("Observed and hidden depth overlap")
        np.testing.assert_array_equal(metric_mask, valid if noisy else hidden_mask)
        if np.any(inputs[3].numpy()[~observed] != 0):
            raise ValueError("Sparse input contains unobserved depth")
        if not noisy:
            np.testing.assert_allclose(
                inputs[3].numpy()[observed] * 5,
                self.depth[index][observed],
                atol=1e-6,
                rtol=0,
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("Nonfinite corruption input")
        return result


def reproduce_baselines(project, output):
    """Use the existing preprocessors/evaluators, fixed seeds and historical masks."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to reproduce the original GPU metric reductions"
        )
    torch.set_num_threads(4)
    cache160 = project / "data/caches/depth_160"
    cache320 = project / "data/caches/depth_320"
    results, configurations = {}, {}
    baseline_config = read_json(project / "models/depth_baseline/config.json")
    baseline_data = depth_baseline.Frames(
        cache160 / "validation.npz", baseline_config["validation_dropout_seed"]
    )
    original = depth_baseline.evaluate(
        NearestOnly().cuda(), DataLoader(baseline_data, batch_size=32), "cuda"
    )["nearest"]
    historical = read_json(project / "models/depth_baseline/results.json")[
        "validation"
    ]["nearest"]
    results["random_160"] = {
        "metrics": original,
        "historical_deltas": compare_metrics(
            {"nearest": original}, {"nearest": historical}
        ),
    }
    configurations["random_160"] = baseline_config
    print("Original 160px nearest baseline reproduced", flush=True)
    del baseline_data
    context_config = read_json(project / "models/depth_context/config.json")
    context_data = depth_context.Data(
        cache160 / "validation.npz", context_config["validation_seed"]
    )
    context_metrics, _ = depth_context.evaluate(
        None, DataLoader(context_data, batch_size=32)
    )
    results["blocks_160"] = {
        "metrics": context_metrics,
        "historical_deltas": compare_metrics(
            context_metrics, context_config["baseline_metrics"]
        ),
    }
    configurations["blocks_160"] = context_config
    print("Context 160px nearest/smooth baselines reproduced", flush=True)
    del context_data
    quality_config = read_json(project / "models/depth_quality/config.json")
    historical_results = read_json(project / "models/depth_quality/results.json")
    with np.load(cache320 / "train.npz", allow_pickle=False) as payload:
        donors = payload["depth_mm"]
        donor_ids = require_cached_ids(payload["sample_ids"].tolist(), "train")
    quality_metrics = {}
    donor_records = []
    for scenario in quality_config["scenarios"]:
        data = ReviewedQualitySamples(
            cache320 / "validation.npz",
            donors,
            scenario=scenario,
            seed=quality_config["validation_corruption_seed"],
        )
        metrics = depth_quality.evaluate(
            NearestOnly().cuda(), DataLoader(data, batch_size=32)
        )
        selected = {
            key: value
            for key, value in metrics.items()
            if key.startswith("nearest_") or key.startswith("smooth_")
        }
        historical = {
            key: value
            for key, value in historical_results[0]["scenarios"][scenario].items()
            if key in selected
        }
        if not historical:
            raise ValueError("Historical scenario metric schema not understood")
        quality_metrics[scenario] = {
            "metrics": selected,
            "historical_deltas": compare_metrics(selected, historical),
        }
        # Check donor identities for every validation frame, not just exported previews.
        if scenario not in ("random", "blocks"):
            for index, sample_id in enumerate(data.ids):
                _, _, _, _, metadata = depth_quality.corrupt(
                    data.rgb[index],
                    data.depth[index],
                    donors,
                    index,
                    data.seed,
                    scenario,
                )
                donor_id = donor_ids[metadata["donor_train_index"]]
                donor_records.append(
                    {
                        "sample_id": sample_id,
                        "scenario": scenario,
                        "donor_sample_id": donor_id,
                        **metadata,
                    }
                )
        print(f"Quality 320px baseline reproduced: {scenario}", flush=True)
        del data
    results["quality_320"] = quality_metrics
    configurations["quality_320"] = quality_config
    write_json(output / "historical_configurations.json", configurations)
    write_json(
        output / "baseline_metrics.json",
        {
            "status": "passed",
            "results": results,
            "absolute_tolerance_metres": 1e-7,
            "validation_frames_per_scenario": 1037,
            "aggregation": "pixel-weighted",
            "test_evaluated": False,
        },
    )
    write_json(
        output / "donor_membership.json",
        {
            "status": "passed",
            "checked_queries": len(donor_records),
            "donor_split": "train",
            "full_validation_scenario_mask_checks": len(quality_config["scenarios"])
            * 1037,
            "records": donor_records,
        },
    )
    return results


def export_training_previews(project, output):
    """Export four training-only RGB/depth/valid/hidden/nearest inspection panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth_palette = plt.get_cmap("viridis").copy()
    depth_palette.set_bad("black")

    with np.load(
        project / "data/caches/depth_320/train.npz", allow_pickle=False
    ) as payload:
        rgb, depth_mm, ids = (
            payload["rgb"],
            payload["depth_mm"],
            payload["sample_ids"].tolist(),
        )
    panels = []
    for index in (0, len(ids) // 3, 2 * len(ids) // 3, len(ids) - 1):
        depth = depth_mm[index].astype(np.float32) * 0.001
        valid = (depth > 0) & (depth <= 20)
        inputs, _, hidden, _, metadata = depth_quality.corrupt(
            rgb[index], depth, depth_mm, index, 4242, "blocks"
        )
        observed = inputs[4] > 0
        if (
            np.any(observed & ~valid)
            or np.any(hidden & ~valid)
            or np.any(observed & hidden)
        ):
            raise ValueError("Invalid or overlapping supervision/observation masks")
        np.testing.assert_array_equal(observed | hidden, valid)
        # Actual files retain units and masks; colormaps are display-only.
        directory = output / "training_pairs" / ids[index].replace("/", "_")
        directory.mkdir(parents=True)
        Image.fromarray(rgb[index]).save(directory / "rgb.png")
        Image.fromarray(depth_mm[index]).save(directory / "depth_mm.png")
        Image.fromarray((valid * 255).astype(np.uint8)).save(
            directory / "valid_mask.png"
        )
        np.savez(
            directory / "pair.npz",
            rgb=rgb[index],
            depth_mm=depth_mm[index],
            depth_metres=depth,
            valid_mask=valid,
            observed_mask=observed,
            hidden_mask=hidden,
        )
        panels.append(
            (ids[index], rgb[index], depth, valid, observed, hidden, inputs[5] * 5)
        )
    figure, axes = plt.subplots(len(panels), 6, figsize=(18, 10))
    for row, (sample_id, image, depth, valid, observed, hidden, nearest) in enumerate(
        panels
    ):
        displays = (
            image,
            np.ma.masked_where(~valid, depth),
            valid,
            observed,
            hidden,
            nearest,
        )
        labels = (
            "RGB",
            "Measured depth (m)",
            "Original valid",
            "Observed",
            "Hidden valid target",
            "Nearest fill (m)",
        )
        for column, (display, label) in enumerate(zip(displays, labels)):
            options = (
                {}
                if column == 0
                else (
                    {"cmap": depth_palette, "vmin": 0, "vmax": 8}
                    if column in (1, 5)
                    else {"cmap": "gray", "vmin": 0, "vmax": 1}
                )
            )
            axes[row, column].imshow(display, **options)
            axes[row, column].set_title(label if row == 0 else "")
            axes[row, column].set_axis_off()
        axes[row, 0].text(
            0, -0.09, sample_id, transform=axes[row, 0].transAxes, fontsize=8
        )
    figure.suptitle(
        "Training pairs: black invalid depth; white masks; depth display fixed to 0–8 m"
    )
    figure.tight_layout()
    figure.savefig(output / "training_pairs.png", dpi=130)
    plt.close(figure)
    return [panel[0] for panel in panels]


def reuse_cache_evidence(project, raw, manifest, evidence_directory, output):
    """Reuse full decoding evidence only after rehashing every source and cache.

    This avoids expensive repeated PNG decoding after an evaluator-only change.
    It never bypasses source identity or reports an incomplete prior audit as valid.
    """
    import shutil

    summary = read_json(evidence_directory / "cache_integrity.json")
    if summary["status"] != "passed" or summary["manifest_sha256"] != file_hash(
        manifest
    ):
        raise ValueError("Prior cache evidence is incomplete or has another manifest")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    evaluation_records(rows)
    sources = {row["sample_id"]: row for row in rows if row["split"] != "test"}
    evidence_path = evidence_directory / "source_verification.jsonl"
    verified = [
        json.loads(line) for line in evidence_path.read_text().splitlines() if line
    ]
    if len(verified) != 2173 or {item["sample_id"] for item in verified} != set(
        sources
    ):
        raise ValueError("Prior source verification has missing/duplicate frames")
    for item in verified:
        row = sources[item["sample_id"]]
        for modality in ("rgb", "depth"):
            source = (raw / row[modality + "_path"]).resolve()
            if (
                not source.is_relative_to(raw)
                or file_hash(source) != item[modality + "_sha256"]
            ):
                raise ValueError("Source changed since the full native decoding audit")
    for relative, cache in summary["cache_files"].items():
        if file_hash(project / f"data/caches/{relative}.npz") != cache["sha256"]:
            raise ValueError("Cache changed since the full native decoding audit")
    shutil.copyfile(evidence_path, output / evidence_path.name)
    summary["reused_full_decode_evidence"] = str(
        evidence_directory.relative_to(project)
    )
    summary["all_source_and_cache_hashes_rechecked"] = True
    write_json(output / "cache_integrity.json", summary)
    print(
        "Reused full cache decoding evidence after rehashing all sources/caches",
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-evidence",
        type=Path,
        help="Prior passed cache evidence; every source/cache hash is rechecked",
    )
    args = parser.parse_args()
    project = args.project.resolve()
    raw = project / "data/raw/torwic_slam"
    manifest = project / "data/manifests/manifest.jsonl"
    output = require_output(
        args.output,
        sources=[raw, project / "data/caches", project / "models", manifest],
    )
    output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        caches = (
            reuse_cache_evidence(
                project, raw, manifest, args.cache_evidence.resolve(), output
            )
            if args.cache_evidence
            else audit_caches(project, raw, manifest, output)
        )
        baselines = reproduce_baselines(project, output)
        previews = export_training_previews(project, output)
        summary = {
            "status": "passed",
            "verification_date": "2026-10-13",
            "cache_integrity": caches,
            "baseline_reproduction": "passed",
            "training_preview_ids": previews,
            "visual_review": "pending",
            "elapsed_seconds": time.perf_counter() - started,
            "environment": {
                "python": sys.version,
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "gpu": torch.cuda.get_device_name(),
            },
            "code_sha256": {
                str(path.relative_to(project)): file_hash(path)
                for path in (
                    Path(__file__),
                    project / "src/training/depth_baseline.py",
                    project / "src/training/depth_context.py",
                    project / "src/training/depth_quality.py",
                )
            },
            "full_retraining": False,
            "test_sensor_files_opened": 0,
            "limitations": [
                "Single development-validation trajectory; repeated selection is not unbiased testing.",
                "Controlled corruption and hypothetical noise do not certify sensor realism.",
                "Original zero-depth pixels have no supervised reference; filled values are estimates.",
                "Low-resolution depth research is separate from goods-segmentation evidence.",
            ],
        }
        write_json(output / "review_summary.json", summary)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "output": str(output),
                }
            ),
            flush=True,
        )
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
