"""Bounded-memory RGB/depth cache preparation independent of source dataset names."""

import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash, reserve_output
from src.core.contracts import DatasetAdapter, validate_job, validate_record


def verify_npz_arrays(path: Path, expected: dict) -> None:
    """Check lossless NPZ contents in bounded chunks instead of loading all frames."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != {
            key + ".npy" for key in expected
        }:
            raise ValueError("Unexpected or duplicate NPZ members")
        for key, array in expected.items():
            with archive.open(key + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported NPZ array header version")
                if shape != array.shape or dtype != array.dtype or fortran:
                    raise ValueError("Cache array header mismatch")
                flat = array.reshape(-1)
                chunk_items = max(1, 1024 * 1024 // dtype.itemsize)
                for offset in range(0, flat.size, chunk_items):
                    count = min(chunk_items, flat.size - offset)
                    payload = stream.read(count * dtype.itemsize)
                    if len(payload) != count * dtype.itemsize or not np.array_equal(
                        np.frombuffer(payload, dtype=dtype),
                        flat[offset : offset + count],
                        equal_nan=dtype.kind in "fc",
                    ):
                        raise ValueError("Lossless cache round-trip failed")
                if stream.read(1):
                    raise ValueError("Unexpected trailing array data")


def validate_membership(records: list[dict]) -> None:
    """Reject duplicate identities or groups shared across declared split boundaries."""
    identities, groups, payload_splits = set(), {}, {}
    for record in records:
        validate_record(record)
        identity = record["dataset_id"], record["sample_id"]
        if identity in identities:
            raise ValueError("Duplicate canonical sample identity")
        identities.add(identity)
        group = record["dataset_id"], record["group_id"]
        if group in groups and groups[group] != record["split"]:
            raise ValueError("A source group crosses dataset split boundaries")
        groups[group] = record["split"]
        for reference in record["modalities"].values():
            if reference["status"] != "available":
                continue
            payload = record["dataset_id"], reference["path"]
            if payload in payload_splits and payload_splits[payload] != record["split"]:
                raise ValueError(
                    "A source payload path crosses dataset split boundaries"
                )
            payload_splits[payload] = record["split"]
    if not records:
        raise ValueError("Empty input manifest")


def resized_camera(record: dict, resolution: tuple[int, int]) -> dict | None:
    """Adjust intrinsics using Pillow's pixel-centre resize mapping; no undistortion."""
    if "camera" not in record:
        return None
    sx, sy = (
        resolution[0] / record["image_size"][0],
        resolution[1] / record["image_size"][1],
    )
    fx, fy, cx, cy = record["camera"]["intrinsics"]
    return {
        **record["camera"],
        "intrinsics": [fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5],
        "image_size": list(resolution),
        "resampling_geometry": "pixel_centre_resize",
    }


def _write_split(adapter, selected, split, output, parameters, provenance, prepared):
    """Stream native pairs into disk-backed arrays, retaining lossless source depths."""
    width, height = parameters["resolution"]
    shape = (len(selected), height, width)
    work = output / f".{split}_arrays"
    work.mkdir()
    formats = {"rgb": (np.uint8, (*shape, 3)), "depth_mm": (np.uint16, shape)}
    canonical = parameters["cache_format"] == "canonical"
    if canonical:
        formats = {
            "rgb": (np.uint8, (*shape, 3)),
            "depth_raw": (np.uint16, shape),
            "depth_metres": (np.float32, shape),
            "depth_valid": (np.bool_, shape),
        }
    arrays = {
        key: np.lib.format.open_memmap(
            work / f"{key}.npy", mode="w+", dtype=dtype, shape=array_shape
        )
        for key, (dtype, array_shape) in formats.items()
    }
    native_valid, native_pixels, excluded = 0, 0, 0
    fractions = []
    raw_conventions = set()
    try:
        for index, record in enumerate(selected):
            pair = adapter.read_rgb_depth(record)
            if not canonical and (
                pair.scale_to_metres != 0.001 or pair.invalid_raw_value != 0
            ):
                raise ValueError(
                    "Legacy cache requires uint16 millimetres with zero invalid; use canonical format"
                )
            raw_conventions.add((pair.scale_to_metres, pair.invalid_raw_value))
            native_m = pair.depth_raw.astype(np.float32) * pair.scale_to_metres
            source_valid = (pair.depth_raw != pair.invalid_raw_value) & (native_m > 0)
            valid = source_valid & (native_m <= parameters["depth_max_metres"])
            native_valid += int(valid.sum())
            native_pixels += valid.size
            excluded += int((source_valid & ~valid).sum())
            fractions.append(float(valid.mean()))
            rgb = np.array(
                Image.fromarray(pair.rgb).resize((width, height), Image.Resampling.BOX)
            )
            depth_raw = np.array(
                Image.fromarray(pair.depth_raw).resize(
                    (width, height), Image.Resampling.NEAREST
                )
            )
            arrays["rgb"][index] = rgb
            if canonical:
                depth_m = depth_raw.astype(np.float32) * pair.scale_to_metres
                depth_valid = (
                    (depth_raw != pair.invalid_raw_value)
                    & (depth_m > 0)
                    & (depth_m <= parameters["depth_max_metres"])
                )
                arrays["depth_raw"][index] = depth_raw
                arrays["depth_metres"][index] = np.where(depth_valid, depth_m, np.nan)
                arrays["depth_valid"][index] = depth_valid
            else:
                arrays["depth_mm"][index] = depth_raw
            provenance.write(json.dumps(pair.provenance, allow_nan=False) + "\n")
            prepared.write(
                json.dumps(
                    {
                        "dataset_id": record["dataset_id"],
                        "sample_id": record["sample_id"],
                        "split": split,
                        "cache": f"{split}.npz",
                        "array_index": index,
                        "source_image_size": record["image_size"],
                        "camera": resized_camera(record, (width, height)),
                        "depth_raw_scale_to_metres": pair.scale_to_metres,
                        "depth_invalid_raw_value": pair.invalid_raw_value,
                        "source_hashes": pair.provenance,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            if (index + 1) % 100 == 0:
                print(f"{split}: decoded {index+1}/{len(selected)}", flush=True)
        ids = np.array([record["sample_id"] for record in selected])
        np.savez(output / f"{split}.npz", **arrays, sample_ids=ids)
        verify_npz_arrays(output / f"{split}.npz", {**arrays, "sample_ids": ids})
    finally:
        # Windows requires every mapping to close before deleting temporary arrays.
        for array in arrays.values():
            array.flush()
            array._mmap.close()
        arrays.clear()
        shutil.rmtree(work)
    groups = sorted({record["group_id"] for record in selected})
    return {
        "frames": len(selected),
        "groups": groups,
        "sequence": groups[0] if len(groups) == 1 else None,
        "native_valid_depth_fraction": native_valid / native_pixels,
        "native_pixels_over_20m": (
            excluded if parameters["depth_max_metres"] == 20 else None
        ),
        "native_pixels_above_supervision_limit": excluded,
        "native_valid_fraction_min_median_max": [
            min(fractions),
            float(np.median(fractions)),
            max(fractions),
        ],
        "source_raw_depth_conventions": [
            {"scale_to_metres": scale, "invalid_raw_value": sentinel}
            for scale, sentinel in sorted(raw_conventions)
        ],
    }


def prepare_cache(adapter: DatasetAdapter, job: dict, output: Path) -> dict:
    """Prepare train/development caches; reject test requests before sensor reads.

    Full source metadata membership is checked before creating derivatives. Native
    image pairs are decoded one at a time; array storage uses disk-backed mappings.
    The canonical format retains raw depth plus masked float32 metres. Legacy mode
    preserves the existing research NPZ layout without modifying any old cache.
    """
    validate_job(job)
    if (
        job["dataset_id"] != adapter.dataset_id
        or job["source_manifest_sha256"] != adapter.manifest_sha256
    ):
        raise ValueError("Job is bound to a different dataset or manifest")
    records = list(adapter.records())
    validate_membership(records)
    parameters = job["parameters"]
    width, height = parameters["resolution"]
    selections = {
        split: [record for record in records if record["split"] == split]
        for split in parameters["splits"]
    }
    for selected in selections.values():
        if not selected:
            raise ValueError("Requested split has no source frames")
        for record in selected:
            native_width, native_height = record["image_size"]
            if width > native_width or width * native_height != height * native_width:
                raise ValueError(
                    "Cache resolution must preserve aspect ratio without upscaling"
                )
            if any(
                record["modalities"].get(name, {}).get("status") != "available"
                for name in ("rgb", "depth")
            ):
                raise ValueError("Requested source lacks RGB/depth")
    output = reserve_output(output, (adapter.root, adapter.manifest))
    report = {
        "schema_version": "dubaidata/1.0",
        "status": "running",
        "job_id": job["job_id"],
        "dataset_id": adapter.dataset_id,
        "manifest_sha256": adapter.manifest_sha256,
        "resolution": [width, height],
        "cache_format": parameters["cache_format"],
        "depth_units": "metres",
        "depth_storage_units": (
            "declared source units"
            if parameters["cache_format"] == "canonical"
            else "millimetres"
        ),
        "depth_valid_range_metres": [0, parameters["depth_max_metres"]],
        "rgb_resampling": "BOX area averaging",
        "depth_resampling": "nearest; no invented depths",
        "test_sensor_files_opened": 0,
        "splits": {},
        "cache_round_trip": "pending",
        "limitations": [
            "RGB/depth preparation does not certify LiDAR, registration or semantic accuracy.",
            "Measured sensor depth is a reference at valid pixels, not independent ground truth.",
            "Resampling is a declared low-resolution derivative; unknown depth remains invalid.",
        ],
    }
    (output / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    try:
        with (
            (output / "source_provenance.jsonl").open(
                "w", encoding="utf-8"
            ) as provenance,
            (output / "prepared_manifest.jsonl").open(
                "w", encoding="utf-8"
            ) as prepared,
        ):
            for split, selected in selections.items():
                report["splits"][split] = _write_split(
                    adapter, selected, split, output, parameters, provenance, prepared
                )
        if file_hash(adapter.manifest) != adapter.manifest_sha256:
            raise ValueError("Source manifest changed during preparation")
        report.update(
            status="passed",
            cache_round_trip="passed",
            cache_sha256={
                split: file_hash(output / f"{split}.npz") for split in selections
            },
        )
        (output / "profile.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        return report
    except Exception as error:
        (output / "failure.json").write_text(
            json.dumps({"status": "failed", "error": str(error)}), encoding="utf-8"
        )
        raise
