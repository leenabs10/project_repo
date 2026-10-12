"""Validate and package a Blender geometry pilot; physical realism stays unapproved."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "profiling"))
from task_dataset_review import goods_mask


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_arrays(rgb, depth, valid, semantic, camera):
    """Check native formats, depth masks and discrete labels without repairing them."""
    shape = (camera["height"], camera["width"])
    if rgb.shape != (*shape, 3) or rgb.dtype != np.uint8:
        raise ValueError("Unexpected native RGB shape/type")
    if (
        depth.shape != shape
        or depth.dtype != np.float32
        or valid.shape != shape
        or valid.dtype != np.bool_
    ):
        raise ValueError("Unexpected native depth/mask shape/type")
    if semantic.shape != shape or semantic.dtype != np.uint8 or np.any(semantic > 15):
        raise ValueError("Unexpected semantic shape/type/classes")
    if (
        not np.array_equal(np.isnan(depth), ~valid)
        or not np.isfinite(depth[valid]).all()
        or np.any(depth[valid] <= 0)
    ):
        raise ValueError("Depth invalids/mask/finite metric values disagree")
    if np.any((semantic != 0) != valid):
        raise ValueError(
            "Clean scene depth validity and visible object labels disagree"
        )
    if not np.any(semantic == 7):
        raise ValueError("No goods pixels in the goods-task pilot")


def main():
    """Create a fresh local dataset ZIP after source and lossless export verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.render_root.resolve(), args.output.resolve()
    raw = args.raw_root.resolve()
    if output.exists() or output.is_relative_to(raw) or output.is_relative_to(source):
        raise ValueError("Choose a fresh package directory outside raw/render sources")
    summary_path = source / "render_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "rendered_geometry_checks_passed"
        or summary.get("samples") != 3
    ):
        raise ValueError("Expected a complete, geometry-validated three-sample pilot")
    if (
        summary.get("training_eligible") is not False
        or summary.get("physical_sensor_realism_certified") is not False
    ):
        raise ValueError(
            "Pilot must retain its explicit unapproved training/realism state"
        )
    output.mkdir(parents=True)
    rows, previews = [], []
    for record in summary["records"]:
        name = record["sample_id"]
        folder = (source / name).resolve()
        if not folder.is_relative_to(source) or folder == source:
            raise ValueError("Sample folder escapes render root")
        for filename, expected in record["files"].items():
            path = (folder / expected["path"]).resolve()
            if (
                path.parent != folder
                or path.name != filename
                or file_hash(path) != expected["sha256"]
            ):
                raise ValueError("Rendered payload path/hash mismatch")
        camera = record["camera"]
        if (
            (camera["width"], camera["height"]) != (1280, 720)
            or camera["convention"] != "synthetic_pinhole"
            or camera["D"] != [0] * 5
        ):
            raise ValueError(
                "Pilot geometry differs from declared native synthetic camera"
            )
        pose = np.asarray(record["optical_camera_to_world"], dtype=float)
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1)
        ):
            raise ValueError("Invalid optical-camera metric pose")
        with Image.open(folder / "rgb.png") as image:
            rgb = np.array(image)
            luma = np.array(image.convert("L"))
        depth = np.load(folder / "depth_axial_m.npy", allow_pickle=False)
        valid = np.load(folder / "depth_valid.npy", allow_pickle=False)
        semantic = np.load(folder / "semantic_ids.npy", allow_pickle=False)
        validate_arrays(rgb, depth, valid, semantic, camera)
        target = goods_mask(semantic)
        destination = output / name
        destination.mkdir()
        for filename in ("rgb.png", "depth_axial_m.npy", "depth_valid.npy"):
            shutil.copyfile(folder / filename, destination / filename)
            if file_hash(destination / filename) != file_hash(folder / filename):
                raise ValueError("Copied export is not byte-identical")
        for filename, array in (
            ("semantic_ids.png", semantic),
            ("goods_target.png", target),
        ):
            Image.fromarray(array).save(destination / filename)
            with Image.open(destination / filename) as image:
                np.testing.assert_array_equal(np.array(image), array)
        overlay = rgb.copy()
        overlay[target > 0] = (
            rgb[target > 0].astype(float) * 0.55 + np.array([235, 70, 210]) * 0.45
        ).astype(np.uint8)
        Image.fromarray(overlay).save(destination / "goods_overlay.png")
        metadata = {
            **record,
            "packaged_files": {
                path.name: file_hash(path)
                for path in sorted(destination.iterdir())
                if path.is_file()
            },
            "binary_target_mapping": {
                "0": "other",
                "1": "goods/materials semantic class 7",
            },
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        rows.append(
            {
                "sample_id": name,
                "mean_luma": float(luma.mean()),
                "goods_pixel_fraction": float(target.mean()),
                "rgb_round_trip": "byte_identical",
                "labels_round_trip": "pixel_identical",
                "depth_and_mask_round_trip": "byte_identical",
                "geometry_checks": record["geometry_checks"],
                "training_eligible": False,
            }
        )
        previews.append((name, Image.fromarray(rgb), Image.fromarray(overlay)))
    sheet = Image.new("RGB", (1280, 1170), "#202830")
    draw = ImageDraw.Draw(sheet)
    for index, (name, rgb, overlay) in enumerate(previews):
        y = index * 390
        draw.text((15, y + 10), name + " | RGB", fill="white")
        draw.text((655, y + 10), "Goods geometry label overlay", fill="white")
        sheet.paste(rgb.resize((640, 360), Image.Resampling.LANCZOS), (0, y + 30))
        sheet.paste(overlay.resize((640, 360), Image.Resampling.LANCZOS), (640, y + 30))
    sheet.save(output / "renderer_examples.png")
    readme = """Warehouse goods geometry pilot
Native RGB: 1280x720, uint8 RGB, lossless PNG.
Depth: float32 axial Z in metres, NPY; NaN invalid, accompanied by bool valid mask.
Semantic IDs: uint8 PNG with geometry classes 0..15 matching the TorWIC class names.
Goods target: uint8 PNG, binary 0=other, 1=goods/materials source class 7.
Camera: independent zero-distortion synthetic pinhole; optical X right/Y down/Z forward.
Poses and world geometry are synthetic and expressed in metres.
No real-image rectification, IMU fusion, LiDAR scan or calibrated sensor noise.
These clean geometry examples passed export/alignment checks. Appearance and
downstream utility are not certified. Training eligibility remains false.
"""
    (output / "README.txt").write_text(readme, encoding="utf-8")
    archive = output / "warehouse_goods_pilot.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(output / "README.txt", "README.txt")
        for record in rows:
            for path in sorted((output / record["sample_id"]).iterdir()):
                bundle.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Dataset ZIP failed its CRC integrity check")
        for member in bundle.namelist():
            if hashlib.sha256(bundle.read(member)).hexdigest() != file_hash(
                output / member
            ):
                raise ValueError("ZIP payload differs from verified export")
    report = {
        "status": "geometry_and_export_integrity_passed",
        "samples": rows,
        "render_summary_sha256": file_hash(summary_path),
        "code_sha256": file_hash(Path(__file__)),
        "archive_sha256": file_hash(archive),
        "archive_bytes": archive.stat().st_size,
        "native_resolution": [1280, 720],
        "all_exports_lossless": True,
        "independent_interior_rays": sum(
            row["geometry_checks"]["interior_rays"] for row in rows
        ),
        "device": summary["device"],
        "render_times_seconds": [
            record["render_seconds_including_export"] for record in summary["records"]
        ],
        "fixture": summary["analytic_fixture"],
        "training_eligible": False,
        "physical_sensor_realism_certified": False,
        "raw_trajectory_images_opened": 0,
    }
    (output / "validation_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "samples": len(rows),
                "independent_interior_rays": report["independent_interior_rays"],
                "archive_bytes": report["archive_bytes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
