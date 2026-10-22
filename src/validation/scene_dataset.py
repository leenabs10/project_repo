"""Verify rendered payloads and export canonical camera samples without admission.

Structural correctness is separate from appearance qualification, calibrated
sensor realism and measured task utility. A passing pilot stays training-ineligible.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.adapters.storage.local import file_hash, source_path
from src.core.contracts import validate_record
from src.core.scene import validate_scene_job
from src.adapters.tasks.goods_scene import validate_goods_scene
from src.adapters.generation.material_assets import verify_material_assets


def validate_arrays(rgb, depth, valid, semantic, instances, camera, objects):
    """Reject format, unit, validity, identity or taxonomy inconsistencies."""
    shape = (camera["height"], camera["width"])
    if rgb.dtype != np.uint8 or rgb.shape != (*shape, 3):
        raise ValueError("RGB must retain declared native uint8 geometry")
    if (
        depth.dtype != np.float32
        or depth.shape != shape
        or not np.isfinite(depth).all()
    ):
        raise ValueError("Depth must be finite float32 axial metres")
    if valid.dtype != np.bool_ or valid.shape != shape:
        raise ValueError("Depth validity must be a native boolean array")
    if (
        not np.array_equal(depth > 0, valid)
        or np.any(depth[~valid] != 0)
        or np.any(depth[valid] < camera["near_metres"])
        or np.any(depth[valid] >= camera["far_metres"])
    ):
        raise ValueError(
            "Depth values disagree with declared metric range/invalid sentinel"
        )
    if (
        semantic.dtype != np.uint8
        or semantic.shape != shape
        or instances.dtype != np.uint16
        or instances.shape != shape
    ):
        raise ValueError("Visible labels require native discrete uint8/uint16 geometry")
    lookup = {obj["instance_id"]: obj["class_id"] for obj in objects}
    lookup[0] = 0
    if not set(np.unique(instances)).issubset(lookup):
        raise ValueError("Instance is absent from resolved scene")
    if not np.array_equal(instances > 0, valid):
        raise ValueError("Visible geometry and clean depth validity disagree")
    for identity in np.unique(instances):
        if np.any(semantic[instances == identity] != lookup[int(identity)]):
            raise ValueError("Equipment/goods labels disagree with scene taxonomy")
    if not np.any(semantic == 1) or not np.any((semantic == 0) & valid):
        raise ValueError(
            "Goods benchmark pilot needs both visible goods and background"
        )


def export_dataset(job, worker_root, output, *, asset_root=None):
    """Hash-check one capture at a time, then write a canonical quarantine export."""
    validate_scene_job(job)
    validate_goods_scene(job)
    material_hashes = verify_material_assets(job, asset_root)
    worker_root, output = Path(worker_root), Path(output)
    profile = json.loads(
        (worker_root / "worker_profile.json").read_text(encoding="utf-8")
    )
    # The local runner serializes the resolved job exactly this way before
    # launch. Independently bind direct exports too, rather than trusting a
    # worker's capture names or a caller-provided metadata object.
    serialized = json.dumps(job, indent=2, allow_nan=False)
    # Text writes use CRLF on Windows and LF on other hosts. Either byte
    # representation must describe the same resolved object; do not reject a
    # valid Windows run or accept altered scene parameters during normalization.
    allowed_hashes = {
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in (serialized, serialized.replace("\n", "\r\n"))
    }
    resolved_job_sha256 = profile.get("job_sha256")
    if resolved_job_sha256 not in allowed_hashes:
        raise ValueError("Worker profile is bound to a different resolved scene job")
    if (
        profile.get("status") != "rendered_geometry_passed"
        or profile.get("training_eligible") is not False
        or profile.get("physical_sensor_realism_certified") is not False
        or profile.get("backend") != job["backend"]
    ):
        raise ValueError("Worker state does not describe a quarantined geometry pilot")
    fixture = profile["fixture"]
    if not 0 <= fixture["maximum_error_metres"] < 0.0005:
        raise ValueError("Analytic plane fixture failed")
    rows = profile["records"]
    if [row["sample_id"] for row in rows] != [
        scene["sample_id"] for scene in job["scenes"]
    ]:
        raise ValueError("Worker capture membership differs from resolved scene plan")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    # Keep lineage resolvable after the export leaves its execution directory.
    resolved_bytes = serialized.encode("utf-8")
    if hashlib.sha256(resolved_bytes).hexdigest() != resolved_job_sha256:
        resolved_bytes = serialized.replace("\n", "\r\n").encode("utf-8")
    (output / "generation_job.json").write_bytes(resolved_bytes)
    if material_hashes:
        (output / "asset_catalog.json").write_bytes(
            (Path(asset_root) / "catalog.json").read_bytes()
        )
    records, quality = [], []
    thumbnails = []
    for capture, row in zip(job["scenes"], rows, strict=True):
        identifier = row["sample_id"]
        required = {
            f"{identifier}/{name}"
            for name in (
                "rgb.png",
                "depth_axial_m.npy",
                "depth_valid.npy",
                "semantic_ids.npy",
                "instance_ids.npy",
            )
        }
        if not required.issubset(row["artifact_sha256"]):
            raise ValueError("Worker omitted required capture artifacts")
        for relative, expected in row["artifact_sha256"].items():
            if file_hash(source_path(worker_root, relative)) != expected:
                raise ValueError("Rendered artifact checksum mismatch")
        geometry = row["geometry"]
        if (
            geometry["interior_rays"] < 128
            or geometry["instance_agreement"] < 0.99
            or geometry["depth_error_metres_p50_p95_max"][1] > 0.005
        ):
            raise ValueError("Independent depth/instance ray checks failed")
        folder = worker_root / identifier
        with Image.open(folder / "rgb.png") as image:
            rgb = np.array(image)
        arrays = {
            name: np.load(folder / f"{name}.npy", allow_pickle=False)
            for name in ("depth_axial_m", "depth_valid", "semantic_ids", "instance_ids")
        }
        depth, valid = arrays["depth_axial_m"], arrays["depth_valid"]
        semantic, instances = arrays["semantic_ids"], arrays["instance_ids"]
        validate_arrays(
            rgb, depth, valid, semantic, instances, job["camera"], capture["objects"]
        )
        destination = output / identifier
        destination.mkdir()
        Image.fromarray(rgb).save(destination / "rgb.png")
        Image.fromarray(semantic).save(destination / "semantic_ids.png")
        Image.fromarray(instances).save(destination / "instance_ids.png")
        np.save(destination / "depth_axial_m.npy", depth, allow_pickle=False)
        np.save(destination / "depth_valid.npy", valid, allow_pickle=False)
        # Test categorical PNG round-trips at native resolution, never thumbnails.
        for filename, expected in (
            ("rgb.png", rgb),
            ("semantic_ids.png", semantic),
            ("instance_ids.png", instances),
        ):
            with Image.open(destination / filename) as image:
                if not np.array_equal(np.asarray(image), expected):
                    raise ValueError("Lossless image round-trip failed")
        modalities = {}
        for name, filename in (
            ("rgb", "rgb.png"),
            ("depth", "depth_axial_m.npy"),
            ("depth_valid", "depth_valid.npy"),
            ("semantic", "semantic_ids.png"),
            ("instance", "instance_ids.png"),
        ):
            modalities[name] = dict(
                status="available",
                path=f"{identifier}/{filename}",
                sha256=file_hash(destination / filename),
            )
        modalities["depth"].update(
            units="metres",
            scale_to_metres=1.0,
            invalid_raw_value=0,
            frame="optical_camera_axial_z",
        )
        modalities["lidar"] = dict(
            status="unsupported", reason="Camera-only static scene backend"
        )
        record = validate_record(
            dict(
                schema_version="dubaidata/1.0",
                # Preserve scene-family grouping while separating dataset
                # versions with different geometry/material/render parameters.
                dataset_id=f"{job['job_id']}.{resolved_job_sha256}",
                sample_id=identifier,
                group_id=job["job_id"],
                split="train",
                timestamp_seconds=0.0,
                timestamp_basis="static_scene_shared_render_not_measured_capture_time",
                image_size=[job["camera"]["width"], job["camera"]["height"]],
                camera=dict(
                    sensor_id="synthetic_camera",
                    image_convention=job["camera"]["model"],
                    intrinsics=job["camera"]["intrinsics"],
                ),
                modalities=modalities,
                source_metadata=dict(
                    origin="scene_simulation",
                    job_sha256=profile["job_sha256"],
                    backend=job["backend"],
                    conditioning=job["conditioning"],
                    scenario=capture["scenario"],
                    optical_camera_to_world=capture["optical_camera_to_world"],
                    taxonomy=job["taxonomy"],
                    rgb_response=row.get(
                        "rgb_response", {"primary": "legacy_undeclared"}
                    ),
                    generation_job=dict(
                        path="generation_job.json", sha256=resolved_job_sha256
                    ),
                    asset_catalog=(
                        dict(
                            path="asset_catalog.json",
                            sha256=job["asset_catalog_sha256"],
                        )
                        if material_hashes
                        else None
                    ),
                    training_eligible=False,
                ),
            )
        )
        records.append(record)
        visible_equipment = [
            obj["instance_id"]
            for obj in capture["objects"]
            if obj["role"] == "equipment"
        ]
        equipment_pixels = int(np.isin(instances, visible_equipment).sum())
        if not equipment_pixels:
            raise ValueError("Requested transport-equipment coverage is not visible")
        quality.append(
            dict(
                sample_id=identifier,
                goods_pixels=int((semantic == 1).sum()),
                equipment_pixels=equipment_pixels,
                valid_depth_fraction=float(valid.mean()),
                mean_luma=float(np.mean(rgb @ np.array([0.2126, 0.7152, 0.0722]))),
                geometry=geometry,
                render_seconds=row["render_seconds"],
            )
        )
        overlay = rgb.copy()
        overlay[semantic == 1] = (
            0.55 * rgb[semantic == 1] + 0.45 * np.array([235, 70, 210])
        ).astype(np.uint8)
        equipment = np.isin(instances, visible_equipment)
        overlay[equipment] = (
            0.55 * rgb[equipment] + 0.45 * np.array([30, 210, 235])
        ).astype(np.uint8)
        Image.fromarray(overlay).save(destination / "label_overlay.png")
        # A fixed 0-20 m display is diagnostic only; scientific data remains float32.
        shade = np.clip(depth / 20, 0, 1)
        depth_rgb = np.stack(
            [shade * 255, np.sqrt(shade) * 255, (1 - shade) * 180], axis=-1
        ).astype(np.uint8)
        depth_rgb[~valid] = 0
        Image.fromarray(depth_rgb).save(destination / "depth_preview.png")
        thumbnails.append((identifier, rgb, overlay, depth_rgb))
    (output / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    sheet = Image.new("RGB", (1440, len(thumbnails) * 300), "#17212b")
    draw = ImageDraw.Draw(sheet)
    for index, (identifier, *images) in enumerate(thumbnails):
        y = index * 300
        for column, (label, image) in enumerate(
            zip(("RGB", "Goods magenta | equipment cyan", "Axial depth 0-20 m"), images)
        ):
            draw.text(
                (column * 480 + 8, y + 8), identifier + " | " + label, fill="white"
            )
            sheet.paste(
                Image.fromarray(image).resize((480, 270), Image.Resampling.LANCZOS),
                (column * 480, y + 30),
            )
    sheet.save(output / "scene_contact_sheet.png")
    artifacts = {
        path.relative_to(output).as_posix(): file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    result = dict(
        status="structural_checks_passed",
        backend=job["backend"],
        job_sha256=profile["job_sha256"],
        worker_profile_sha256=file_hash(worker_root / "worker_profile.json"),
        captures=len(records),
        image_size=records[0]["image_size"],
        quality=quality,
        fixture=fixture,
        artifact_sha256=artifacts,
        training_eligible=False,
        appearance_qualification="not_established",
        physical_sensor_realism_certified=False,
        simulated_lidar_supported=False,
        utility_measured=False,
    )
    (output / "profile.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
