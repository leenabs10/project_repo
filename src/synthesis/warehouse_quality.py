"""Resolve source-backed packaging and equipment coverage for appearance qualification.

The layout and cases are authored, not reconstructed from customer scenes. Only a
verified training profile conditions this job. Scene intent remains portable;
source conversion, renderer execution and task admission remain separate adapters.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import random
import re

import numpy as np

from src.adapters.storage.local import file_hash
from src.adapters.tasks.goods_scene import validate_goods_scene
from src.synthesis.warehouse_mesh import build_mesh_job

ASSET_NAMES = {"cardboard_box_01", "cement_bag", "plastic_crate_01"}
GOODS_NAMES = {"cardboard_box_01", "cement_bag"}


def _fields(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("Unknown or missing warehouse appearance fields")


def _finite_vector(value, length):
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(
            type(item) not in (int, float) or not math.isfinite(item) for item in value
        )
    ):
        raise ValueError("Expected explicit finite metric coordinates")


def validate_quality_configuration(configuration):
    """Bound authored positions, camera poses, source families and support regions."""
    _fields(configuration, ("schema_version", "stock_seed", "cases"))
    if configuration["schema_version"] != "dubaidata.warehouse-quality/1.0":
        raise ValueError("Unsupported warehouse appearance version")
    if (
        type(configuration["stock_seed"]) is not int
        or not 0 <= configuration["stock_seed"] < 2**31
    ):
        raise ValueError("Explicit nonnegative stock seed is required")
    cases = configuration["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        raise ValueError("Appearance coverage requires one to eight cases")
    identifiers = set()
    for case in cases:
        _fields(case, ("sample_id", "objective", "camera", "light_scale", "loads"))
        name = case["sample_id"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name)
            or name in identifiers
        ):
            raise ValueError("Duplicate or unsafe appearance sample identifier")
        identifiers.add(name)
        if not isinstance(case["objective"], str) or not case["objective"].strip():
            raise ValueError("Every appearance case requires its coverage objective")
        _finite_vector(case["camera"], 4)
        x, y, height, yaw = case["camera"]
        if not (
            abs(x) <= 0.6
            and -6 <= y <= -4.5
            and 1.2 <= height <= 1.45
            and abs(yaw) <= 5
        ):
            raise ValueError("Camera pose exceeds bounded aisle coverage")
        if (
            type(case["light_scale"]) not in (int, float)
            or not 0.8 <= case["light_scale"] <= 1.1
        ):
            raise ValueError("Light scaling exceeds declared appearance bounds")
        if not isinstance(case["loads"], list) or len(case["loads"]) > 2:
            raise ValueError("At most two separate foreground loads are supported")
        for load in case["loads"]:
            _fields(load, ("asset", "variant", "support", "xy", "scale"))
            _finite_vector(load["xy"], 2)
            if load["asset"] not in GOODS_NAMES or load["variant"] not in {
                "front",
                "back",
            }:
                raise ValueError("Loads require a supported identifiable-goods asset")
            if (
                type(load["scale"]) not in (int, float)
                or not 0.6 <= load["scale"] <= 2.0
            ):
                raise ValueError("Uniform load scale exceeds bounded source dimensions")
            x, y = load["xy"]
            if load["support"] == "pallet":
                if not (-1.45 <= x <= -0.25 and -2.1 <= y <= -1.1):
                    raise ValueError("Pallet load is outside its declared support")
            elif load["support"] == "floor":
                if not (-1.5 <= x <= 1.5 and 1.0 <= y <= 5):
                    raise ValueError("Floor load is outside declared aisle coverage")
            else:
                raise ValueError("Unknown load support")
    return configuration


def build_quality_job(
    profile, profile_sha256, camera, catalog, catalog_sha256, configuration
):
    """Resolve every metric bound, stock choice and camera/light pose before rendering.

    Source aspect ratios are retained through uniform scale. Pallet footprints and
    separate load bodies are checked explicitly; cross-role penetration is checked
    by the goods task adapter. Bounding-box contact is not a stability simulation.
    """
    validate_quality_configuration(configuration)
    job = build_mesh_job(profile, profile_sha256, camera, catalog, catalog_sha256)
    job["job_id"] = "warehouse_goods_appearance_coverage"
    job["conditioning"][
        "purpose"
    ] = "Training-only equipment discrimination and authored packaging coverage; not an observed customer asset inventory"
    sources = {}
    for name in sorted(ASSET_NAMES):
        for variant in ("front", "back"):
            entry = catalog["assets"][f"polyhaven.{name}.{variant}.4k"]
            extent = np.asarray(
                entry["conversion"]["original_extent_metres"], dtype=float
            )
            if (
                extent.shape != (3,)
                or not np.isfinite(extent).all()
                or np.any(extent <= 0)
            ):
                raise ValueError("Registered source dimensions must be positive metres")
            material = f"source_{name}_{variant}"
            job["materials"][material] = dict(
                color=[0.5, 0.4, 0.3],
                roughness=0.8,
                metallic=0,
                texture=False,
                texture_set=dict(
                    asset_id=entry["asset_id"],
                    tile_size_metres=entry["tile_size_metres"],
                    base_color_mix=1.0,
                    projection="UV",
                    maps=copy.deepcopy(entry["maps"]),
                ),
            )
            sources[name, variant] = (entry, extent, material)

    def source_object(identifier, name, variant, xy, bottom, scale, role):
        entry, extent, material = sources[name, variant]
        dimensions = (extent * scale).tolist()
        return dict(
            instance_id=identifier,
            asset_id="registered.normalized_triangle_mesh/1.0",
            role=role,
            class_id=int(role == "goods"),
            center_metres=[*xy, bottom + dimensions[2] / 2],
            dimensions_metres=dimensions,
            material=material,
            bevel_metres=0,
            mesh=copy.deepcopy(entry["mesh"]),
        )

    base = copy.deepcopy(job["scenes"][0])
    objects = []
    rng = random.Random(configuration["stock_seed"])
    for obj in base["objects"]:
        if obj["material"] == "container_blue":
            continue
        if obj["role"] != "goods":
            objects.append(obj)
            continue
        # Resolve stock independently per slot, rather than cloning mirror-image
        # rows. Empty crates are equipment even when stored beside goods.
        draw = rng.random()
        if draw < 0.18:
            continue
        name = (
            "plastic_crate_01"
            if draw < 0.32
            else "cement_bag" if draw < 0.58 else "cardboard_box_01"
        )
        variant = rng.choice(("front", "back"))
        extent = sources[name, variant][1]
        scale = min(np.asarray(obj["dimensions_metres"]) / extent) * rng.uniform(
            0.86, 1.0
        )
        x, y, z = obj["center_metres"]
        objects.append(
            source_object(
                obj["instance_id"],
                name,
                variant,
                [x, y],
                z - obj["dimensions_metres"][2] / 2,
                scale,
                "equipment" if name == "plastic_crate_01" else "goods",
            )
        )
    next_id = max(obj["instance_id"] for obj in base["objects"]) + 1
    # These feet are explicit background geometry, outside occupied load bounds.
    for side in (-1, 1):
        for dx in (-0.66, 0.66):
            for y in (-1.7, 2.3, 6.3, 10.3, 14.3, 17.7):
                objects.append(
                    dict(
                        instance_id=next_id,
                        asset_id="builtin.metric_cuboid/1.0",
                        role="structure",
                        class_id=0,
                        center_metres=[side * 3.1 + dx, y, 0.008],
                        dimensions_metres=[0.20, 0.24, 0.016],
                        material="steel",
                        bevel_metres=0.002,
                    )
                )
                next_id += 1
    crate = source_object(
        next_id, "plastic_crate_01", "front", [0.9, 0.1], 0, 2.0, "equipment"
    )
    objects.append(crate)
    next_id += 1
    captures = []
    for case in configuration["cases"]:
        capture = copy.deepcopy(base)
        capture.update(
            sample_id=case["sample_id"],
            scenario=case["objective"],
            objects=copy.deepcopy(objects),
        )
        loads = []
        for offset, load in enumerate(case["loads"]):
            obj = source_object(
                next_id + offset,
                load["asset"],
                load["variant"],
                load["xy"],
                0.22 if load["support"] == "pallet" else 0,
                load["scale"],
                "goods",
            )
            if load["support"] == "pallet":
                center = np.asarray(obj["center_metres"][:2])
                half = np.asarray(obj["dimensions_metres"][:2]) / 2
                if np.any(center - half < [-1.45, -2.1]) or np.any(
                    center + half > [-0.25, -1.1]
                ):
                    raise ValueError("Load footprint overhangs its declared pallet")
            for previous in loads:
                overlap = (
                    np.asarray(previous["dimensions_metres"]) + obj["dimensions_metres"]
                ) / 2 - np.abs(
                    np.asarray(previous["center_metres"]) - obj["center_metres"]
                )
                if np.all(overlap > 1e-4):
                    raise ValueError("Separate foreground loads penetrate each other")
            loads.append(obj)
        capture["objects"].extend(loads)
        x, y, height, yaw = case["camera"]
        angle = math.radians(yaw)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        pose = np.asarray(base["optical_camera_to_world"], dtype=float)
        pose[:3, :3] = rotation @ pose[:3, :3]
        pose[:3, 3] = [x, y, height]
        capture["optical_camera_to_world"] = pose.tolist()
        for index, light in enumerate(capture["lights"]):
            light["power_watts"] *= case["light_scale"] * (0.94 if index % 2 else 1.03)
        captures.append(capture)
    job["scenes"] = captures
    job["materials"]["concrete"]["texture_set"]["base_color_mix"] = 1.0
    job["render"].update(exposure_ev=-1.1, seed=configuration["stock_seed"])
    # Drop unused materials so each retained texture has a real scene consumer.
    used = {obj["material"] for capture in captures for obj in capture["objects"]}
    job["materials"] = {
        name: value for name, value in job["materials"].items() if name in used
    }
    return validate_goods_scene(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "profile",
        "camera-config",
        "asset-catalog",
        "coverage-config",
        "output",
    ):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    source_camera = json.loads(args.camera_config.read_text(encoding="utf-8"))["camera"]
    camera = dict(
        width=source_camera["width"],
        height=source_camera["height"],
        intrinsics=[source_camera[key] for key in ("fx", "fy", "cx", "cy")],
        near_metres=0.05,
        far_metres=60,
        model="pinhole_zero_distortion",
    )
    job = build_quality_job(
        json.loads(args.profile.read_text(encoding="utf-8")),
        file_hash(args.profile),
        camera,
        json.loads(args.asset_catalog.read_text(encoding="utf-8")),
        file_hash(args.asset_catalog),
        json.loads(args.coverage_config.read_text(encoding="utf-8")),
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(job, stream, indent=2, allow_nan=False)
    print(
        json.dumps(dict(captures=len(job["scenes"]), job_sha256=file_hash(args.output)))
    )


if __name__ == "__main__":
    main()
