"""Resolve training-supported carton and tyre cargo with explicit coverage intent.

All load dimensions and placements are authored within source aspect ratios.
This module does not reconstruct the warehouse or simulate load stability. The
training evidence is hash-bound to the resolved job's conditioning purpose.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from src.adapters.tasks.goods_scene import validate_goods_scene
from src.adapters.storage.local import file_hash
from src.synthesis.warehouse_quality import build_quality_job


def build_balanced_job(
    profile,
    profile_sha256,
    camera,
    catalog,
    catalog_sha256,
    base_configuration,
    configuration,
    training_evidence,
    evidence_sha256,
):
    """Build eight bounded cases without reading development/test images or labels."""
    if configuration.get("schema_version") != "dubaidata.warehouse-balanced/1.0" or set(
        configuration
    ) != {
        "schema_version",
        "exposure_ev",
        "stock_seed",
        "cases",
        "training_evidence_sha256",
    }:
        raise ValueError("Unsupported balanced warehouse configuration")
    if (
        configuration["training_evidence_sha256"] != evidence_sha256
        or len(evidence_sha256) != 64
    ):
        raise ValueError("Training object evidence binding mismatch")
    supported = {row["sample_id"] for row in training_evidence["training_support"]}
    if not supported or any(
        not name.startswith("Aisle_CCW_Run_1/") for name in supported
    ):
        raise ValueError("Object diversity requires explicit training-only evidence")
    if (
        not -1 <= configuration["exposure_ev"] <= 0
        or type(configuration["stock_seed"]) is not int
        or not 0 <= configuration["stock_seed"] < 2**31
    ):
        raise ValueError("Unsupported exposure or source seed")
    cases = configuration["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        raise ValueError("Balanced coverage requires one to eight captures")
    base_configuration = copy.deepcopy(base_configuration)
    base_configuration["stock_seed"] = configuration["stock_seed"]
    job = build_quality_job(
        profile, profile_sha256, camera, catalog, catalog_sha256, base_configuration
    )
    job["job_id"] = "warehouse_goods_training_coverage"
    job["conditioning"]["purpose"] = (
        "Training-supported carton/tyre cargo; evidence SHA256 " + evidence_sha256
    )
    sources = {}
    for name in ("cardboard_box_01", "old_tyre"):
        for variant in ("front", "back"):
            entry = catalog["assets"][f"polyhaven.{name}.{variant}.4k"]
            material = f"cargo_{name}_{variant}"
            extent = np.array(
                entry["conversion"]["original_extent_metres"], dtype=float
            )
            if (
                extent.shape != (3,)
                or not np.isfinite(extent).all()
                or np.any(extent <= 0)
            ):
                raise ValueError(
                    "Cargo source extents must be positive metric dimensions"
                )
            job["materials"][material] = dict(
                color=[0.4, 0.3, 0.2],
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

    def cargo(identifier, name, variant, xy, bottom, scale):
        entry, extent, material = sources[name, variant]
        dimensions = (extent * scale).tolist()
        return dict(
            instance_id=identifier,
            asset_id="registered.normalized_triangle_mesh/1.0",
            role="goods",
            class_id=1,
            center_metres=[*xy, bottom + dimensions[2] / 2],
            dimensions_metres=dimensions,
            material=material,
            bevel_metres=0,
            mesh=copy.deepcopy(entry["mesh"]),
        )

    base = copy.deepcopy(job["scenes"][0])
    # Remove exploratory bag stock. The new batch's goods families have direct
    # training-source evidence. Existing empty crates retain background semantics.
    for obj in base["objects"]:
        if obj["role"] == "goods":
            name = "old_tyre" if obj["instance_id"] % 4 == 0 else "cardboard_box_01"
            variant = "back" if obj["instance_id"] % 2 else "front"
            extent = sources[name, variant][1]
            scale = float(np.min(np.asarray(obj["dimensions_metres"]) / extent))
            x, y, z = obj["center_metres"]
            obj.update(
                cargo(
                    obj["instance_id"],
                    name,
                    variant,
                    [x, y],
                    z - obj["dimensions_metres"][2] / 2,
                    scale,
                )
            )
        elif (
            obj["dimensions_metres"][2] > 3 and max(obj["dimensions_metres"][:2]) < 0.12
        ):
            obj["material"] = "orange"
    next_id = max(obj["instance_id"] for obj in base["objects"]) + 1
    captures, identifiers = [], set()
    for case in cases:
        if (
            set(case) != {"sample_id", "objective", "camera", "light_scale", "stacks"}
            or case["sample_id"] in identifiers
        ):
            raise ValueError("Unknown or duplicate balanced coverage case")
        identifiers.add(case["sample_id"])
        if (
            not isinstance(case["objective"], str)
            or not case["objective"].strip()
            or not isinstance(case["stacks"], list)
            or len(case["stacks"]) > 2
        ):
            raise ValueError("Cases require an objective and at most two cargo stacks")
        pose_values = np.asarray(case["camera"], dtype=float)
        if pose_values.shape != (4,) or not np.isfinite(pose_values).all():
            raise ValueError("Balanced camera requires finite x/y/height/yaw")
        x, y, z, yaw = pose_values
        if (
            not (
                abs(x) <= 0.6
                and -5.5 <= y <= -3.4
                and 0.6 <= z <= 1.0
                and abs(yaw) <= 6
            )
            or not 0.85 <= case["light_scale"] <= 1.1
        ):
            raise ValueError("Balanced camera/light setting exceeds authored bounds")
        capture = copy.deepcopy(base)
        capture.update(sample_id=case["sample_id"], scenario=case["objective"])
        loads = []
        for stack in case["stacks"]:
            if set(stack) != {"asset", "xy", "support", "scale", "count"} or stack[
                "asset"
            ] not in {"old_tyre", "cardboard_box_01"}:
                raise ValueError("Unsupported identifiable cargo stack")
            if (
                type(stack["count"]) is not int
                or not 1 <= stack["count"] <= 6
                or not 0.7 <= stack["scale"] <= 1.7
            ):
                raise ValueError("Cargo stack exceeds bounded count/scale")
            xy = np.asarray(stack["xy"], dtype=float)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError("Cargo stack position must be finite metres")
            if stack["support"] == "pallet":
                bottom = 0.22
            elif (
                stack["support"] == "floor"
                and -1.5 <= xy[0] <= 1.5
                and 1.0 <= xy[1] <= 4
            ):
                bottom = 0
            else:
                raise ValueError("Unknown or misplaced cargo support")
            for layer in range(stack["count"]):
                obj = cargo(
                    next_id + len(loads),
                    stack["asset"],
                    "front" if layer % 2 else "back",
                    xy.tolist(),
                    bottom,
                    stack["scale"],
                )
                half = np.asarray(obj["dimensions_metres"][:2]) / 2
                if stack["support"] == "pallet" and (
                    np.any(xy - half < [-1.45, -2.1])
                    or np.any(xy + half > [-0.25, -1.1])
                ):
                    raise ValueError("Cargo stack overhangs its pallet")
                for previous in loads:
                    overlap = (
                        np.asarray(previous["dimensions_metres"])
                        + obj["dimensions_metres"]
                    ) / 2 - np.abs(
                        np.asarray(previous["center_metres"]) - obj["center_metres"]
                    )
                    if np.all(overlap > 1e-4):
                        raise ValueError("Distinct cargo bodies intersect")
                loads.append(obj)
                bottom += obj["dimensions_metres"][2]
        capture["objects"].extend(loads)
        angle = math.radians(yaw)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        pose = np.asarray(capture["optical_camera_to_world"], dtype=float)
        pose[:3, :3] = rotation @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        pose[:3, 3] = [x, y, z]
        capture["optical_camera_to_world"] = pose.tolist()
        for light in capture["lights"]:
            light["power_watts"] *= case["light_scale"]
        captures.append(capture)
    job["scenes"] = captures
    job["render"].update(
        exposure_ev=configuration["exposure_ev"], seed=configuration["stock_seed"]
    )
    used = {obj["material"] for scene in captures for obj in scene["objects"]}
    job["materials"] = {
        key: value for key, value in job["materials"].items() if key in used
    }
    return validate_goods_scene(job)


def main():
    """Resolve a portable job from explicit source-bound files; never render on import."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "profile",
        "camera-config",
        "asset-catalog",
        "base-configuration",
        "coverage-config",
        "training-evidence",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    camera = read(args.camera_config)["camera"]
    if "intrinsics" not in camera:
        camera = dict(
            width=camera["width"],
            height=camera["height"],
            intrinsics=[camera[key] for key in ("fx", "fy", "cx", "cy")],
            near_metres=0.05,
            far_metres=60,
            model="pinhole_zero_distortion",
        )
    job = build_balanced_job(
        read(args.profile),
        file_hash(args.profile),
        camera,
        read(args.asset_catalog),
        file_hash(args.asset_catalog),
        read(args.base_configuration),
        read(args.coverage_config),
        read(args.training_evidence),
        file_hash(args.training_evidence),
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(job, stream, indent=2, allow_nan=False)
    print(
        json.dumps(dict(captures=len(job["scenes"]), job_sha256=file_hash(args.output)))
    )


if __name__ == "__main__":
    main()
