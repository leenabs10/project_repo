"""Resolve a bounded warehouse pilot using registered source-backed carton meshes.

The carton keeps its original aspect ratio. Repeated rack loads are fitted within
the previous slots, while foreground examples use explicit uniform scale factors.
This authored pilot is not a warehouse reconstruction or a physics simulation.
"""

import argparse
import copy
import json
from pathlib import Path

from src.adapters.storage.local import file_hash
from src.adapters.tasks.goods_scene import validate_goods_scene
from src.synthesis.warehouse_scene import build_job
from src.synthesis.warehouse_surfaces import bind_surfaces


def build_mesh_job(profile, profile_sha256, camera, catalog, catalog_sha256):
    """Return resolved metric geometry with source UVs and explicit load semantics."""
    job = build_job(profile, profile_sha256, camera)
    job["schema_version"] = "dubaidata.scene/1.1"
    job["job_id"] = "warehouse_goods_mesh_pilot"
    base = job["scenes"][0]
    identifiers = [
        f"polyhaven.cardboard_box_01.{variant}.4k" for variant in ("front", "back")
    ]
    assets = [catalog["assets"][identifier] for identifier in identifiers]
    extent = assets[0]["conversion"]["original_extent_metres"]
    if any(asset["conversion"]["original_extent_metres"] != extent for asset in assets):
        raise ValueError(
            "Registered carton variants have inconsistent source dimensions"
        )
    for variant, asset in zip(("front", "back"), assets, strict=True):
        job["materials"][f"mesh_carton_{variant}"] = dict(
            color=[0.5, 0.35, 0.18],
            roughness=0.85,
            metallic=0,
            texture=False,
            texture_set=dict(
                asset_id=asset["asset_id"],
                tile_size_metres=asset["tile_size_metres"],
                base_color_mix=1.0,
                projection="UV",
                maps={
                    role: {
                        key: binding[key] for key in ("path", "sha256", "color_space")
                    }
                    for role, binding in asset["maps"].items()
                },
            ),
        )

    def carton(identifier, x, y, bottom, scale, variant):
        dimensions = [value * scale for value in extent]
        return dict(
            instance_id=identifier,
            asset_id="registered.normalized_triangle_mesh/1.0",
            role="goods",
            class_id=1,
            center_metres=[x, y, bottom + dimensions[2] / 2],
            dimensions_metres=dimensions,
            material=f"mesh_carton_{variant}",
            bevel_metres=0,
            mesh=copy.deepcopy(assets[int(variant == "back")]["mesh"]),
        )

    rack_objects = []
    for obj in base["objects"]:
        if obj["role"] != "goods":
            rack_objects.append(copy.deepcopy(obj))
        elif obj["material"] in {
            "paper",
            "carton_0",
            "carton_1",
            "carton_2",
            "carton_3",
        }:
            scale = min(
                dimension / original
                for dimension, original in zip(
                    obj["dimensions_metres"], extent, strict=True
                )
            )
            x, y, z = obj["center_metres"]
            rack_objects.append(
                carton(
                    obj["instance_id"],
                    x,
                    y,
                    z - obj["dimensions_metres"][2] / 2,
                    scale,
                    "front" if obj["instance_id"] % 2 else "back",
                )
            )
        # Procedural tape, printed labels and barcodes are removed together with
        # their old body; the registered mesh/atlas already defines its surfaces.
    cases = (
        (
            "mesh_empty_equipment",
            "Empty foreground transport equipment with source-backed rack cartons",
            [],
        ),
        (
            "mesh_pallet_load",
            "Uniformly scaled carton on background pallet",
            [(-0.85, -1.6, 0.22, 1.3, "front")],
        ),
        (
            "mesh_paired_load",
            "Two separate cartons form a wider pallet load",
            [(-1.1, -1.6, 0.22, 0.95, "front"), (-0.6, -1.6, 0.22, 0.95, "back")],
        ),
        (
            "mesh_container_occlusion",
            "Large floor carton partly occluded by empty handling container",
            [(1.1, 1.7, 0, 2.3, "back")],
        ),
    )
    captures = []
    for name, objective, loads in cases:
        capture = copy.deepcopy(base)
        capture.update(
            sample_id=name, scenario=objective, objects=copy.deepcopy(rack_objects)
        )
        next_id = max(obj["instance_id"] for obj in rack_objects) + 1
        for offset, (x, y, bottom, scale, variant) in enumerate(loads):
            capture["objects"].append(
                carton(next_id + offset, x, y, bottom, scale, variant)
            )
        captures.append(capture)
    job["scenes"] = captures
    job["asset_catalog_sha256"] = catalog_sha256
    job = bind_surfaces(
        job,
        catalog,
        catalog_sha256,
        dict(
            schema_version="dubaidata.surface-bindings/1.0",
            surfaces={
                "wood": dict(asset_id="polyhaven.wooden_planks.4k", base_color_mix=1.0),
                "concrete": dict(
                    asset_id="polyhaven.hangar_concrete_floor.4k", base_color_mix=0.7
                ),
            },
        ),
    )
    job["render"].update(view_transform="AgX", response_candidates=[])
    return validate_goods_scene(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("profile", "camera-config", "asset-catalog", "output"):
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
    job = build_mesh_job(
        json.loads(args.profile.read_text(encoding="utf-8")),
        file_hash(args.profile),
        camera,
        json.loads(args.asset_catalog.read_text(encoding="utf-8")),
        file_hash(args.asset_catalog),
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(job, stream, indent=2, allow_nan=False)
    print(
        json.dumps(dict(captures=len(job["scenes"]), job_sha256=file_hash(args.output)))
    )


if __name__ == "__main__":
    main()
