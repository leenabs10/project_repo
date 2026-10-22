"""Resolve a small warehouse pilot into portable, fully specified geometry.

The scene contains identifiable cartons and transport equipment with distinct
labels. No evaluation image, pose or mask is used to construct the scene.
Built-in procedural assets avoid external licensing/download dependencies.
"""

import argparse
import copy
import json
from pathlib import Path
import random

from src.adapters.storage.local import file_hash
from src.synthesis.warehouse_scenarios import validate_coverage
from src.adapters.tasks.goods_scene import validate_goods_scene


def build_job(profile, profile_sha256, camera, *, coverage=None):
    """Resolve geometry, materials and optical poses from training-only evidence.

    ``profile_sha256`` binds the supplied profile; ``camera`` declares native
    dimensions, pinhole intrinsics and metric clip planes. Optional ``coverage``
    selects bounded authored cases. The returned job contains all sampled values
    and is validated before any simulator process or output allocation.
    """
    if profile.get("split") != "train" or profile.get("status") != "passed":
        raise ValueError("A passing training-only customer profile is required")
    if coverage is not None:
        validate_coverage(coverage)
    materials = {}
    for name, color, roughness, metallic in (
        ("concrete", [0.36, 0.38, 0.39], 0.83, 0),
        ("wall", [0.62, 0.65, 0.67], 0.76, 0),
        ("steel", [0.27, 0.31, 0.35], 0.36, 0.65),
        ("orange", [0.64, 0.12, 0.028], 0.48, 0.2),
        ("wood", [0.37, 0.24, 0.12], 0.82, 0),
        ("paper", [0.47, 0.31, 0.16], 0.83, 0),
        ("tape", [0.49, 0.32, 0.15], 0.4, 0),
        ("label", [0.8, 0.78, 0.72], 0.82, 0),
        ("ink", [0.015, 0.017, 0.019], 0.8, 0),
        ("yellow", [0.68, 0.46, 0.045], 0.73, 0),
    ):
        materials[name] = dict(
            color=color,
            roughness=roughness,
            metallic=metallic,
            texture=name in ("concrete", "wall", "wood", "paper"),
        )
    # A local RNG resolves every variation here, never independently in an engine.
    # Dimensions are authored coverage ranges, not measured warehouse dimensions.
    rng = random.Random(1729)
    for index, color in enumerate(
        (
            [0.32, 0.20, 0.10],
            [0.42, 0.28, 0.15],
            [0.56, 0.41, 0.25],
            [0.64, 0.54, 0.40],
        )
    ):
        materials[f"carton_{index}"] = dict(
            color=color, roughness=0.82, metallic=0, texture=True
        )
    materials["container_blue"] = dict(
        color=[0.10, 0.20, 0.28], roughness=0.58, metallic=0.25, texture=False
    )
    materials["roof"] = dict(
        color=[0.16, 0.18, 0.20], roughness=0.6, metallic=0.45, texture=False
    )
    materials["diffuser"] = dict(
        color=[0.88, 0.93, 1.0],
        roughness=0.3,
        metallic=0,
        texture=False,
        emission_strength=25.0,
    )
    for name in (
        "concrete",
        "wall",
        "wood",
        "paper",
        "carton_0",
        "carton_1",
        "carton_2",
        "carton_3",
    ):
        materials[name]["detail"] = dict(
            scale_per_metre=([1.5, 30.0, 30.0] if name == "wood" else [0.8, 0.8, 0.8]),
            color_variation=(0.25 if name in ("concrete", "wood") else 0.14),
            roughness_variation=0.08,
            bump_metres=(0.0008 if name == "concrete" else 0.0003),
        )
    materials["concrete"]["roughness"] = 0.68
    objects = []

    def box(center, dimensions, material, role="structure", bevel=0.003):
        objects.append(
            dict(
                instance_id=len(objects) + 1,
                asset_id="builtin.metric_cuboid/1.0",
                role=role,
                class_id=int(role == "goods"),
                center_metres=list(center),
                dimensions_metres=list(dimensions),
                material=material,
                bevel_metres=min(bevel, min(dimensions) / 4),
            )
        )

    def carton(x, y, bottom, width=0.72, length=0.75, height=0.65, paper="paper"):
        box(
            (x, y, bottom + height / 2),
            (width, length, height),
            paper,
            "goods",
            0.006,
        )
        box(
            (x, y, bottom + height + 0.0006),
            (0.06, length * 0.98, 0.001),
            "tape",
            "goods",
            0,
        )
        box(
            (x, y - length / 2 - 0.0006, bottom + height / 2),
            (0.06, 0.001, height * 0.97),
            "tape",
            "goods",
            0,
        )
        box(
            (x - width * 0.22, y - length / 2 - 0.0015, bottom + height * 0.6),
            (0.16, 0.001, 0.1),
            "label",
            "goods",
            0,
        )
        for strip in range(5):
            box(
                (
                    x - width * 0.22 - 0.055 + strip * 0.024,
                    y - length / 2 - 0.0022,
                    bottom + height * 0.6,
                ),
                (0.004 + (strip % 2) * 0.003, 0.0005, 0.065),
                "ink",
                "goods",
                0,
            )

    def pallet(x, y):
        # Open gaps and separate timber parts make this visibly empty equipment.
        for offset in (-0.4, 0, 0.4):
            box((x + offset, y, 0.09), (0.12, 1.0, 0.18), "wood", "equipment")
        for offset in (-0.44, -0.22, 0, 0.22, 0.44):
            box((x, y + offset, 0.2), (1.2, 0.15, 0.04), "wood", "equipment")

    box((0, 5, -0.08), (12, 28, 0.16), "concrete")
    box((0, 19, 3), (12, 0.16, 6), "wall")
    box((0, 5, 6.04), (12, 28, 0.08), "roof")
    for y in (-6, -3, 0, 3, 6, 9, 12, 15, 18):
        box((0, y, 5.90), (12, 0.10, 0.20), "steel")
    for y in (-4, 2, 8, 14):
        # Opaque luminaires are background geometry with visible emitting panels.
        # Their area lights remain separate, explicit illumination parameters.
        box((0, y, 5.72), (1.8, 0.48, 0.10), "steel", "equipment")
        box((0, y, 5.663), (1.6, 0.34, 0.014), "diffuser", "equipment", 0.002)
    for side in (-1, 1):
        box((side * 6, 5, 3), (0.16, 28, 6), "wall")
        box((side * 1.8, 5, 0.002), (0.07, 27, 0.003), "yellow", bevel=0)
        for y in (0, 4, 8, 12, 16):
            x = side * 3.1
            for dx in (-0.66, 0.66):
                for dy in (-1.7, 1.7):
                    box((x + dx, y + dy, 1.95), (0.075, 0.075, 3.9), "steel")
            for z in (0.48, 1.62, 2.78):
                box((x, y, z), (1.4, 3.5, 0.07), "wood")
                for dx in (-0.68, 0.68):
                    box((x + dx, y, z - 0.04), (0.07, 3.55, 0.17), "orange")
                for dy in (-1.12, 0, 1.12):
                    # Gaps and unequal loads prevent a perfectly cloned rack face.
                    if rng.random() < 0.16:
                        continue
                    width = rng.uniform(0.58, 0.96)
                    length = rng.uniform(0.62, 0.92)
                    height = rng.uniform(0.34, 0.84)
                    paper = f"carton_{rng.randrange(4)}"
                    carton(
                        x + rng.uniform(-0.05, 0.05),
                        y + dy + rng.uniform(-0.06, 0.06),
                        z + 0.035,
                        width,
                        length,
                        height,
                        paper,
                    )
    pallet(-0.85, -1.6)
    pallet(1.0, 3.4)
    # Empty handling container: opaque base and walls, no inferred goods inside.
    box((1.1, 0.4, 0.12), (1.0, 1.1, 0.24), "container_blue", "equipment")
    for dx in (-0.49, 0.49):
        box((1.1 + dx, 0.4, 0.5), (0.02, 1.1, 0.52), "container_blue", "equipment")
    for dy in (-0.54, 0.54):
        box((1.1, 0.4 + dy, 0.5), (0.96, 0.02, 0.52), "container_blue", "equipment")
    base = copy.deepcopy(objects)
    scenes = []
    if coverage is None:
        cases = [
            dict(
                sample_id=name,
                objective=name,
                camera_x_metres=camera_x,
                loads=(
                    [
                        dict(
                            support="foreground_pallet",
                            center_xy_metres=[-0.85, -1.6],
                            dimensions_metres=[0.85, 0.8, 0.7],
                            material="paper",
                        ),
                        dict(
                            support="floor",
                            center_xy_metres=[0.8, 1.8],
                            dimensions_metres=[0.65, 0.65, 0.9],
                            material="paper",
                        ),
                    ]
                    if added
                    else []
                ),
            )
            for name, added, camera_x in (
                ("empty_transport", False, 0),
                ("loaded_transport", True, 0),
                ("offset_view", True, 0.45),
            )
        ]
    else:
        cases = coverage["cases"]
    for case in cases:
        name = case["sample_id"]
        camera_x = case["camera_x_metres"]
        objects[:] = copy.deepcopy(base)
        for load in case["loads"]:
            x, y = load["center_xy_metres"]
            bottom = 0.22 if load["support"] == "foreground_pallet" else 0.0
            carton(x, y, bottom, *load["dimensions_metres"], paper=load["material"])
        scenes.append(
            dict(
                sample_id=name,
                scenario=case["objective"],
                objects=copy.deepcopy(objects),
                lights=[
                    # Position the area source below the opaque diffuser so the
                    # fixture does not block its own illumination.
                    dict(
                        center_metres=[0, y, 5.645], power_watts=1500, size_metres=1.65
                    )
                    for y in (-4, 2, 8, 14)
                ],
                optical_camera_to_world=[
                    [1, 0, 0, camera_x],
                    [0, 0, 1, -6],
                    [0, -1, 0, 1.35],
                    [0, 0, 0, 1],
                ],
            )
        )
    job = dict(
        schema_version="dubaidata.scene/1.0",
        kind="generate_scene",
        job_id="warehouse_goods_scene_pilot",
        backend="blender_cycles",
        units="metres",
        world_axes="right_handed_x_right_y_forward_z_up",
        required_modalities=["rgb", "depth", "semantic", "instance"],
        conditioning=dict(
            manifest_sha256=profile["manifest_sha256"],
            profile_sha256=profile_sha256,
            profile_split="train",
            purpose="Goods versus empty transport equipment coverage",
        ),
        taxonomy={"0": "background", "1": "identifiable_goods"},
        camera=camera,
        render=dict(samples=128, seed=1729, exposure_ev=-1.4),
        materials=materials,
        scenes=scenes,
    )
    if coverage is not None:
        job["job_id"] = "warehouse_goods_scenario_coverage"
    return validate_goods_scene(job)


def main():
    """Create a fresh resolved plan from a training profile and declared camera."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-catalog", type=Path)
    parser.add_argument("--coverage-config", type=Path)
    args = parser.parse_args()
    configuration = json.loads(args.camera_config.read_text(encoding="utf-8"))
    source_camera = configuration["camera"]
    camera = dict(
        width=source_camera["width"],
        height=source_camera["height"],
        intrinsics=[source_camera[key] for key in ("fx", "fy", "cx", "cy")],
        near_metres=0.05,
        far_metres=60,
        model="pinhole_zero_distortion",
    )
    job = build_job(
        json.loads(args.profile.read_text(encoding="utf-8")),
        file_hash(args.profile),
        camera,
        coverage=(
            json.loads(args.coverage_config.read_text(encoding="utf-8"))
            if args.coverage_config
            else None
        ),
    )
    if args.asset_catalog:
        catalog = json.loads(args.asset_catalog.read_text(encoding="utf-8"))
        job["asset_catalog_sha256"] = file_hash(args.asset_catalog)
        for material_name, asset_id, mix in (
            ("wood", "polyhaven.wooden_planks.4k", 1.0),
            ("concrete", "polyhaven.hangar_concrete_floor.4k", 0.25),
        ):
            asset = catalog["assets"][asset_id]
            job["materials"][material_name]["texture_set"] = dict(
                asset_id=asset_id,
                tile_size_metres=asset["tile_size_metres"],
                base_color_mix=mix,
                maps={
                    role: {
                        key: binding[key] for key in ("path", "sha256", "color_space")
                    }
                    for role, binding in asset["maps"].items()
                },
            )
        # Keep the primary curve and exposure; the alternate is diagnostic,
        # not an inferred customer camera response or automatic qualification.
        job["render"].update(view_transform="AgX", response_candidates=["Standard"])
        validate_goods_scene(job)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(job, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {"job_sha256": file_hash(args.output), "captures": len(job["scenes"])}
        )
    )


if __name__ == "__main__":
    main()
