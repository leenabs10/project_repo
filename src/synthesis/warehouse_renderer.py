"""Blender Cycles pilot: paired native RGB, clean axial depth and semantic IDs.

Run with portable Blender --background --factory-startup --python this_file --
--config configs/warehouse_renderer.json --root data/raw/torwic_slam --output
reports/warehouse_renderer_pilot. No TorWIC trajectory images or poses are read.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renderer_geometry import axial_depth, ray_lengths, semantic_ids


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def material(bpy, name, color, *, roughness=0.6, metallic=0.0, texture=False):
    """Create opaque PBR material; procedural texture changes normals, not labels."""
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    nodes = result.node_tree.nodes
    links = result.node_tree.links
    shader = nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    if texture:
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 65
        noise.inputs["Detail"].default_value = 3
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*[channel * 0.82 for channel in color], 1)
        ramp.color_ramp.elements[1].color = (
            *[min(channel * 1.12, 1) for channel in color],
            1,
        )
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.16
        bump.inputs["Distance"].default_value = 0.002
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return result


def box(bpy, name, center, dimensions, surface, class_id, *, bevel=0.008):
    """Instance a metric cuboid mesh, retaining per-object materials and labels.

    Repeated beams, tape and barcodes share geometry. Direct datablock creation
    avoids repeated operator/dependency-graph updates as the scene grows. Mesh
    vertices carry metric dimensions, so bevel radii remain metres under scale 1.
    """
    dimensions = np.asarray(dimensions, dtype=float)
    if (
        dimensions.shape != (3,)
        or not np.isfinite(dimensions).all()
        or np.any(dimensions <= 0)
    ):
        raise ValueError("Cuboid dimensions must be three positive finite metres")
    key = "Metric cuboid " + hashlib.sha256(dimensions.tobytes()).hexdigest()[:16]
    mesh = bpy.data.meshes.get(key)
    if mesh is None:
        x, y, z = dimensions / 2
        vertices = [
            (-x, -y, -z),
            (-x, -y, z),
            (-x, y, -z),
            (-x, y, z),
            (x, -y, -z),
            (x, -y, z),
            (x, y, -z),
            (x, y, z),
        ]
        # Outward winding supports physically correct normals and lighting.
        faces = [
            (0, 2, 6, 4),
            (1, 5, 7, 3),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 1, 3, 2),
            (4, 6, 7, 5),
        ]
        mesh = bpy.data.meshes.new(key)
        mesh.from_pydata(vertices, [], faces)
        mesh.materials.append(surface)
        mesh.update()
    result = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(result)
    result.location = center
    result.material_slots[0].link = "OBJECT"
    result.material_slots[0].material = surface
    result.pass_index = class_id
    if bevel:
        modifier = result.modifiers.new("Manufactured edge radius", "BEVEL")
        modifier.width = bevel
        modifier.segments = 2
    return result


def carton(bpy, name, center, size, paper, tape):
    """Create a labelled carton including its opaque packaging tape."""
    box(bpy, name, center, size, paper, 7, bevel=0.006)
    x, y, z = center
    width, length, height = size
    box(
        bpy,
        name + "_top_tape",
        (x, y, z + height / 2 + 0.0006),
        (0.075, length * 0.99, 0.001),
        tape,
        7,
        bevel=0,
    )
    box(
        bpy,
        name + "_front_tape",
        (x, y - length / 2 - 0.0006, z),
        (0.075, 0.001, height * 0.97),
        tape,
        7,
        bevel=0,
    )

    # Shipping labels are opaque parts of the goods surface, with the same ID.
    sticker = bpy.data.materials.get("Shipping label paper")
    ink = bpy.data.materials.get("Shipping label ink")
    if sticker is not None and ink is not None:
        label_x = x - width * 0.22
        label_z = z + height * 0.12
        box(
            bpy,
            name + "_shipping_label",
            (label_x, y - length / 2 - 0.0015, label_z),
            (0.19, 0.001, 0.12),
            sticker,
            7,
            bevel=0,
        )
        for index, strip_width in enumerate(
            (0.004, 0.009, 0.005, 0.003, 0.008, 0.006, 0.003)
        ):
            box(
                bpy,
                name + f"_barcode_{index}",
                (label_x - 0.064 + index * 0.017, y - length / 2 - 0.0022, label_z),
                (strip_width, 0.0005, 0.07),
                ink,
                7,
                bevel=0,
            )


def build_warehouse(bpy, seed):
    """Build one compact aisle in metres; background is fixed across scenarios."""
    rng = np.random.default_rng(seed)
    concrete = material(
        bpy, "Concrete", (0.36, 0.38, 0.39), roughness=0.83, texture=True
    )
    wall = material(
        bpy, "Painted wall", (0.62, 0.65, 0.67), roughness=0.76, texture=True
    )
    steel = material(
        bpy, "Galvanized steel", (0.32, 0.36, 0.40), roughness=0.32, metallic=0.72
    )
    orange = material(
        bpy, "Rack safety orange", (0.64, 0.12, 0.028), roughness=0.42, metallic=0.25
    )
    wood = material(
        bpy, "Pallet wood", (0.37, 0.24, 0.12), roughness=0.82, texture=True
    )
    tape = material(bpy, "Packaging tape", (0.49, 0.32, 0.15), roughness=0.36)
    yellow = material(bpy, "Aisle marking", (0.68, 0.46, 0.045), roughness=0.73)
    roof = material(
        bpy, "Industrial roof", (0.055, 0.064, 0.075), roughness=0.55, metallic=0.55
    )
    material(bpy, "Shipping label paper", (0.74, 0.72, 0.65), roughness=0.85)
    material(bpy, "Shipping label ink", (0.012, 0.014, 0.016), roughness=0.8)
    luminaire = material(bpy, "LED diffuser", (0.8, 0.86, 0.94), roughness=0.3)
    emission = luminaire.node_tree.nodes.get("Principled BSDF")
    emission.inputs["Emission Color"].default_value = (0.8, 0.86, 0.94, 1)
    emission.inputs["Emission Strength"].default_value = 4
    papers = [
        material(
            bpy,
            f"Corrugated paper {index}",
            (0.40 + index * 0.024, 0.25 + index * 0.018, 0.12 + index * 0.012),
            roughness=0.83,
            texture=True,
        )
        for index in range(5)
    ]
    box(bpy, "Floor", (0, 6, -0.08), (12, 30, 0.16), concrete, 1)
    box(bpy, "Back wall", (0, 21, 3), (12, 0.16, 6), wall, 4)
    for side in (-1, 1):
        box(bpy, f"Side wall {side}", (side * 6, 6, 3), (0.16, 30, 6), wall, 4)
        box(
            bpy,
            f"Floor boundary {side}",
            (side * 1.7, 6, 0.002),
            (0.075, 28, 0.003),
            yellow,
            1,
            bevel=0,
        )
    box(bpy, "Ceiling", (0, 6, 6.04), (12, 30, 0.08), roof, 2)
    for index, y in enumerate(np.arange(-7, 22, 2.8)):
        box(
            bpy,
            f"Roof crossbeam {index}",
            (0, float(y), 5.96),
            (12, 0.085, 0.15),
            steel,
            5,
            bevel=0.003,
        )
    for side in (-1, 1):
        for bay, y in enumerate((0, 4, 8, 12, 16)):
            x = side * 3.1
            for offset_x in (-0.66, 0.66):
                for offset_y in (-1.74, 1.74):
                    box(
                        bpy,
                        f"Upright {side} {bay} {offset_x} {offset_y}",
                        (x + offset_x, y + offset_y, 1.95),
                        (0.075, 0.075, 3.9),
                        steel,
                        6,
                        bevel=0.003,
                    )
            for level, height in enumerate((0.48, 1.62, 2.78)):
                box(
                    bpy,
                    f"Shelf {side} {bay} {level}",
                    (x, y, height),
                    (1.4, 3.5, 0.07),
                    wood,
                    6,
                )
                for offset_x in (-0.68, 0.68):
                    box(
                        bpy,
                        f"Beam {side} {bay} {level} {offset_x}",
                        (x + offset_x, y, height - 0.04),
                        (0.07, 3.55, 0.17),
                        orange,
                        6,
                    )
                for item, offset_y in enumerate((-1.14, 0, 1.14)):
                    size = (
                        float(rng.uniform(0.7, 1.04)),
                        float(rng.uniform(0.7, 1.0)),
                        float(rng.uniform(0.45, 0.8)),
                    )
                    carton(
                        bpy,
                        f"Stock {side} {bay} {level} {item}",
                        (x, y + offset_y, height + 0.035 + size[2] / 2),
                        size,
                        papers[int(rng.integers(len(papers)))],
                        tape,
                    )
    # Light fixtures and area lights use the same geometry in every scenario.
    for index, y in enumerate((-4, 2, 8, 14, 19)):
        box(bpy, f"Luminaire {index}", (0, y, 5.82), (1.8, 0.45, 0.07), steel, 5)
        box(
            bpy,
            f"LED diffuser {index}",
            (0, y, 5.775),
            (1.6, 0.33, 0.01),
            luminaire,
            5,
            bevel=0.003,
        )
        light = bpy.data.lights.new(f"Ceiling light {index}", "AREA")
        light.energy = 1500
        light.shape = "RECTANGLE"
        light.size = 1.65
        light.size_y = 0.4
        obj = bpy.data.objects.new(light.name, light)
        bpy.context.collection.objects.link(obj)
        obj.location = (0, y, 5.76)
    return papers, tape


def configure_camera(bpy, scene, specification):
    """Match declared OpenCV intrinsics and record optical-camera-to-world pose."""
    from mathutils import Matrix

    camera = bpy.data.cameras.new("Synthetic left pinhole camera")
    obj = bpy.data.objects.new(camera.name, camera)
    bpy.context.collection.objects.link(obj)
    scene.camera = obj
    width, height = specification["width"], specification["height"]
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1
    aspect = specification["fx"] / specification["fy"]
    scene.render.pixel_aspect_y = aspect
    camera.sensor_fit = "HORIZONTAL"
    camera.sensor_width = 36
    camera.lens = specification["fx"] * camera.sensor_width / width
    camera.shift_x = ((width - 1) / 2 - specification["cx"]) / width
    camera.shift_y = (specification["cy"] - (height - 1) / 2) * aspect / width
    camera.clip_start = 0.05
    camera.clip_end = 60
    camera.dof.use_dof = False
    # Blender local -Z is forward and +Y is up; optical +Z forward, +Y down.
    obj.matrix_world = Matrix(
        ((1, 0, 0, 0), (0, 0, -1, -7), (0, 1, 0, 1.35), (0, 0, 0, 1))
    )
    bpy.context.view_layer.update()
    corners = np.array([tuple(vector) for vector in camera.view_frame(scene=scene)])
    horizontal = corners[:, 0] / -corners[:, 2]
    vertical = corners[:, 1] / -corners[:, 2]
    measured_fx = width / np.ptp(horizontal)
    measured_fy = height / np.ptp(vertical)
    measured = {
        "fx": measured_fx,
        "fy": measured_fy,
        "cx": -horizontal.min() * measured_fx - 0.5,
        "cy": vertical.max() * measured_fy - 0.5,
    }
    for key, value in measured.items():
        if abs(value - specification[key]) > 0.0002:
            raise ValueError(
                f"Blender camera does not implement declared {key}: {value}"
            )
    optical_pose = np.array(obj.matrix_world)
    optical_pose[:3, :3] = optical_pose[:3, :3] @ np.diag([1, -1, -1])
    return optical_pose


def configure_cycles(bpy, scene, config):
    """Select available GPU rendering explicitly and report any CPU fallback."""
    scene.render.engine = "CYCLES"
    scene.cycles.samples = config["samples"]
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = config["adaptive_threshold"]
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 8
    scene.cycles.seed = config["seed"]
    scene.render.use_motion_blur = False
    preferences = bpy.context.preferences.addons["cycles"].preferences
    selected = []
    backend = "CPU"
    for candidate in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = candidate
            preferences.get_devices()
            selected = [
                device.name
                for device in preferences.devices
                if device.type == candidate
            ]
            if selected:
                for device in preferences.devices:
                    device.use = device.type == candidate
                backend = candidate
                scene.cycles.device = "GPU"
                break
        except (TypeError, RuntimeError):
            continue
    if not selected:
        scene.cycles.device = "CPU"
        selected = ["CPU fallback"]
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = config.get("exposure_ev", 0.0)
    world = bpy.data.worlds.new("Warehouse ambient")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.75,
        0.8,
        0.9,
        1,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.15
    scene.world = world
    return {
        "backend": backend,
        "devices": selected,
        "rgb_denoising": True,
        "labels_and_depth_denoised": False,
    }


def render_passes(bpy, scene, folder):
    """Save lossless RGB and independent float EXR passes, then decode top-left arrays."""
    folder.mkdir(parents=True)
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_object_index = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(layers.outputs["Image"], composite.inputs["Image"])
    output = nodes.new("CompositorNodeOutputFile")
    output.base_path = str(folder / "passes")
    output.format.file_format = "OPEN_EXR"
    output.format.color_depth = "32"
    output.format.color_mode = "RGB"
    output.format.exr_codec = "ZIP"
    output.file_slots.clear()
    for name, socket in (("range", "Depth"), ("classes", "IndexOB")):
        output.file_slots.new(name)
        scene.node_tree.links.new(layers.outputs[socket], output.inputs[name])
    scene.render.filepath = str(folder / "rgb.png")
    scene.frame_set(1)
    started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - started
    arrays = []
    for name in ("range", "classes"):
        paths = list((folder / "passes").glob(name + "*.exr"))
        if len(paths) != 1:
            raise ValueError(f"Missing or ambiguous {name} EXR export")
        image = bpy.data.images.load(str(paths[0]), check_existing=False)
        image.colorspace_settings.name = "Non-Color"
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        width, height = image.size
        arrays.append(
            np.flipud(pixels.reshape(height, width, image.channels)[:, :, 0]).copy()
        )
        bpy.data.images.remove(image)
    return arrays[0], semantic_ids(arrays[1]), elapsed


def ray_review(bpy, scene, camera, pose, depth, labels, seed):
    """Compare stable interior pixels against independent scene ray intersections."""
    from mathutils import Vector

    rng = np.random.default_rng(seed)
    errors, matches = [], []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for _ in range(2000):
        u = int(rng.integers(2, camera["width"] - 2))
        v = int(rng.integers(2, camera["height"] - 2))
        neighborhood = labels[v - 2 : v + 3, u - 2 : u + 3]
        if not np.all(neighborhood == labels[v, u]) or not np.isfinite(depth[v, u]):
            continue
        local = np.array(
            [(u - camera["cx"]) / camera["fx"], (v - camera["cy"]) / camera["fy"], 1.0]
        )
        direction = pose[:3, :3] @ local
        direction /= np.linalg.norm(direction)
        hit, location, _, _, obj, _ = scene.ray_cast(
            depsgraph, Vector(pose[:3, 3]), Vector(direction), distance=60
        )
        if not hit:
            raise ValueError("Rendered valid depth has no geometric ray intersection")
        axial = float(np.dot(np.asarray(location) - pose[:3, 3], pose[:3, 2]))
        errors.append(abs(float(depth[v, u]) - axial))
        matches.append(obj.pass_index == int(labels[v, u]))
        if len(errors) == 256:
            break
    if len(errors) < 128:
        raise ValueError("Too few independent interior geometry checks")
    report = {
        "interior_rays": len(errors),
        "depth_abs_error_metres_p50_p95_max": np.quantile(
            errors, [0.5, 0.95, 1]
        ).tolist(),
        "semantic_agreement": float(np.mean(matches)),
        "boundary_pixels_excluded": True,
    }
    if (
        report["depth_abs_error_metres_p50_p95_max"][1] > 0.005
        or report["semantic_agreement"] < 0.99
    ):
        raise ValueError(f"Renderer geometry/semantic alignment failed: {report}")
    return report


def main():
    """Render an analytic fixture first, then three controlled aisle configurations."""
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(arguments)
    root, output = args.root.resolve(), args.output.resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError("Choose a fresh renderer output outside immutable raw data")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    renderer_hash = file_hash(Path(__file__))
    geometry_hash = file_hash(Path(__file__).parent / "renderer_geometry.py")
    config_hash = file_hash(args.config)
    if file_hash(root / "calibrations.txt") != config["reference_calibration_sha256"]:
        raise ValueError("Reference calibration changed")
    camera = config["camera"]
    ray_lengths(camera)
    output.mkdir(parents=True)
    records = []
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    device = configure_cycles(bpy, scene, config)
    pose = configure_camera(bpy, scene, camera)
    surface = material(bpy, "Analytic plane", (0.5, 0.5, 0.5))
    plane = box(
        bpy, "Analytic front surface", (0, 3, 1.35), (50, 0.02, 50), surface, 7, bevel=0
    )
    depth_pass, labels, fixture_time = render_passes(
        bpy, scene, output / "analytic_fixture"
    )
    expected_axial = 9.99
    hypotheses = {}
    for convention in ("euclidean_ray_range", "axial_z"):
        depth, valid = axial_depth(depth_pass, camera, convention, far_clip_metres=60)
        hypotheses[convention] = (
            float(np.max(np.abs(depth - expected_axial))) if valid.all() else None
        )
    passing = [
        name
        for name, error in hypotheses.items()
        if error is not None and error < 0.002
    ]
    if len(passing) != 1 or not np.all(labels == 7):
        raise ValueError(f"Analytic fixture failed: {hypotheses}")
    convention = passing[0]
    bpy.data.objects.remove(plane, do_unlink=True)
    assembly_started = time.perf_counter()
    papers, tape = build_warehouse(bpy, config["seed"])
    assembly_seconds = time.perf_counter() - assembly_started
    for specification in config["scenarios"]:
        added = []
        for index, item in enumerate(specification["goods"]):
            previous = set(bpy.data.objects)
            carton(
                bpy,
                f"Scenario goods {index}",
                item["center_metres"],
                item["size_metres"],
                papers[2],
                tape,
            )
            added.extend(set(bpy.data.objects) - previous)
        folder = output / specification["name"]
        depth_pass, labels, elapsed = render_passes(bpy, scene, folder)
        depth, valid = axial_depth(depth_pass, camera, convention, far_clip_metres=60)
        if np.any((labels == 0) & valid) or np.any((labels != 0) & ~valid):
            raise ValueError("Depth validity and object labels disagree")
        ray_checks = ray_review(bpy, scene, camera, pose, depth, labels, config["seed"])
        np.save(folder / "depth_axial_m.npy", depth, allow_pickle=False)
        np.save(folder / "depth_valid.npy", valid, allow_pickle=False)
        np.save(folder / "semantic_ids.npy", labels, allow_pickle=False)
        record = {
            "sample_id": specification["name"],
            "source": "procedural_blender_warehouse",
            "split": "synthetic_pilot_only",
            "training_eligible": False,
            "camera": {
                **camera,
                "D": [0, 0, 0, 0, 0],
                "convention": "synthetic_pinhole",
            },
            "optical_camera_to_world": pose.tolist(),
            "world_units": "metres",
            "depth_units": "axial_metres",
            "semantic_source": "renderer_geometry_class_indices",
            "goods_class_id": 7,
            "seed": config["seed"],
            "scenario": specification,
            "render_seconds_including_export": elapsed,
            "geometry_checks": ray_checks,
            "valid_depth_fraction": float(valid.mean()),
            "renderer_depth_pass_convention": convention,
            "reference_calibration_sha256": config["reference_calibration_sha256"],
            "config_sha256": config_hash,
            "code_sha256": renderer_hash,
            "geometry_code_sha256": geometry_hash,
            "files": {
                name: {"path": name, "sha256": file_hash(folder / name)}
                for name in (
                    "rgb.png",
                    "depth_axial_m.npy",
                    "depth_valid.npy",
                    "semantic_ids.npy",
                )
            },
            "physical_sensor_realism_certified": False,
            "limitations": [
                "Procedural engineering scene; appearance requires visual/domain review.",
                "Clean geometry depth has no calibrated Azure failure/noise model.",
                "No spinning LiDAR or IMU simulation.",
                "Reference K is used only for synthetic FOV; real exported-image distortion remains unresolved.",
            ],
        }
        (folder / "metadata.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "rendered": specification["name"],
                    "seconds": elapsed,
                    "geometry": ray_checks,
                }
            ),
            flush=True,
        )
        if specification == config["scenarios"][-1]:
            bpy.ops.wm.save_as_mainfile(filepath=str(output / "warehouse_scene.blend"))
        for obj in added:
            bpy.data.objects.remove(obj, do_unlink=True)
    report = {
        "status": "rendered_geometry_checks_passed",
        "samples": len(records),
        "blender_version": bpy.app.version_string,
        "device": device,
        "scene_assembly_seconds": assembly_seconds,
        "geometry_reuse": "Shared metric cuboid meshes with per-object material and class IDs",
        "analytic_fixture": {
            "expected_axial_metres": expected_axial,
            "hypothesis_max_error_metres": hypotheses,
            "verified_depth_pass_convention": convention,
            "render_seconds_including_export": fixture_time,
        },
        "records": records,
        "raw_trajectory_image_files_opened": 0,
        "physical_sensor_realism_certified": False,
        "training_eligible": False,
    }
    if (
        renderer_hash != file_hash(Path(__file__))
        or geometry_hash != file_hash(Path(__file__).parent / "renderer_geometry.py")
        or config_hash != file_hash(args.config)
    ):
        raise ValueError(
            "Renderer code/config changed during the run; do not accept its provenance"
        )
    (output / "render_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "samples": len(records),
                "blender_version": bpy.app.version_string,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
