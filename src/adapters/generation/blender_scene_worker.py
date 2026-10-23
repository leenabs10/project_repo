"""Isolated Cycles worker for resolved portable scenes; run inside Blender only.

RGB, axial depth and discrete instance labels share one render and camera pose.
Depth convention is established by an analytic plane before customer scenes.
This worker deliberately does not emulate LiDAR or measured sensor noise.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

# Blender's Python is isolated from the data/ML environment. Reuse only the
# existing renderer's small engine helpers; core validation runs in the host.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "synthesis"))
from warehouse_renderer import box, configure_cycles, material
from renderer_geometry import axial_depth

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_assets import load_normalized_mesh


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def create_surface(name, specification, asset_root=None):
    """Translate bounded material intent without modifying mesh geometry or IDs.

    Detail uses world-space metres, so a larger object does not stretch the
    texture. Shader bump affects RGB shading only; exported depth remains the
    clean mesh surface. Emission strength is an authored rendering parameter,
    not a calibrated luminance measurement.
    """
    detail = specification.get("detail")
    surface = material(
        bpy,
        name,
        tuple(specification["color"]),
        roughness=specification["roughness"],
        metallic=specification["metallic"],
        texture=specification["texture"] and detail is None,
    )
    nodes, links = surface.node_tree.nodes, surface.node_tree.links
    shader = nodes.get("Principled BSDF")
    if specification.get("emission_strength", 0):
        shader.inputs["Emission Color"].default_value = (*specification["color"], 1)
        shader.inputs["Emission Strength"].default_value = specification[
            "emission_strength"
        ]
    if detail is not None:
        position = nodes.new("ShaderNodeNewGeometry")
        mapping = nodes.new("ShaderNodeVectorMath")
        mapping.operation = "MULTIPLY"
        mapping.inputs[1].default_value = detail["scale_per_metre"]
        links.new(position.outputs["Position"], mapping.inputs[0])
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 1
        noise.inputs["Detail"].default_value = 4
        noise.inputs["Roughness"].default_value = 0.7
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        ramp = nodes.new("ShaderNodeValToRGB")
        variation = detail["color_variation"]
        ramp.color_ramp.elements[0].color = (
            *[value * (1 - variation) for value in specification["color"]],
            1,
        )
        ramp.color_ramp.elements[1].color = (
            *[min(value * (1 + variation), 1) for value in specification["color"]],
            1,
        )
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
        roughness = nodes.new("ShaderNodeMapRange")
        roughness.inputs["To Min"].default_value = max(
            0, specification["roughness"] - detail["roughness_variation"]
        )
        roughness.inputs["To Max"].default_value = min(
            1, specification["roughness"] + detail["roughness_variation"]
        )
        links.new(noise.outputs["Fac"], roughness.inputs["Value"])
        links.new(roughness.outputs["Result"], shader.inputs["Roughness"])
        if detail["bump_metres"]:
            fine = nodes.new("ShaderNodeTexNoise")
            fine.inputs["Scale"].default_value = 80
            fine.inputs["Detail"].default_value = 2
            links.new(position.outputs["Position"], fine.inputs["Vector"])
            bump = nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = 0.18
            bump.inputs["Distance"].default_value = detail["bump_metres"]
            links.new(fine.outputs["Fac"], bump.inputs["Height"])
            links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    if "texture_set" in specification:
        texture = specification["texture_set"]
        if asset_root is None:
            raise ValueError("Texture root is missing in the isolated worker")
        position = nodes.new("ShaderNodeNewGeometry")
        scale = nodes.new("ShaderNodeVectorMath")
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = [
            1 / value for value in texture["tile_size_metres"]
        ]
        links.new(position.outputs["Position"], scale.inputs[0])
        if texture.get("projection", "BOX") == "UV":
            coordinates = nodes.new("ShaderNodeTexCoord")
            coordinate_socket = coordinates.outputs["UV"]
        else:
            coordinate_socket = scale.outputs["Vector"]
        images = {}
        for role, binding in texture["maps"].items():
            root = asset_root.resolve()
            path = (root / binding["path"]).resolve()
            if not path.is_relative_to(root) or digest(path) != binding["sha256"]:
                raise ValueError("Worker texture path/checksum verification failed")
            image = bpy.data.images.load(str(path), check_existing=True)
            image.colorspace_settings.name = binding["color_space"]
            node = nodes.new("ShaderNodeTexImage")
            node.image = image
            node.projection = "FLAT" if texture.get("projection") == "UV" else "BOX"
            node.projection_blend = 0.2
            node.extension = "REPEAT"
            links.new(coordinate_socket, node.inputs["Vector"])
            images[role] = node
        palette = nodes.new("ShaderNodeMixRGB")
        palette.blend_type = "MIX"
        palette.inputs[0].default_value = texture["base_color_mix"]
        palette.inputs[1].default_value = (*specification["color"], 1)
        links.new(images["base_color"].outputs["Color"], palette.inputs[2])
        links.new(palette.outputs["Color"], shader.inputs["Base Color"])
        links.new(images["roughness"].outputs["Color"], shader.inputs["Roughness"])
    return surface


def create_triangle_object(specification, surface, asset_root, cache):
    """Instantiate registered normalized geometry with unchanged UVs and IDs.

    Metric dimensions scale the normalized bounds. No remeshing, bevel modifier
    or displacement changes the registered shape. Instances share mesh data.
    """
    binding = specification["mesh"]
    key = (binding["sha256"], surface.name)
    if key not in cache:
        arrays = load_normalized_mesh(binding, asset_root)
        mesh = bpy.data.meshes.new(binding["asset_id"])
        mesh.from_pydata(arrays["vertices"].tolist(), [], arrays["triangles"].tolist())
        mesh.update()
        uv = mesh.uv_layers.new(name="UVMap")
        for polygon in mesh.polygons:
            polygon.use_smooth = True
            for loop_index in polygon.loop_indices:
                uv.data[loop_index].uv = arrays["uv"][
                    mesh.loops[loop_index].vertex_index
                ]
        mesh.materials.append(surface)
        cache[key] = mesh
    obj = bpy.data.objects.new(f"instance_{specification['instance_id']}", cache[key])
    bpy.context.collection.objects.link(obj)
    obj.location = specification["center_metres"]
    obj.scale = specification["dimensions_metres"]
    obj.pass_index = specification["instance_id"]
    return obj


def configure_camera(camera, optical_pose):
    """Translate optical +Z forward/+Y down to Blender -Z forward/+Y up."""
    from warehouse_renderer import configure_camera as legacy_camera

    fx, fy, cx, cy = camera["intrinsics"]
    specification = dict(
        width=camera["width"], height=camera["height"], fx=fx, fy=fy, cx=cx, cy=cy
    )
    # The retained helper analytically verifies the actual Blender view frustum.
    legacy_camera(bpy, bpy.context.scene, specification)
    conversion = np.diag([1.0, -1.0, -1.0, 1.0])
    bpy.context.scene.camera.matrix_world = Matrix(
        np.asarray(optical_pose) @ conversion
    )
    bpy.context.scene.camera.data.clip_start = camera["near_metres"]
    bpy.context.scene.camera.data.clip_end = camera["far_metres"]
    bpy.context.view_layer.update()
    return specification


def render_passes(folder, response_candidates=()):
    """Decode float EXRs without interpolating IDs or denoising depth/labels."""
    folder.mkdir()
    scene = bpy.context.scene
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
    for name, socket in (
        ("range", "Depth"),
        ("instances", "IndexOB"),
        ("scene_linear", "Image"),
    ):
        output.file_slots.new(name)
        scene.node_tree.links.new(layers.outputs[socket], output.inputs[name])
    scene.render.filepath = str(folder / "rgb.png")
    scene.frame_set(1)
    started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - started
    # Derive alternate display encodings from the SAME Render Result. Never
    # rerender geometry, tone-map depth or relabel a second independent image.
    primary = (scene.view_settings.view_transform, scene.view_settings.look)
    for transform in response_candidates:
        scene.view_settings.view_transform = transform
        scene.view_settings.look = "None"
        bpy.data.images["Render Result"].save_render(
            str(folder / f"rgb_{transform.lower()}.png"), scene=scene
        )
    scene.view_settings.view_transform, scene.view_settings.look = primary
    arrays = []
    for name in ("range", "instances"):
        paths = list((folder / "passes").glob(name + "*.exr"))
        if len(paths) != 1:
            raise ValueError("Missing or ambiguous renderer pass")
        image = bpy.data.images.load(str(paths[0]), check_existing=False)
        image.colorspace_settings.name = "Non-Color"
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        width, height = image.size
        arrays.append(
            np.flipud(pixels.reshape(height, width, image.channels)[:, :, 0]).copy()
        )
        bpy.data.images.remove(image)
    if not np.isfinite(arrays[1]).all() or not np.allclose(
        arrays[1], np.rint(arrays[1]), atol=1e-5
    ):
        raise ValueError("Instance pass must contain discrete, finite IDs")
    return arrays[0], np.rint(arrays[1]).astype(np.uint16), elapsed


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def geometry_check(camera, pose, depth, instances):
    """Check independent scene rays for metric depth and instance alignment.

    Ray agreement verifies correspondence to the rendered mesh, not appearance
    fidelity. Boundaries are excluded because RGB is antialiased and IDs are not.
    """
    pose = np.asarray(pose)
    fx, fy, cx, cy = camera["intrinsics"]
    rng = np.random.default_rng(9103)
    errors = []
    matches = []
    graph = bpy.context.evaluated_depsgraph_get()
    for _ in range(4000):
        u = int(rng.integers(2, camera["width"] - 2))
        v = int(rng.integers(2, camera["height"] - 2))
        identity = int(instances[v, u])
        if not identity or not np.all(
            instances[v - 2 : v + 3, u - 2 : u + 3] == identity
        ):
            continue
        direction = pose[:3, :3] @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        direction /= np.linalg.norm(direction)
        hit, location, _, _, obj, _ = bpy.context.scene.ray_cast(
            graph, Vector(pose[:3, 3]), Vector(direction), distance=camera["far_metres"]
        )
        if not hit:
            raise ValueError("Valid rendered pixel has no independent ray intersection")
        expected = float(np.dot(np.asarray(location) - pose[:3, 3], pose[:3, 2]))
        errors.append(abs(float(depth[v, u]) - expected))
        matches.append(obj.pass_index == identity)
        if len(errors) == 256:
            break
    if (
        len(errors) < 128
        or np.quantile(errors, 0.95) > 0.005
        or np.mean(matches) < 0.99
    ):
        raise ValueError("Depth/instance geometry checks failed")
    return dict(
        interior_rays=len(errors),
        depth_error_metres_p50_p95_max=np.quantile(errors, [0.5, 0.95, 1]).tolist(),
        instance_agreement=float(np.mean(matches)),
    )


def main():
    """Render to a fresh worker directory after the host has validated the job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--job-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if digest(args.job) != args.job_sha256:
        raise ValueError("Resolved scene job changed before worker execution")
    job = json.loads(args.job.read_text(encoding="utf-8"))
    if job["backend"] != "blender_cycles":
        raise ValueError("Backend does not match this worker")
    args.output.mkdir(parents=True, exist_ok=False)
    clear_scene()
    scene = bpy.context.scene
    config = dict(job["render"], adaptive_threshold=0.01)
    device = configure_cycles(bpy, scene, config)
    if "view_transform" in job["render"]:
        scene.view_settings.view_transform = job["render"]["view_transform"]
        scene.view_settings.look = "None"
        scene.view_settings.gamma = 1
    camera = job["camera"]
    fixture_pose = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 1.35], [0, 0, 0, 1]]
    specification = configure_camera(camera, fixture_pose)
    surface = material(bpy, "Analytic surface", (0.5, 0.5, 0.5))
    box(
        bpy,
        "Analytic front surface",
        (0, 10.01, 1.35),
        (100, 0.02, 100),
        surface,
        1,
        bevel=0,
    )
    scene.cycles.samples = 16
    raw, identities, fixture_seconds = render_passes(args.output / "analytic_fixture")
    hypotheses = {}
    for convention in ("euclidean_ray_range", "axial_z"):
        depth, valid = axial_depth(
            raw, specification, convention, far_clip_metres=camera["far_metres"]
        )
        hypotheses[convention] = (
            float(np.max(np.abs(depth - 10))) if valid.all() else None
        )
    passing = [
        key for key, error in hypotheses.items() if error is not None and error < 0.0005
    ]
    if len(passing) != 1 or not np.all(identities == 1):
        raise ValueError(f"Analytic depth/label fixture failed: {hypotheses}")
    convention = passing[0]
    records = []
    for capture in job["scenes"]:
        clear_scene()
        scene.cycles.samples = job["render"]["samples"]
        surfaces = {
            name: create_surface(
                f"{capture['sample_id']}.{name}", value, args.asset_root
            )
            for name, value in job["materials"].items()
        }
        mesh_cache = {}
        for obj in capture["objects"]:
            if "mesh" in obj:
                create_triangle_object(
                    obj, surfaces[obj["material"]], args.asset_root, mesh_cache
                )
                continue
            box(
                bpy,
                f"instance_{obj['instance_id']}",
                obj["center_metres"],
                obj["dimensions_metres"],
                surfaces[obj["material"]],
                obj["instance_id"],
                bevel=obj["bevel_metres"],
            )
        for index, lamp in enumerate(capture["lights"]):
            light = bpy.data.lights.new(f"Ceiling light {index}", "AREA")
            light.energy = lamp["power_watts"]
            light.size = lamp["size_metres"]
            lamp_object = bpy.data.objects.new(light.name, light)
            bpy.context.collection.objects.link(lamp_object)
            lamp_object.location = lamp["center_metres"]
        configure_camera(camera, capture["optical_camera_to_world"])
        folder = args.output / capture["sample_id"]
        raw, instances, elapsed = render_passes(
            folder, job["render"].get("response_candidates", [])
        )
        depth, valid = axial_depth(
            raw, specification, convention, far_clip_metres=camera["far_metres"]
        )
        lookup = np.zeros(
            max(obj["instance_id"] for obj in capture["objects"]) + 1, dtype=np.uint8
        )
        for obj in capture["objects"]:
            lookup[obj["instance_id"]] = obj["class_id"]
        if instances.max() >= len(lookup) or not np.array_equal(instances > 0, valid):
            raise ValueError(
                "Rendered instances disagree with scene membership or depth validity"
            )
        semantic = lookup[instances]
        check = geometry_check(
            camera, capture["optical_camera_to_world"], depth, instances
        )
        depth = np.where(valid, depth, 0).astype(np.float32)
        for name, array in (
            ("depth_axial_m", depth),
            ("depth_valid", valid),
            ("semantic_ids", semantic),
            ("instance_ids", instances),
        ):
            np.save(folder / f"{name}.npy", array, allow_pickle=False)
        records.append(
            dict(
                sample_id=capture["sample_id"],
                rgb_response=dict(
                    primary=scene.view_settings.view_transform,
                    candidates=job["render"].get("response_candidates", []),
                    look=scene.view_settings.look,
                    exposure_ev=scene.view_settings.exposure,
                    gamma=scene.view_settings.gamma,
                    display_device=scene.display_settings.display_device,
                    shared_render=True,
                    calibrated_camera_response=False,
                ),
                geometry=check,
                render_seconds=elapsed,
                artifact_sha256={
                    path.relative_to(args.output).as_posix(): digest(path)
                    for path in sorted(folder.rglob("*"))
                    if path.is_file()
                },
            )
        )
    write_json(
        args.output / "worker_profile.json",
        dict(
            status="rendered_geometry_passed",
            job_sha256=args.job_sha256,
            backend="blender_cycles",
            blender_version=bpy.app.version_string,
            device=device,
            fixture=dict(
                depth_convention=convention,
                maximum_error_metres=hypotheses[convention],
                hypotheses=hypotheses,
                seconds=fixture_seconds,
            ),
            records=records,
            training_eligible=False,
            physical_sensor_realism_certified=False,
        ),
    )


if __name__ == "__main__":
    main()
