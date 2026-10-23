"""Portable metric scene contracts, independent of renderer and customer formats.

Scene 1.0 supports opaque axis-aligned cuboids. Scene 1.1 adds registered,
checksum-bound normalized triangle meshes with explicit UV materials.
Additional geometry and sensor capabilities require versioned extensions.
Unsupported requests are rejected.
"""

import json
import re
from pathlib import PurePosixPath

import numpy as np


def _keys(value, required, optional=()):
    """Reject misspelled or unsupported fields instead of silently dropping intent."""
    if not isinstance(value, dict) or set(required) - value.keys():
        raise ValueError(f"Missing required scene fields: {required}")
    if value.keys() - set(required) - set(optional):
        raise ValueError("Unsupported scene fields")


def _vector(value, size, *, positive=False):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError("Expected a finite scene vector")
    if positive and np.any(array <= 0):
        raise ValueError("Metric dimensions and focal lengths must be positive")
    return array


def validate_object_separation(
    objects, subject_role, obstacle_roles, *, penetration_tolerance_metres=0.0001
):
    """Reject declared cross-role penetrations while allowing surface contact.

    A task adapter chooses the protected roles. Composite parts within a role
    may overlap. This cuboid check is not a physics simulation.
    """
    solids = [obj for obj in objects if obj["role"] in obstacle_roles]
    if not solids:
        return
    centers = np.asarray([obj["center_metres"] for obj in solids])
    halves = np.asarray([obj["dimensions_metres"] for obj in solids]) / 2
    for goods in (obj for obj in objects if obj["role"] == subject_role):
        center = np.asarray(goods["center_metres"])
        half = np.asarray(goods["dimensions_metres"]) / 2
        # Compare actual intersection extents, including contained thin parts.
        overlap = np.minimum(centers + halves, center + half) - np.maximum(
            centers - halves, center - half
        )
        penetrating = np.all(overlap > penetration_tolerance_metres, axis=1)
        if penetrating.any():
            solid = solids[int(np.flatnonzero(penetrating)[0])]
            raise ValueError(
                f"{subject_role} instance {goods['instance_id']} penetrates {solid['role']} instance {solid['instance_id']}"
            )


def validate_scene_job(job):
    """Validate resolved geometry, optical poses, labels and bounded resource use."""
    json.dumps(job, allow_nan=False)
    _keys(
        job,
        (
            "schema_version",
            "kind",
            "job_id",
            "backend",
            "units",
            "world_axes",
            "required_modalities",
            "conditioning",
            "taxonomy",
            "camera",
            "render",
            "materials",
            "scenes",
        ),
        ("asset_catalog_sha256",),
    )
    if (
        job["schema_version"] not in {"dubaidata.scene/1.0", "dubaidata.scene/1.1"}
        or job["kind"] != "generate_scene"
        or job["units"] != "metres"
        or job["world_axes"] != "right_handed_x_right_y_forward_z_up"
    ):
        raise ValueError("Unsupported scene version, kind or coordinate convention")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", job["job_id"]):
        raise ValueError("Unsafe job identifier")
    if job["required_modalities"] != ["rgb", "depth", "semantic", "instance"]:
        raise ValueError(
            "This scene contract supports RGB, axial depth and visible labels only"
        )
    conditioning = job["conditioning"]
    _keys(
        conditioning, ("manifest_sha256", "profile_sha256", "profile_split", "purpose")
    )
    if conditioning["profile_split"] != "train":
        raise ValueError("Scene conditioning must use training data only")
    for key in ("manifest_sha256", "profile_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", conditioning[key]):
            raise ValueError("Conditioning evidence requires SHA256 bindings")
    taxonomy = job["taxonomy"]
    if (
        not isinstance(taxonomy, dict)
        or "0" not in taxonomy
        or len(taxonomy) > 255
        or any(
            not re.fullmatch(r"0|[1-9][0-9]{0,2}", key)
            or int(key) > 254
            or not isinstance(label, str)
            or not label.strip()
            for key, label in taxonomy.items()
        )
    ):
        raise ValueError("Taxonomy requires declared uint8 IDs; 255 remains reserved")
    camera = job["camera"]
    _keys(
        camera, ("width", "height", "intrinsics", "near_metres", "far_metres", "model")
    )
    if camera["model"] != "pinhole_zero_distortion":
        raise ValueError("Only explicitly undistorted pinhole cameras are supported")
    for dimension in ("width", "height"):
        if type(camera[dimension]) is not int or not 16 <= camera[dimension] <= 2048:
            raise ValueError("Camera resolution exceeds the bounded pilot contract")
    intrinsics = _vector(camera["intrinsics"], 4)
    if np.any(intrinsics[:2] <= 0) or not (
        0 <= intrinsics[2] < camera["width"] and 0 <= intrinsics[3] < camera["height"]
    ):
        raise ValueError("Invalid camera intrinsics")
    if not 0 < camera["near_metres"] < camera["far_metres"] <= 100:
        raise ValueError("Invalid metric clip range")
    render = job["render"]
    _keys(
        render,
        ("samples", "seed", "exposure_ev"),
        ("view_transform", "response_candidates"),
    )
    if render.get("view_transform", "AgX") not in ("AgX", "Standard"):
        raise ValueError("Unsupported display transform")
    candidates = render.get("response_candidates", [])
    if (
        not isinstance(candidates, list)
        or len(candidates) > 1
        or any(value not in ("AgX", "Standard") for value in candidates)
        or render.get("view_transform", "AgX") in candidates
    ):
        raise ValueError(
            "Response comparison must declare one distinct supported transform"
        )
    if type(render["samples"]) is not int or not 16 <= render["samples"] <= 512:
        raise ValueError("Render sample budget outside supported bounds")
    if type(render["seed"]) is not int or not 0 <= render["seed"] < 2**31:
        raise ValueError("Render seed must be an unsigned 31-bit integer")
    if not -5 <= render["exposure_ev"] <= 5:
        raise ValueError("Exposure outside bounded pilot range")
    if not 1 <= len(job["materials"]) <= 32:
        raise ValueError("Unsupported material budget")
    for surface in job["materials"].values():
        _keys(
            surface,
            ("color", "roughness", "metallic", "texture"),
            ("detail", "emission_strength", "texture_set"),
        )
        color = _vector(surface["color"], 3)
        if (
            np.any(color < 0)
            or np.any(color > 1)
            or not 0 <= surface["roughness"] <= 1
            or not 0 <= surface["metallic"] <= 1
            or type(surface["texture"]) is not bool
        ):
            raise ValueError("Invalid opaque material parameters")
        if not 0 <= surface.get("emission_strength", 0) <= 100:
            raise ValueError("Emission strength exceeds the bounded material model")
        if "detail" in surface:
            detail = surface["detail"]
            _keys(
                detail,
                (
                    "scale_per_metre",
                    "color_variation",
                    "roughness_variation",
                    "bump_metres",
                ),
            )
            scale = _vector(detail["scale_per_metre"], 3, positive=True)
            if (
                np.max(scale) > 500
                or not 0 <= detail["color_variation"] <= 0.35
                or not 0 <= detail["roughness_variation"] <= 0.2
                or not 0 <= detail["bump_metres"] <= 0.003
            ):
                raise ValueError("Unsupported metric material detail")
        if "texture_set" in surface:
            texture = surface["texture_set"]
            _keys(
                texture,
                ("asset_id", "tile_size_metres", "base_color_mix", "maps"),
                ("projection",),
            )
            if texture.get("projection", "BOX") not in {"BOX", "UV"}:
                raise ValueError("Unsupported material projection")
            if not isinstance(texture["asset_id"], str) or not texture["asset_id"]:
                raise ValueError("Texture requires a registered asset identifier")
            tile = _vector(texture["tile_size_metres"], 3, positive=True)
            if np.max(tile) > 100 or not 0 <= texture["base_color_mix"] <= 1:
                raise ValueError("Invalid metric texture scale or palette mix")
            _keys(texture["maps"], ("base_color", "roughness"))
            for role, mapping in texture["maps"].items():
                _keys(mapping, ("path", "sha256", "color_space"))
                relative = mapping["path"]
                portable = PurePosixPath(relative)
                if (
                    not relative
                    or "\\" in relative
                    or ":" in relative
                    or portable.is_absolute()
                    or ".." in portable.parts
                    or portable.suffix.lower() != ".png"
                ):
                    raise ValueError(
                        "Texture paths must be contained relative PNG references"
                    )
                if not re.fullmatch(r"[0-9a-f]{64}", mapping["sha256"]):
                    raise ValueError("Texture requires a SHA256 binding")
                if mapping["color_space"] != (
                    "sRGB" if role == "base_color" else "Non-Color"
                ):
                    raise ValueError(
                        "Texture role and color-space declaration disagree"
                    )
    textured = any("texture_set" in surface for surface in job["materials"].values())
    mesh_bound = any(
        "mesh" in obj for scene in job["scenes"] for obj in scene["objects"]
    )
    if (textured or mesh_bound) and not re.fullmatch(
        r"[0-9a-f]{64}", job.get("asset_catalog_sha256", "")
    ):
        raise ValueError("Textured scenes require an immutable asset catalog binding")
    if not 1 <= len(job["scenes"]) <= 8:
        raise ValueError("Pilot must contain one to eight captures")
    scene_ids = set()
    for scene in job["scenes"]:
        _keys(
            scene,
            ("sample_id", "scenario", "objects", "lights", "optical_camera_to_world"),
        )
        if (
            not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", scene["sample_id"])
            or scene["sample_id"] in scene_ids
        ):
            raise ValueError("Duplicate or unsafe sample identifier")
        scene_ids.add(scene["sample_id"])
        pose = np.asarray(scene["optical_camera_to_world"], dtype=float)
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-7)
        ):
            raise ValueError("Camera pose must be a rigid optical-to-world transform")
        if (
            not 1 <= len(scene["objects"]) <= 2000
            or not 1 <= len(scene["lights"]) <= 16
        ):
            raise ValueError("Unsupported scene object/light budget")
        identifiers = set()
        for obj in scene["objects"]:
            _keys(
                obj,
                (
                    "instance_id",
                    "asset_id",
                    "role",
                    "class_id",
                    "center_metres",
                    "dimensions_metres",
                    "material",
                    "bevel_metres",
                ),
                ("mesh",),
            )
            identifier = obj["instance_id"]
            if (
                type(identifier) is not int
                or not 1 <= identifier <= 65534
                or identifier in identifiers
            ):
                raise ValueError("Instances require unique nonzero uint16 IDs")
            identifiers.add(identifier)
            if obj["asset_id"] not in {
                "builtin.metric_cuboid/1.0",
                "registered.normalized_triangle_mesh/1.0",
            }:
                raise ValueError(
                    "Unsupported asset; register and qualify an adapter first"
                )
            is_mesh = obj["asset_id"] == "registered.normalized_triangle_mesh/1.0"
            if is_mesh:
                if (
                    job["schema_version"] != "dubaidata.scene/1.1"
                    or "mesh" not in obj
                    or obj["bevel_metres"] != 0
                ):
                    raise ValueError(
                        "Triangle assets require scene 1.1 and explicit unmodified mesh geometry"
                    )
                binding = obj["mesh"]
                _keys(binding, ("asset_id", "path", "sha256"))
                relative = binding["path"]
                portable = PurePosixPath(relative)
                if (
                    not relative
                    or "\\" in relative
                    or ":" in relative
                    or portable.is_absolute()
                    or ".." in portable.parts
                    or portable.suffix != ".npz"
                    or not re.fullmatch(r"[0-9a-f]{64}", binding["sha256"])
                ):
                    raise ValueError(
                        "Mesh requires a safe data-only asset path and SHA256 binding"
                    )
                texture = (
                    job["materials"].get(obj["material"], {}).get("texture_set", {})
                )
                if (
                    texture.get("projection") != "UV"
                    or texture.get("asset_id") != binding["asset_id"]
                ):
                    raise ValueError(
                        "Mesh geometry and UV material must bind the same registered asset"
                    )
            elif "mesh" in obj:
                raise ValueError("Cuboid geometry cannot carry an ignored mesh binding")
            if not isinstance(obj["role"], str) or not re.fullmatch(
                r"[a-zA-Z][a-zA-Z0-9_.-]*", obj["role"]
            ):
                raise ValueError("Object role must be an explicit portable identifier")
            if type(obj["class_id"]) is not int or str(obj["class_id"]) not in taxonomy:
                raise ValueError("Object class is absent from declared taxonomy")
            dimensions = _vector(obj["dimensions_metres"], 3, positive=True)
            center = _vector(obj["center_metres"], 3)
            if np.max(dimensions) > 100 or np.max(np.abs(center)) > 100:
                raise ValueError("Object exceeds metric scene bounds")
            if (
                obj["material"] not in job["materials"]
                or not 0 <= obj["bevel_metres"] < min(dimensions) / 2
            ):
                raise ValueError("Unknown material or invalid manufactured edge radius")
            local_camera = pose[:3, 3] - center
            if np.all(np.abs(local_camera) <= dimensions / 2):
                raise ValueError("Camera is inside scene geometry")
        for light in scene["lights"]:
            _keys(light, ("center_metres", "power_watts", "size_metres"))
            _vector(light["center_metres"], 3)
            if (
                not 0 < light["power_watts"] <= 10000
                or not 0 < light["size_metres"] <= 10
            ):
                raise ValueError("Invalid downward area light")
    return job
