"""Simulator-independent ray checks for bounded cuboid and triangle scenes."""

import numpy as np

from src.adapters.generation.mesh_assets import load_normalized_mesh


def triangle_distance(origin, direction, triangle_vertices):
    """Return the closest positive ray parameter using two-sided intersections.

    Direction is deliberately unnormalized: when camera optical Z is one, the
    parameter is axial depth in metres even after transforming into asset space.
    The calculation depends on registered triangles, not renderer ray APIs.
    """
    first = triangle_vertices[:, 0]
    edge_a = triangle_vertices[:, 1] - first
    edge_b = triangle_vertices[:, 2] - first
    cross = np.cross(direction, edge_b)
    determinant = np.einsum("ij,ij->i", edge_a, cross)
    usable = np.abs(determinant) > 1e-12
    inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=usable)
    offset = origin - first
    u = np.einsum("ij,ij->i", offset, cross) * inverse
    q = np.cross(offset, edge_a)
    v = np.einsum("j,ij->i", direction, q) * inverse
    distance = np.einsum("ij,ij->i", edge_b, q) * inverse
    usable &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (distance > 0)
    return float(np.min(distance[usable])) if usable.any() else float("inf")


def scene_intersections(job, capture, asset_root, pixels):
    """Return independent nearest instance/axial-depth hits for native pixel rays.

    Registered asset bounds cull candidates before triangle intersections. Scene
    geometry is static and opaque; this does not model scan timing or sensor noise.
    """
    objects = capture["objects"]
    centers = np.asarray([obj["center_metres"] for obj in objects], dtype=float)
    dimensions = np.asarray([obj["dimensions_metres"] for obj in objects], dtype=float)
    low, high = centers - dimensions / 2, centers + dimensions / 2
    cache = {}
    for obj in objects:
        if "mesh" in obj and obj["mesh"]["sha256"] not in cache:
            arrays = load_normalized_mesh(obj["mesh"], asset_root)
            cache[obj["mesh"]["sha256"]] = arrays["vertices"][
                arrays["triangles"]
            ].astype(float)
    pose = np.asarray(capture["optical_camera_to_world"], dtype=float)
    origin = pose[:3, 3]
    fx, fy, cx, cy = job["camera"]["intrinsics"]
    results = []
    for u, v in pixels:
        direction = pose[:3, :3] @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            first, second = (low - origin) / direction, (high - origin) / direction
        near, far = np.minimum(first, second), np.maximum(first, second)
        for axis in np.flatnonzero(np.abs(direction) < 1e-12):
            inside = (low[:, axis] <= origin[axis]) & (origin[axis] <= high[:, axis])
            near[:, axis] = np.where(inside, -np.inf, np.inf)
            far[:, axis] = np.where(inside, np.inf, -np.inf)
        entry, exit = near.max(axis=1), far.min(axis=1)
        candidates = np.flatnonzero((exit >= entry) & (exit > 0))
        best, identity = float("inf"), 0
        for index in sorted(candidates, key=lambda index: max(entry[index], 0)):
            if entry[index] > best:
                break
            obj = objects[index]
            if "mesh" in obj:
                distance = triangle_distance(
                    (origin - centers[index]) / dimensions[index],
                    direction / dimensions[index],
                    cache[obj["mesh"]["sha256"]],
                )
            else:
                distance = (
                    float(entry[index]) if entry[index] > 0 else float(exit[index])
                )
            if distance < best:
                best, identity = distance, obj["instance_id"]
        results.append(
            dict(u=int(u), v=int(v), instance_id=identity, depth_axial_metres=best)
        )
    return results
