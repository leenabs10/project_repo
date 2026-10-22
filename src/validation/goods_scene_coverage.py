"""Measure requested carton visibility using resolved metric geometry.

Projected cuboid silhouettes are computed independently of Blender. Foreground
occlusion is measured from clean axial depth, not inferred from RGB appearance.
These diagnostics qualify scenario intent; they do not qualify photorealism.
"""

import numpy as np

from src.synthesis.warehouse_scenarios import validate_coverage


def measure_target_coverage(job, configuration, capture, depth, instances):
    """Return per-load native pixel coverage and measured equipment occlusion.

    ``depth`` is float32 axial metres and ``instances`` contains visible uint16
    IDs. Call after canonical payload validation. Packaging within 1 cm of the
    carton surface is treated as its surface, avoiding false occlusion from tape.
    """
    validate_coverage(configuration)
    case = next(
        row
        for row in configuration["cases"]
        if row["sample_id"] == capture["sample_id"]
    )
    width, height = job["camera"]["width"], job["camera"]["height"]
    if depth.shape != (height, width) or instances.shape != (height, width):
        raise ValueError("Coverage arrays must use the resolved native camera")
    pose = np.asarray(capture["optical_camera_to_world"], dtype=float)
    origin, rotation = pose[:3, 3], pose[:3, :3]
    fx, fy, cx, cy = job["camera"]["intrinsics"]
    roles = {obj["instance_id"]: obj["role"] for obj in capture["objects"]}
    equipment_ids = [
        identifier for identifier, role in roles.items() if role == "equipment"
    ]
    goods_ids = [identifier for identifier, role in roles.items() if role == "goods"]
    measurements = []
    for load in case["loads"]:
        x, y = load["center_xy_metres"]
        dimensions = np.asarray(load["dimensions_metres"], dtype=float)
        bottom = 0.22 if load["support"] == "foreground_pallet" else 0.0
        center = np.array([x, y, bottom + dimensions[2] / 2])
        matching = [
            obj
            for obj in capture["objects"]
            if obj["role"] == "goods"
            and np.allclose(obj["center_metres"], center, atol=1e-8, rtol=0)
            and np.allclose(obj["dimensions_metres"], dimensions, atol=1e-8, rtol=0)
            and obj["material"] == load["material"]
        ]
        if len(matching) != 1:
            raise ValueError("Requested load has no unique resolved carton body")
        half = dimensions / 2
        corners = np.array(
            [
                center + half * [sx, sy, sz]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )
        optical = (corners - origin) @ rotation
        if (optical[:, 2] <= job["camera"]["near_metres"]).any():
            raise ValueError("Coverage target crosses the camera near plane")
        projected = optical[:, :2] / optical[:, 2, None] * [fx, fy] + [cx, cy]
        low = np.maximum(np.floor(projected.min(axis=0)).astype(int), [0, 0])
        high = np.minimum(
            np.ceil(projected.max(axis=0)).astype(int) + 1, [width, height]
        )
        if np.any(high <= low):
            raise ValueError("Requested coverage target is outside the camera")
        u, v = np.meshgrid(np.arange(low[0], high[0]), np.arange(low[1], high[1]))
        direction = (
            np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), axis=-1)
            @ rotation.T
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            first = (center - half - origin) / direction
            second = (center + half - origin) / direction
        near, far = np.minimum(first, second), np.maximum(first, second)
        # Parallel rays miss when their origin is outside the slab. Set explicit
        # infinities rather than depending on a 0/0 result at a box boundary.
        for axis in range(3):
            parallel = np.abs(direction[..., axis]) < 1e-12
            inside = (
                center[axis] - half[axis] <= origin[axis] <= center[axis] + half[axis]
            )
            near[..., axis] = np.where(
                parallel, -np.inf if inside else np.inf, near[..., axis]
            )
            far[..., axis] = np.where(
                parallel, np.inf if inside else -np.inf, far[..., axis]
            )
        entry, exit = near.max(axis=-1), far.min(axis=-1)
        silhouette = (
            (exit >= entry) & (entry > 0) & (entry <= job["camera"]["far_metres"])
        )
        native_depth, native_ids = depth[v, u], instances[v, u]
        foreground = silhouette & (native_depth > 0) & (native_depth < entry - 0.01)
        visible_surface = (
            silhouette
            & (np.abs(native_depth - entry) <= 0.01)
            & np.isin(native_ids, goods_ids)
        )
        projected_pixels = int(silhouette.sum())
        visible_pixels = int(visible_surface.sum())
        if projected_pixels == 0 or visible_pixels < 100:
            raise ValueError(
                "Requested coverage load has insufficient visible target pixels"
            )
        measurements.append(
            dict(
                body_instance_id=matching[0]["instance_id"],
                support=load["support"],
                dimensions_metres=load["dimensions_metres"],
                axial_center_metres=float(optical[:, 2].mean()),
                projected_silhouette_pixels=projected_pixels,
                visible_surface_pixels=visible_pixels,
                visible_surface_fraction=visible_pixels / projected_pixels,
                foreground_occlusion_fraction=float(
                    foreground.sum() / projected_pixels
                ),
                equipment_occlusion_fraction=float(
                    (foreground & np.isin(native_ids, equipment_ids)).sum()
                    / projected_pixels
                ),
            )
        )
    return dict(
        sample_id=case["sample_id"], objective=case["objective"], targets=measurements
    )
