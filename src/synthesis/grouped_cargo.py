"""Add bounded cargo groups on declared supports without changing source objects."""

import copy

import numpy as np

from src.adapters.tasks.goods_scene import validate_goods_scene


def build_grouped_cargo(source, configuration):
    """Return a new goods job and per-instance lineage for supported cargo grids.

    Supports and mesh templates must already exist in the source capture. Metric
    axis-aligned grids preserve each template's dimensions, materials, UV/mesh
    binding and role. Every grid uses equal template dimensions and fits within
    one horizontal support's XY bounds. Layers start at AABB contact; callers
    must run triangle contact refinement before rendering. This does not certify
    physical stability. Inputs are unchanged. Invalid fields, budgets, supports,
    existing-cargo collisions or structural penetrations raise ValueError.
    """
    validate_goods_scene(source)
    if (
        not isinstance(configuration, dict)
        or set(configuration) != {"schema_version", "job_id", "cases"}
        or configuration["schema_version"] != "dubaidata.grouped-cargo/1.0"
        or not isinstance(configuration["cases"], list)
        or not configuration["cases"]
    ):
        raise ValueError("Unsupported grouped cargo configuration")
    job = copy.deepcopy(source)
    job["job_id"] = configuration["job_id"]
    job["conditioning"]["purpose"] += "; grouped cargo coverage pilot"
    scenes = {scene["sample_id"]: scene for scene in job["scenes"]}
    seen, lineage = set(), []
    for case in configuration["cases"]:
        if not isinstance(case, dict) or set(case) != {"sample_id", "groups"}:
            raise ValueError("Unknown grouped capture fields")
        identity = case["sample_id"]
        if identity in seen or identity not in scenes:
            raise ValueError("Unknown or duplicate grouped capture")
        seen.add(identity)
        scene = scenes[identity]
        lookup = {obj["instance_id"]: obj for obj in scene["objects"]}
        if not any(obj["role"] == "goods" for obj in scene["objects"]):
            raise ValueError("Background-only captures must remain unchanged")
        if not isinstance(case["groups"], list) or not 1 <= len(case["groups"]) <= 4:
            raise ValueError("Grouped capture needs one to four explicit supports")
        next_id = max(lookup) + 1
        for group in case["groups"]:
            if not isinstance(group, dict) or set(group) != {
                "support_instance_id",
                "template_instance_ids",
                "counts_xyz",
                "gap_xy_metres",
                "center_xy_offset_metres",
            }:
                raise ValueError("Unknown cargo grid fields")
            templates = group["template_instance_ids"]
            support_id = group["support_instance_id"]
            if type(support_id) is not int or support_id not in lookup:
                raise ValueError("Unknown cargo support")
            support = lookup[support_id]
            if (
                support["role"] != "structure"
                or support["class_id"] != 0
                or support["material"] != "wood"
                or "mesh" in support
            ):
                raise ValueError("Grid requires an existing horizontal wooden support")
            if (
                not isinstance(templates, list)
                or not templates
                or any(type(key) is not int for key in templates)
                or len(set(templates)) != len(templates)
            ):
                raise ValueError("Grid requires unique existing cargo templates")
            if any(
                type(key) is not int
                or key not in lookup
                or lookup[key]["role"] != "goods"
                or "mesh" not in lookup[key]
                for key in templates
            ):
                raise ValueError("Only existing registered goods meshes may be cloned")
            extent = np.array(lookup[templates[0]]["dimensions_metres"], dtype=float)
            if any(
                not np.array_equal(extent, lookup[key]["dimensions_metres"])
                for key in templates
            ):
                raise ValueError("Grid templates must have identical metric dimensions")
            counts = group["counts_xyz"]
            if (
                not isinstance(counts, list)
                or len(counts) != 3
                or any(
                    type(count) is not int or not 1 <= count <= 4 for count in counts
                )
                or np.prod(counts) > 32
            ):
                raise ValueError("Cargo grid exceeds its explicit instance budget")
            gaps = group["gap_xy_metres"]
            if (
                not isinstance(gaps, list)
                or len(gaps) != 2
                or any(
                    type(gap) not in (int, float)
                    or not np.isfinite(gap)
                    or not 0.01 <= gap <= 0.5
                    for gap in gaps
                )
            ):
                raise ValueError("Cargo grid needs finite 0.01-0.5 metre XY gaps")
            offset = group["center_xy_offset_metres"]
            if (
                not isinstance(offset, list)
                or len(offset) != 2
                or any(
                    type(value) not in (int, float) or not np.isfinite(value)
                    for value in offset
                )
            ):
                raise ValueError("Cargo group center offset must be two finite metres")
            footprint = extent[:2] * counts[:2] + np.array(gaps) * (
                np.array(counts[:2]) - 1
            )
            support_half = np.array(support["dimensions_metres"][:2]) / 2
            if np.any(
                footprint > np.array(support["dimensions_metres"][:2]) - 0.02
            ) or np.any(np.abs(offset) + footprint / 2 > support_half - 0.01):
                raise ValueError("Cargo grid overhangs its support")
            support_top = (
                support["center_metres"][2] + support["dimensions_metres"][2] / 2
            )
            for z in range(counts[2]):
                for y in range(counts[1]):
                    for x in range(counts[0]):
                        template_id = templates[(x + y + z) % len(templates)]
                        clone = copy.deepcopy(lookup[template_id])
                        clone["instance_id"] = next_id
                        clone["center_metres"] = [
                            float(
                                support["center_metres"][0]
                                + offset[0]
                                + (x - (counts[0] - 1) / 2) * (extent[0] + gaps[0])
                            ),
                            float(
                                support["center_metres"][1]
                                + offset[1]
                                + (y - (counts[1] - 1) / 2) * (extent[1] + gaps[1])
                            ),
                            float(support_top + 0.0002 + (z + 0.5) * extent[2]),
                        ]
                        for old in scene["objects"]:
                            if old["role"] != "goods":
                                continue
                            overlap = (
                                np.array(old["dimensions_metres"]) + extent
                            ) / 2 - np.abs(
                                np.array(old["center_metres"]) - clone["center_metres"]
                            )
                            if np.all(overlap > 1e-7):
                                raise ValueError("Cargo grid overlaps existing goods")
                        scene["objects"].append(clone)
                        lineage.append(
                            dict(
                                sample_id=identity,
                                instance_id=next_id,
                                template_instance_id=template_id,
                                support_instance_id=support_id,
                                grid_index_xyz=[x, y, z],
                                center_metres=clone["center_metres"],
                                dimensions_metres=clone["dimensions_metres"],
                            )
                        )
                        next_id += 1
    validate_goods_scene(job)
    return job, lineage
