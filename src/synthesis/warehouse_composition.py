"""Compose palletized loads in a source-bound warehouse bay.

The planner reuses a verified scene and registered assets. It changes authored
layout and camera viewpoints, not materials, sensor response or renderer APIs.
Positions and sizes are metres; mesh scale remains uniform. These static scenes
require geometry and appearance qualification before training admission.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import re

import numpy as np

from src.adapters.storage.local import file_hash
from src.adapters.tasks.goods_scene import validate_goods_scene


def require_fields(value, fields):
    """Reject unknown intent instead of silently ignoring configuration fields."""
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("Unsupported warehouse composition fields")


def finite_vector(value, length):
    """Read one bounded finite metric vector; booleans are not coordinates."""
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(number) not in (int, float) for number in value)
    ):
        raise ValueError("Expected explicit numeric composition coordinates")
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all() or np.max(np.abs(array)) > 100:
        raise ValueError("Composition coordinates exceed finite metric bounds")
    return array


def build_composition_job(
    source,
    source_sha256,
    catalog,
    catalog_sha256,
    configuration,
    evidence,
    evidence_sha256,
):
    """Return a new portable job and per-load lineage; leave every input intact.

    Explicit removals are bound to the source job checksum. The pallet template
    must be axis-aligned background cuboids with a planar top. Asset dimensions
    come from the registered metric conversion, with original UV/material maps.
    Camera motion is part of this composition intervention, not a calibrated
    reconstruction or an isolated material/lighting experiment.
    """
    validate_goods_scene(source)
    require_fields(
        configuration,
        {
            "schema_version",
            "job_id",
            "source_job_sha256",
            "source_sample_id",
            "training_evidence_sha256",
            "remove_instance_ids",
            "pallet_template_ids",
            "pallet_centres_xy_metres",
            "load_templates",
            "cases",
        },
    )
    if (
        configuration["schema_version"] != "dubaidata.warehouse-composition/1.0"
        or configuration["source_job_sha256"] != source_sha256
        or configuration["training_evidence_sha256"] != evidence_sha256
        or source.get("asset_catalog_sha256") != catalog_sha256
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (source_sha256, evidence_sha256, catalog_sha256)
        )
    ):
        raise ValueError("Composition source or evidence checksum mismatch")
    support = evidence.get("training_support", [])
    if (
        source["conditioning"]["profile_split"] != "train"
        or not support
        or any(row.get("split") != "train" for row in support)
    ):
        raise ValueError("Composition requires explicit training-only evidence")
    observed = {row["family"] for row in evidence.get("observations", [])}
    captures = {scene["sample_id"]: scene for scene in source["scenes"]}
    if configuration["source_sample_id"] not in captures:
        raise ValueError("Composition source view is absent")
    original = captures[configuration["source_sample_id"]]
    objects = {obj["instance_id"]: obj for obj in original["objects"]}
    removed, pallet_ids = (
        configuration[key] for key in ("remove_instance_ids", "pallet_template_ids")
    )
    for identifiers in (removed, pallet_ids):
        if (
            not isinstance(identifiers, list)
            or not identifiers
            or any(type(item) is not int or item not in objects for item in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("Composition requires unique, existing instance IDs")
    if not set(pallet_ids).issubset(removed):
        raise ValueError("Original pallet must be removed before it is cloned")
    pallet = [objects[identifier] for identifier in pallet_ids]
    if any(
        obj["role"] != "equipment" or obj["class_id"] != 0 or "mesh" in obj
        for obj in pallet
    ):
        raise ValueError("Pallet template must contain background cuboids only")
    bottoms = np.array(
        [
            np.array(obj["center_metres"]) - np.array(obj["dimensions_metres"]) / 2
            for obj in pallet
        ]
    )
    tops = np.array(
        [
            np.array(obj["center_metres"]) + np.array(obj["dimensions_metres"]) / 2
            for obj in pallet
        ]
    )
    if abs(float(bottoms[:, 2].min())) > 1e-6:
        raise ValueError("Pallet template must rest on the zero-height floor")
    footprint_min, footprint_max = bottoms[:, :2].min(0), tops[:, :2].max(0)
    pallet_centre = (footprint_min + footprint_max) / 2
    pallet_half = (footprint_max - footprint_min) / 2
    support_z = float(tops[:, 2].max())
    positions = configuration["pallet_centres_xy_metres"]
    if not isinstance(positions, list) or not 1 <= len(positions) <= 4:
        raise ValueError("Composition requires one to four bounded pallet loads")
    positions = [finite_vector(position, 2) for position in positions]
    for index, position in enumerate(positions):
        if any(
            np.all(2 * pallet_half - np.abs(position - previous) > 1e-4)
            for previous in positions[:index]
        ):
            raise ValueError("Pallet footprints intersect")

    templates = configuration["load_templates"]
    if not isinstance(templates, dict) or not 1 <= len(templates) <= 4:
        raise ValueError("Expected bounded identifiable load templates")
    for template in templates.values():
        require_fields(
            template, {"evidence_family", "assets", "materials", "scale", "count"}
        )
        if (
            template["evidence_family"] not in observed
            or type(template["count"]) is not int
            or not 1 <= template["count"] <= 6
            or type(template["scale"]) not in (float, int)
            or not 0.7 <= template["scale"] <= 1.7
            or not isinstance(template["assets"], list)
            or len(template["assets"]) != 2
            or not isinstance(template["materials"], list)
            or len(template["materials"]) != 2
        ):
            raise ValueError("Load template lacks bounded training-supported intent")
        for asset_id, material_id in zip(
            template["assets"], template["materials"], strict=True
        ):
            entry = catalog["assets"].get(asset_id)
            material = source["materials"].get(material_id, {}).get("texture_set", {})
            if (
                not entry
                or material.get("asset_id") != asset_id
                or material.get("maps") != entry["maps"]
                or material.get("projection") != "UV"
            ):
                raise ValueError(
                    "Load mesh and original UV texture must share an asset"
                )
            extent = finite_vector(entry["conversion"]["original_extent_metres"], 3)
            dimensions = extent * template["scale"]
            if np.any(dimensions <= 0) or np.any(dimensions[:2] / 2 > pallet_half):
                raise ValueError("Load is invalid or overhangs its pallet")

    cases = configuration["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        raise ValueError("Composition requires one to eight captures")
    result, lineage = copy.deepcopy(source), []
    result["job_id"] = configuration["job_id"]
    result["conditioning"]["purpose"] = (
        "Training-supported rack composition; evidence SHA256 " + evidence_sha256
    )
    result["scenes"] = []
    identifiers = set()
    for case in cases:
        require_fields(case, {"sample_id", "objective", "camera", "loads"})
        if (
            not isinstance(case["sample_id"], str)
            or case["sample_id"] in identifiers
            or not isinstance(case["objective"], str)
            or not case["objective"].strip()
            or not isinstance(case["loads"], list)
            or len(case["loads"]) != len(positions)
            or any(name not in templates for name in case["loads"])
        ):
            raise ValueError("Invalid or duplicate composition case")
        identifiers.add(case["sample_id"])
        x, y, z, yaw = finite_vector(case["camera"], 4)
        if not (
            abs(x) <= 0.6 and -5.5 <= y <= -3.4 and 0.6 <= z <= 1 and abs(yaw) <= 6
        ):
            raise ValueError("Composition camera exceeds authored research bounds")
        scene = copy.deepcopy(original)
        scene.update(sample_id=case["sample_id"], scenario=case["objective"])
        scene["objects"] = [
            obj for obj in scene["objects"] if obj["instance_id"] not in removed
        ]
        next_id = max(objects) + 1
        for column, (position, name) in enumerate(
            zip(positions, case["loads"], strict=True)
        ):
            for part in pallet:
                clone = copy.deepcopy(part)
                clone["instance_id"] = next_id
                clone["center_metres"][:2] = (
                    np.array(part["center_metres"][:2]) + position - pallet_centre
                ).tolist()
                next_id += 1
                scene["objects"].append(clone)
            template = templates[name]
            bottom = support_z
            for layer in range(template["count"]):
                variant = layer % 2
                asset_id = template["assets"][variant]
                entry = catalog["assets"][asset_id]
                dimensions = (
                    np.array(entry["conversion"]["original_extent_metres"])
                    * template["scale"]
                )
                scene["objects"].append(
                    dict(
                        instance_id=next_id,
                        asset_id="registered.normalized_triangle_mesh/1.0",
                        role="goods",
                        class_id=1,
                        center_metres=[*position.tolist(), bottom + dimensions[2] / 2],
                        dimensions_metres=dimensions.tolist(),
                        material=template["materials"][variant],
                        bevel_metres=0,
                        mesh=copy.deepcopy(entry["mesh"]),
                    )
                )
                lineage.append(
                    dict(
                        sample_id=case["sample_id"],
                        instance_id=next_id,
                        column=column,
                        layer=layer,
                        asset_id=asset_id,
                        evidence_family=template["evidence_family"],
                        uniform_scale=template["scale"],
                        support_top_metres=support_z,
                    )
                )
                next_id += 1
                bottom += dimensions[2]
        angle = math.radians(yaw)
        scene["optical_camera_to_world"] = [
            [math.cos(angle), 0, -math.sin(angle), float(x)],
            [math.sin(angle), 0, math.cos(angle), float(y)],
            [0, -1, 0, float(z)],
            [0, 0, 0, 1],
        ]
        result["scenes"].append(scene)
    validate_goods_scene(result)
    # Check all authored goods against each other before triangle-contact refinement.
    for scene in result["scenes"]:
        goods = [obj for obj in scene["objects"] if obj["role"] == "goods"]
        for index, first in enumerate(goods):
            for second in goods[index + 1 :]:
                overlap = (
                    np.array(first["dimensions_metres"]) + second["dimensions_metres"]
                ) / 2 - np.abs(
                    np.array(first["center_metres"]) - second["center_metres"]
                )
                if np.all(overlap > 1e-4):
                    raise ValueError("Distinct authored goods intersect")
    return result, lineage


def main():
    """Resolve a fresh job; rendering and dataset admission remain separate steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "source-job",
        "asset-catalog",
        "configuration",
        "training-evidence",
        "output",
    ):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        """Read the explicit local input without searching alternate sources."""
        return json.loads(path.read_text(encoding="utf-8"))

    job, _ = build_composition_job(
        read(args.source_job),
        file_hash(args.source_job),
        read(args.asset_catalog),
        file_hash(args.asset_catalog),
        read(args.configuration),
        read(args.training_evidence),
        file_hash(args.training_evidence),
    )
    # Exclusive creation preserves existing jobs and all historical source bindings.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(job, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
