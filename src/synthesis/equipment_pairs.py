"""Author exact empty/loaded contrasts from a source-bound warehouse scene.

The two members share equipment, structure, camera, materials and lights. Only
explicit identifiable cargo is added. Static contact is not a stability model.
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
from src.synthesis.warehouse_composition import require_fields, finite_vector


def build_equipment_pairs(
    source,
    source_sha256,
    catalog,
    catalog_sha256,
    configuration,
    evidence,
    evidence_sha256,
):
    """Return a separate resolved job and pair lineage; do not alter inputs.

    Metric pallet parts and UV cargo come from the verified source/catalog.
    All old goods are removed in both members, so empty views really contain no
    foreground. Explicit equipment removals are bound to the exact source hash.
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
            "pallet_template_ids",
            "remove_equipment_ids",
            "pairs",
        },
    )
    if (
        configuration["schema_version"] != "dubaidata.equipment-pairs/1.0"
        or configuration["source_job_sha256"] != source_sha256
        or configuration["training_evidence_sha256"] != evidence_sha256
        or source.get("asset_catalog_sha256") != catalog_sha256
        or any(
            not re.fullmatch("[0-9a-f]{64}", s)
            for s in (source_sha256, catalog_sha256, evidence_sha256)
        )
    ):
        raise ValueError("Paired scene source/evidence binding mismatch")
    support = evidence.get("training_support", [])
    if not support or any(
        r.get("split") != "train"
        or not r.get("sample_id", "").startswith("Aisle_CCW_Run_1/")
        for r in support
    ):
        raise ValueError("Paired scenes require training-only support")
    original = next(
        (
            s
            for s in source["scenes"]
            if s["sample_id"] == configuration["source_sample_id"]
        ),
        None,
    )
    if original is None:
        raise ValueError("Missing paired source scene")
    lookup = {o["instance_id"]: o for o in original["objects"]}
    for field in ("pallet_template_ids", "remove_equipment_ids"):
        ids = configuration[field]
        if (
            not isinstance(ids, list)
            or not ids
            or len(set(ids)) != len(ids)
            or any(
                type(i) is not int
                or i not in lookup
                or lookup[i]["role"] != "equipment"
                for i in ids
            )
        ):
            raise ValueError(
                "Paired equipment requires unique existing background parts"
            )
    if not set(configuration["pallet_template_ids"]).issubset(
        configuration["remove_equipment_ids"]
    ):
        raise ValueError("Source pallet must be removed before cloning")
    pallet = [lookup[i] for i in configuration["pallet_template_ids"]]
    if any("mesh" in p for p in pallet):
        raise ValueError("Pallet template must be metric cuboids")
    low = np.min(
        [
            np.array(p["center_metres"]) - np.array(p["dimensions_metres"]) / 2
            for p in pallet
        ],
        axis=0,
    )
    high = np.max(
        [
            np.array(p["center_metres"]) + np.array(p["dimensions_metres"]) / 2
            for p in pallet
        ],
        axis=0,
    )
    if abs(low[2]) > 1e-8:
        raise ValueError("Source pallet must rest on the floor")
    half = (high[:2] - low[:2]) / 2
    centre = (low[:2] + high[:2]) / 2
    pairs = configuration["pairs"]
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 4:
        raise ValueError("Paired pilot requires one to four pairs")
    result = copy.deepcopy(source)
    result.update(
        job_id=configuration["job_id"], required_visible_class_ids=[0], scenes=[]
    )
    result["conditioning"]["purpose"] = (
        "Training-supported empty/loaded equipment contrast; evidence SHA256 "
        + evidence_sha256
    )
    seen, manifest = set(), []
    families = {r["family"] for r in evidence.get("observations", [])}
    for pair in pairs:
        require_fields(
            pair,
            {
                "pair_id",
                "camera",
                "xy",
                "pallet_layers",
                "cargo",
                "cargo_scale",
                "cargo_count",
            },
        )
        pair_id = pair["pair_id"]
        if (
            not isinstance(pair_id, str)
            or not re.fullmatch("[a-z][a-z0-9_]{0,55}", pair_id)
            or pair_id in seen
        ):
            raise ValueError("Invalid or duplicate pair ID")
        seen.add(pair_id)
        if (
            pair["cargo"] not in ("old_tyre", "cardboard_box_01")
            or ("stacked_tyres" if pair["cargo"] == "old_tyre" else "stacked_cartons")
            not in families
        ):
            raise ValueError("Cargo family lacks training support")
        if (
            type(pair["pallet_layers"]) is not int
            or not 1 <= pair["pallet_layers"] <= 4
            or type(pair["cargo_count"]) is not int
            or not 1 <= pair["cargo_count"] <= 5
            or type(pair["cargo_scale"]) not in (int, float)
            or not 0.7 <= pair["cargo_scale"] <= 1.7
        ):
            raise ValueError("Unsupported paired load count/scale")
        xy = finite_vector(pair["xy"], 2)
        if abs(xy[0]) > 3.1 or not -1.7 <= xy[1] <= 1.5:
            raise ValueError("Pallet placement exceeds authored bay bounds")
        x, y, z, yaw = finite_vector(pair["camera"], 4)
        if abs(x) > 0.6 or not -5.5 <= y <= -3.4 or not 0.6 <= z <= 1 or abs(yaw) > 6:
            raise ValueError("Paired camera exceeds authored bounds")
        empty = copy.deepcopy(original)
        empty.update(
            sample_id=pair_id + "_empty",
            scenario="Empty equipment contrast: " + pair_id,
        )
        empty["objects"] = [
            o
            for o in empty["objects"]
            if o["role"] != "goods"
            and o["instance_id"] not in configuration["remove_equipment_ids"]
        ]
        angle = math.radians(yaw)
        empty["optical_camera_to_world"] = [
            [math.cos(angle), 0, -math.sin(angle), float(x)],
            [math.sin(angle), 0, math.cos(angle), float(y)],
            [0, -1, 0, float(z)],
            [0, 0, 0, 1],
        ]
        next_id = max(lookup) + 1
        for layer in range(pair["pallet_layers"]):
            for part in pallet:
                clone = copy.deepcopy(part)
                clone["instance_id"] = next_id
                clone["center_metres"][:2] = (
                    np.array(part["center_metres"][:2]) + xy - centre
                ).tolist()
                clone["center_metres"][2] += layer * float(high[2])
                empty["objects"].append(clone)
                next_id += 1
        loaded = copy.deepcopy(empty)
        loaded.update(
            sample_id=pair_id + "_loaded",
            scenario="Identifiable load on matching equipment: " + pair_id,
        )
        bottom = pair["pallet_layers"] * float(high[2])
        cargo_ids = []
        for layer in range(pair["cargo_count"]):
            asset_id = f"polyhaven.{pair['cargo']}.{'front' if layer%2 else 'back'}.4k"
            entry = catalog["assets"][asset_id]
            material = next(
                (
                    name
                    for name, m in source["materials"].items()
                    if m.get("texture_set", {}).get("asset_id") == asset_id
                    and m["texture_set"]["maps"] == entry["maps"]
                    and m["texture_set"].get("projection") == "UV"
                ),
                None,
            )
            if material is None:
                raise ValueError("Original cargo UV material is absent")
            dimensions = (
                finite_vector(entry["conversion"]["original_extent_metres"], 3)
                * pair["cargo_scale"]
            )
            if np.any(dimensions <= 0) or np.any(dimensions[:2] / 2 > half):
                raise ValueError("Cargo overhangs the paired pallet")
            loaded["objects"].append(
                dict(
                    instance_id=next_id,
                    asset_id="registered.normalized_triangle_mesh/1.0",
                    role="goods",
                    class_id=1,
                    center_metres=[*xy.tolist(), bottom + float(dimensions[2]) / 2],
                    dimensions_metres=dimensions.tolist(),
                    material=material,
                    bevel_metres=0,
                    mesh=copy.deepcopy(entry["mesh"]),
                )
            )
            cargo_ids.append(next_id)
            next_id += 1
            bottom += float(dimensions[2])
        result["scenes"].extend([empty, loaded])
        manifest.append(
            dict(
                pair_id=pair_id,
                empty=empty["sample_id"],
                loaded=loaded["sample_id"],
                cargo_instance_ids=cargo_ids,
                shared_equipment_instance_ids=[
                    o["instance_id"]
                    for o in empty["objects"]
                    if o["role"] == "equipment"
                ],
            )
        )
    validate_goods_scene(result)
    return result, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-job",
        "catalog",
        "configuration",
        "training-evidence",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    job, pairs = build_equipment_pairs(
        read(args.source_job),
        file_hash(args.source_job),
        read(args.catalog),
        file_hash(args.catalog),
        read(args.configuration),
        read(args.training_evidence),
        file_hash(args.training_evidence),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in (("job.json", job), ("pairs.json", pairs)):
        (args.output / name).write_text(
            json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
