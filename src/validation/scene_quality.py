"""Independently verify rendered goods/equipment geometry before appearance admission.

The sampler balances goods meshes, equipment meshes and structural surfaces.
Nearest-hit calculations use NumPy triangles and metric cuboids, not renderer ray
APIs. Passing these checks establishes correspondence, not photographic fidelity.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash, reserve_output
from src.validation.artifacts import artifact_hashes, verify_snapshot, write_json
from src.validation.mesh_geometry import scene_intersections
from src.validation.scene_admission import (
    inspect_scene_dataset,
    require_snapshot_digest,
)


def stable_interior(ids, radius=2):
    """Return pixels whose complete neighbourhood has one nonzero instance ID."""
    stable = ids != 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            stable &= ids == np.roll(ids, (dy, dx), axis=(0, 1))
    stable[:radius] = False
    stable[-radius:] = False
    stable[:, :radius] = False
    stable[:, -radius:] = False
    return stable


def sample_role_pixels(ids, objects, *, seed=5519, allow_background_only=False):
    """Select 256 stable native pixel rays with explicit role coverage.

    Background-only jobs opt into assigning the absent 96 goods rays to structure.
    Equipment still receives 64 rays; the default mixed-scene sampler is unchanged."""
    stable = stable_interior(ids)
    goods_ids = {obj["instance_id"] for obj in objects if obj["role"] == "goods"}
    negative = allow_background_only and not np.isin(ids, list(goods_ids)).any()
    groups = (
        (
            "goods_mesh",
            0 if negative else 96,
            {
                obj["instance_id"]
                for obj in objects
                if "mesh" in obj and obj["role"] == "goods"
            },
        ),
        (
            "equipment_mesh",
            64,
            {
                obj["instance_id"]
                for obj in objects
                if "mesh" in obj and obj["role"] == "equipment"
            },
        ),
        (
            "structure",
            192 if negative else 96,
            {obj["instance_id"] for obj in objects if obj["role"] == "structure"},
        ),
    )
    rng = np.random.default_rng(seed)
    pixels, counts = [], {}
    for name, count, identities in groups:
        candidates = np.argwhere(stable & np.isin(ids, list(identities)))
        if len(candidates) < count:
            raise ValueError("Insufficient visible interior coverage for " + name)
        selected = candidates[rng.choice(len(candidates), count, replace=False)]
        pixels.extend((int(u), int(v)) for v, u in selected)
        counts[name] = count
    return pixels, counts


def verify_scene_quality(
    dataset, dataset_sha256, asset_root, references, references_sha256, output
):
    """Write a fresh source-bound geometry snapshot; never admit training data.

    Depth must agree to p95 <= 5 mm, and instance agreement must be at least 99%.
    Native semantic counts and per-instance visibility are retained for coverage
    inspection. Exact arrays and all source bindings are rechecked before/after.
    """
    dataset, asset_root = Path(dataset), Path(asset_root)
    require_snapshot_digest(references_sha256)
    reference_profile = verify_snapshot(Path(references), references_sha256)
    profile, job, rows = inspect_scene_dataset(dataset, dataset_sha256)
    if file_hash(asset_root / "catalog.json") != job["asset_catalog_sha256"]:
        raise ValueError(
            "Independent geometry requires the exact registered asset catalog"
        )
    evidence, coverage = [], []
    for capture in job["scenes"]:
        folder = dataset / capture["sample_id"]
        with Image.open(folder / "instance_ids.png") as image:
            ids = np.array(image)
        with Image.open(folder / "semantic_ids.png") as image:
            semantic = np.array(image)
        depth = np.load(folder / "depth_axial_m.npy", allow_pickle=False)
        pixels, counts = sample_role_pixels(
            ids,
            capture["objects"],
            allow_background_only=job.get("required_visible_class_ids") == [0],
        )
        hits = scene_intersections(job, capture, asset_root, pixels)
        for hit in hits:
            u, v = hit["u"], hit["v"]
            hit["rendered_instance_id"] = int(ids[v, u])
            hit["depth_error_metres"] = abs(
                hit["depth_axial_metres"] - float(depth[v, u])
            )
        agreement = float(
            np.mean([hit["instance_id"] == hit["rendered_instance_id"] for hit in hits])
        )
        errors = np.quantile(
            [hit["depth_error_metres"] for hit in hits], [0.5, 0.95, 1]
        ).tolist()
        if agreement < 0.99 or not np.isfinite(errors).all() or errors[1] > 0.005:
            raise ValueError(
                "Independent geometry gate failed for " + capture["sample_id"]
            )
        evidence.append(
            dict(
                sample_id=capture["sample_id"],
                ray_groups=counts,
                instance_agreement=agreement,
                depth_error_metres_p50_p95_max=errors,
                rays=hits,
            )
        )
        instances = []
        for obj in capture["objects"]:
            if "mesh" not in obj:
                continue
            visible = ids == obj["instance_id"]
            if np.any(semantic[visible] != obj["class_id"]):
                raise ValueError(
                    "Visible source asset violates declared goods/equipment labels"
                )
            instances.append(
                dict(
                    instance_id=obj["instance_id"],
                    role=obj["role"],
                    class_id=obj["class_id"],
                    asset_id=obj["mesh"]["asset_id"],
                    visible_pixels=int(visible.sum()),
                    center_metres=obj["center_metres"],
                    dimensions_metres=obj["dimensions_metres"],
                )
            )
        coverage.append(
            dict(
                sample_id=capture["sample_id"],
                scenario=capture["scenario"],
                goods_pixels=int((semantic == 1).sum()),
                background_pixels=int((semantic == 0).sum()),
                instances=instances,
            )
        )
    # Reject mutation while verification ran, rather than binding stale arrays.
    inspect_scene_dataset(dataset, dataset_sha256)
    verify_snapshot(Path(references), references_sha256)
    output = reserve_output(
        output,
        (dataset, asset_root, Path(references), Path(__file__).parents[2] / "data/raw"),
    )
    write_json(output / "independent_geometry.json", evidence)
    write_json(output / "coverage.json", coverage)
    shutil.copyfile(dataset / "generation_job.json", output / "job.json")
    shutil.copyfile(asset_root / "catalog.json", output / "asset_catalog.json")
    result = dict(
        status="geometry_verified_appearance_pending",
        captures=len(rows),
        dataset_profile_sha256=dataset_sha256,
        job_sha256=profile["job_sha256"],
        independent_rays=256 * len(rows),
        independent_mesh_rays=sum(
            row["ray_groups"]["goods_mesh"] + row["ray_groups"]["equipment_mesh"]
            for row in evidence
        ),
        independent_instance_agreement=[row["instance_agreement"] for row in evidence],
        independent_depth_p95_metres=[
            row["depth_error_metres_p50_p95_max"][1] for row in evidence
        ],
        training_eligible=False,
        scene_utility_measured=False,
        held_out_test_evaluated=False,
        reference_profile_sha256=references_sha256,
        reference_artifacts_changed=0,
        reference_artifacts_verified=len(reference_profile["artifact_sha256"]),
        geometry_scope="Static opaque scene correspondence; not sensor calibration, physical stability or appearance acceptance",
        artifact_sha256=artifact_hashes(output),
    )
    write_json(output / "profile.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--references-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_scene_quality(
        args.dataset,
        args.dataset_sha256,
        args.asset_root,
        args.references,
        args.references_sha256,
        args.output,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "artifact_sha256"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
