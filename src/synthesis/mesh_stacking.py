"""Resolve vertical contact for aligned static cargo, without simulating stability.

Only Z translations change. Original jobs, meshes, UVs, roles and class IDs are
preserved. RGB, depth and labels must be rendered together from the new job.
"""

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np

from src.adapters.generation.mesh_assets import load_normalized_mesh
from src.adapters.storage.local import file_hash
from src.adapters.tasks.goods_scene import validate_goods_scene

BASES = np.array(list(itertools.combinations(range(6), 4)))


def pair_gap(lower, upper):
    """Minimize upper Z minus lower Z at common XY for two (3, 3) triangles.

    Coordinates use metres in the same world frame. Float64 barycentric solves
    include projected vertices and edge crossings; disjoint XY returns infinity.
    Negative values mean the declared vertical ordering already overlaps.
    """
    matrix = np.zeros((4, 6))
    matrix[0, :3] = 1
    matrix[1, 3:] = 1
    matrix[2:, :3] = lower[:, :2].T
    matrix[2:, 3:] = -upper[:, :2].T
    target = np.array([1.0, 1.0, 0.0, 0.0])
    cost = np.r_[-lower[:, 2], upper[:, 2]]
    systems = matrix[:, BASES].transpose(1, 0, 2)
    usable = np.abs(np.linalg.det(systems)) > 1e-14
    if usable.any():
        weights = np.linalg.solve(
            systems[usable], np.tile(target, (usable.sum(), 1))[..., None]
        )[..., 0]
        valid = np.all(weights >= -1e-9, axis=1)
        if valid.any():
            return float(
                np.min(np.sum(weights[valid] * cost[BASES[usable][valid]], axis=1))
            )
    # Collinear or point projections have lower rank. Enumerate their smaller
    # bases rather than dropping vertical faces or assuming a watertight mesh.
    rank = np.linalg.matrix_rank(matrix, tol=1e-12)
    if rank < 4:
        best = float("inf")
        for indices in itertools.combinations(range(6), rank):
            weights, _, basis_rank, _ = np.linalg.lstsq(
                matrix[:, indices], target, rcond=1e-12
            )
            if (
                basis_rank == rank
                and np.all(weights >= -1e-9)
                and np.max(np.abs(matrix[:, indices] @ weights - target)) < 1e-9
            ):
                best = min(best, float(weights @ cost[list(indices)]))
        return best
    return float("inf")


def vertical_gap(lower, upper):
    """Return minimum upper-minus-lower Z at common XY and tested pair count.

    Inputs are float-compatible triangle arrays (N, 3, 3) in metric world axes.
    The linear program enumerates intersection vertices in barycentric coordinates,
    including edge crossings and degenerate XY projections. No overlap returns inf.
    Bounds prune pairs, but no spatial sampling approximates the minimum. This is
    first contact under vertical translation, not a load-support/stability model.
    """
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if any(
        a.ndim != 3
        or a.shape[1:] != (3, 3)
        or not 1 <= len(a) <= 30000
        or not np.isfinite(a).all()
        for a in (lower, upper)
    ):
        raise ValueError("Contact requires bounded finite metric triangle arrays")
    lo, hi = lower.min(axis=1), lower.max(axis=1)
    umin, umax = upper.min(axis=1), upper.max(axis=1)
    best = float("inf")
    checked = 0
    order = np.argsort(-hi[:, 2])
    for i in order:
        if umin[:, 2].min() - hi[i, 2] >= best:
            break
        candidates = np.flatnonzero(
            np.all(umax[:, :2] >= lo[i, :2] - 1e-10, axis=1)
            & np.all(umin[:, :2] <= hi[i, :2] + 1e-10, axis=1)
            & (umin[:, 2] - hi[i, 2] < best)
        )
        for j in candidates[np.argsort(umin[candidates, 2])]:
            if umin[j, 2] - hi[i, 2] >= best:
                continue
            best = min(best, pair_gap(lower[i], upper[j]))
            checked += 1
            if checked > 1000000:
                raise ValueError("Contact triangle-pair budget exceeded")
    return best, checked


def refine_stacks(job, asset_root, *, clearance_metres=0.0002):
    """Return a new resolved job and contact evidence for AABB-aligned stacks.

    Supports only equal-footprint goods meshes originally stacked at bounding-box
    contact. Ambiguous supports, existing penetration and unverified cargo overlap
    fail. A small positive clearance bounds numerical error; tyre pairs already
    closer than it are retained. Engine-specific physics is deliberately absent.
    """
    validate_goods_scene(job)
    if not np.isfinite(clearance_metres) or not 0.0001 <= clearance_metres <= 0.001:
        raise ValueError("Contact clearance must be 0.1 to 1 millimetre")
    refined = copy.deepcopy(job)
    refined["job_id"] += "_contact_v2"
    cache, contacts = {}, []

    def gap_for(lower, upper):
        key = tuple(
            (obj["mesh"]["sha256"], *obj["dimensions_metres"]) for obj in (lower, upper)
        )
        if key not in cache:
            parts = []
            for obj in (lower, upper):
                data = load_normalized_mesh(obj["mesh"], asset_root)
                parts.append(
                    data["vertices"][data["triangles"]].astype(float)
                    * obj["dimensions_metres"]
                )
            parts[1][:, :, 2] += (
                lower["dimensions_metres"][2] + upper["dimensions_metres"][2]
            ) / 2
            cache[key] = vertical_gap(*parts)
        return cache[key]

    for scene in refined["scenes"]:
        goods = [
            obj for obj in scene["objects"] if obj["role"] == "goods" and "mesh" in obj
        ]
        originals = {obj["instance_id"]: copy.deepcopy(obj) for obj in goods}
        aligned_groups = {}
        for obj in goods:
            key = tuple(obj["center_metres"][:2] + obj["dimensions_metres"][:2])
            aligned_groups.setdefault(key, []).append(obj)
        verified_pairs = set()
        for upper in sorted(goods, key=lambda obj: obj["center_metres"][2]):
            original = originals[upper["instance_id"]]
            parents = []
            key = tuple(
                original["center_metres"][:2] + original["dimensions_metres"][:2]
            )
            for lower in aligned_groups[key]:
                source = originals[lower["instance_id"]]
                separation = (
                    original["center_metres"][2]
                    - source["center_metres"][2]
                    - (
                        source["dimensions_metres"][2]
                        + original["dimensions_metres"][2]
                    )
                    / 2
                )
                if abs(separation) < 1e-8:
                    parents.append(lower)
            if len(parents) > 1:
                raise ValueError("Ambiguous vertical cargo support")
            if not parents:
                continue
            lower = parents[0]
            gap, pairs = gap_for(lower, upper)
            if not np.isfinite(gap) or gap < -1e-7:
                raise ValueError(
                    "Cargo has no common support footprint or already penetrates"
                )
            lower_shift = (
                lower["center_metres"][2]
                - originals[lower["instance_id"]]["center_metres"][2]
            )
            shift = lower_shift - max(0, gap - clearance_metres)
            upper["center_metres"][2] += shift
            verified_pairs.add(frozenset((lower["instance_id"], upper["instance_id"])))
            contacts.append(
                dict(
                    sample_id=scene["sample_id"],
                    lower=lower["instance_id"],
                    upper=upper["instance_id"],
                    original_gap_metres=gap,
                    revised_gap_metres=gap + shift - lower_shift,
                    upper_translation_z_metres=shift,
                    triangle_pairs=pairs,
                )
            )
        # AABB overlap is permitted only for adjacent pairs proved separated by
        # the full triangle calculation. Other goods remain conservatively held.
        for index, first in enumerate(goods):
            for second in goods[index + 1 :]:
                overlap = (
                    np.array(first["dimensions_metres"]) + second["dimensions_metres"]
                ) / 2 - np.abs(
                    np.array(first["center_metres"]) - second["center_metres"]
                )
                if (
                    np.all(overlap > 1e-7)
                    and frozenset((first["instance_id"], second["instance_id"]))
                    not in verified_pairs
                ):
                    raise ValueError("Unverified overlap after contact refinement")
    validate_goods_scene(refined)
    return refined, dict(
        algorithm="vertical_triangle_contact/1.0",
        clearance_metres=clearance_metres,
        contacts=contacts,
        training_eligible=False,
        appearance_accepted=False,
        physical_stability_verified=False,
    )


def main():
    """Write a separate candidate and its source binding; never overwrite inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Contact revision requires a new output directory")
    original_hash = file_hash(args.job)
    job, evidence = refine_stacks(
        json.loads(args.job.read_text(encoding="utf-8")), args.asset_root
    )
    if file_hash(args.job) != original_hash:
        raise ValueError("Source job changed during refinement")
    args.output.mkdir(parents=True, exist_ok=False)
    job_path = args.output / "job.json"
    job_path.write_text(json.dumps(job, indent=2, allow_nan=False), encoding="utf-8")
    evidence.update(
        source_job_sha256=original_hash, revised_job_sha256=file_hash(job_path)
    )
    (args.output / "contact_evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                contacts=len(evidence["contacts"]),
                job_sha256=evidence["revised_job_sha256"],
            )
        )
    )


if __name__ == "__main__":
    main()
