"""Audit extra annotations without automatically assigning them to training."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingestion"))
from torwic_loader import CLASSES, calibration

SEMANTIC_IDS = {
    "unlabeled": 0,
    "background": 0,
    "driveable_ground": 1,
    "ceiling": 2,
    "ego_vehicle": 3,
    "wall_fence_pillar": 4,
    "misc_static_feature": 5,
    "rack_shelf": 6,
    "goods_material": 7,
    "fixed_machinery": 8,
    "cart_pallet_jack": 9,
    "pylon_cone": 10,
    "text_region": 11,
    "misc_non_static_feature": 12,
    "person": 13,
    "fork_truck": 14,
    "misc_dynamic_feature": 15,
}


def semantic_from_instances(mask, label_mapping):
    """Map per-file instance indices to semantic IDs without an identity fallback.

    Values below 16 are still instance IDs. Unknown categories, conflicting
    mappings and unmapped pixels fail closed; zero is retained as background.
    """
    if mask.ndim != 2 or mask.dtype not in (np.uint8, np.uint16):
        raise ValueError("Expected an integer indexed mask")
    lookup = {0: 0}
    for category, entries in label_mapping.items():
        if entries is None:
            continue
        entries = entries if isinstance(entries, list) else [entries]
        for entry in entries:
            if category not in SEMANTIC_IDS or not isinstance(entry, dict):
                raise ValueError("Unknown semantic category or malformed label entry")
            index = entry.get("index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index <= 65535
            ):
                raise ValueError("Invalid instance index")
            semantic_id = SEMANTIC_IDS[category]
            if index in lookup and lookup[index] != semantic_id:
                raise ValueError("One instance ID maps to conflicting semantic classes")
            lookup[index] = semantic_id
    present = np.unique(mask)
    if any(int(index) not in lookup for index in present):
        raise ValueError("Indexed pixels missing from labelMapping")
    table = np.zeros(int(mask.max()) + 1, dtype=np.uint8)
    for index in present:
        table[int(index)] = lookup[int(index)]
    return table[mask], lookup


def audit_pair(folder, annotation_root, native_sizes):
    """Validate paired payloads, hash identity, and record missing provenance.

    Folder numbers are not unique trajectory IDs. Resolution mismatch prevents
    direct native pixel pairing, but also does not prove a lack of transformed
    overlap. Unknown collection provenance remains quarantined.
    """
    source = folder / "source_image.png"
    label = folder / "combined_indexedImage.png"
    metadata_path = folder / "raw_labels_2d.json"
    record = {
        "annotation_id": folder.relative_to(annotation_root).as_posix(),
        "group_id": folder.parent.name,
        "split": "quarantined",
        "training_eligible": False,
        "errors": [],
        "warnings": [],
    }
    try:
        source_bytes = source.read_bytes()
        label_bytes = label.read_bytes()
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("labelMapping"), dict
        ):
            raise ValueError("Expected annotation metadata with labelMapping")
        with Image.open(source) as image:
            image.load()
            rgb = np.array(image)
            source_mode = image.mode
            size = image.size
        with Image.open(label) as image:
            image.load()
            mask = np.array(image)
        if (
            source_mode != "RGB"
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
        ):
            raise ValueError("Expected uint8 RGB source")
        if mask.shape != rgb.shape[:2] or mask.dtype not in (np.uint8, np.uint16):
            raise ValueError("Source and integer semantic mask shapes/types disagree")
        semantic, lookup = semantic_from_instances(mask, metadata["labelMapping"])
        counts = np.bincount(semantic.ravel(), minlength=len(CLASSES))
        record.update(
            source_size=list(size),
            label_dtype=str(mask.dtype),
            class_ids_present=np.flatnonzero(counts).tolist(),
            instance_ids_present=np.unique(mask).tolist(),
            instance_to_semantic_id={str(k): v for k, v in lookup.items()},
            semantic_remap_required=not np.array_equal(mask, semantic),
            semantic_pixel_sha256=hashlib.sha256(semantic.tobytes()).hexdigest(),
            native_trajectory_size_compatible=tuple(size) in native_sizes,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            source_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            mask_sha256=hashlib.sha256(label_bytes).hexdigest(),
            metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
            metadata_keys=sorted(metadata),
            collection_date=None,
            trajectory_id=None,
        )
        if tuple(size) not in native_sizes:
            record["warnings"].append(
                "Source resolution differs from native trajectory cameras; no direct pixel pairing"
            )
        record["warnings"].append(
            "No verified collection-date/trajectory mapping; transformed overlap with held-out data is not ruled out"
        )
        # Integrity is separate from semantic annotation accuracy.
        record["payload_integrity"] = "passed"
    except (OSError, ValueError, KeyError, TypeError) as error:
        record["errors"].append(f"{type(error).__name__}: {error}")
        record["payload_integrity"] = "failed"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="TorWIC dataset root")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    annotations = args.annotations.resolve()
    output = args.output.resolve()
    if output.is_relative_to(root) or output.is_relative_to(annotations):
        parser.error("Audit output must be outside raw/annotation sources")
    cal = calibration(root)
    native_sizes = {
        (int(cal[c + ".width"]), int(cal[c + ".height"]))
        for c in ["Camera1", "Camera2"]
    }
    folders = sorted(path.parent for path in annotations.rglob("raw_labels_2d.json"))
    if not folders:
        parser.error("No annotation metadata found")
    records = [audit_pair(folder, annotations, native_sizes) for folder in folders]
    identities = Counter(
        r.get("source_pixel_sha256") for r in records if not r["errors"]
    )
    for record in records:
        record["duplicate_source_identity"] = (
            identities.get(record.get("source_pixel_sha256"), 0) > 1
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "annotation_manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    errors = sum(bool(r["errors"]) for r in records)
    summary = {
        "status": (
            "integrity_passed_provenance_quarantined"
            if not errors
            else "integrity_failed"
        ),
        "annotation_pairs": len(records),
        "groups": len({r["group_id"] for r in records}),
        "payload_pairs_passed": len(records) - errors,
        "payload_pairs_failed": errors,
        "source_dimensions": dict(
            Counter(
                "x".join(map(str, r["source_size"])) for r in records if not r["errors"]
            )
        ),
        "unique_source_pixel_identities": len(identities),
        "duplicate_source_identities": sum(count > 1 for count in identities.values()),
        "native_resolution_compatible_pairs": sum(
            r.get("native_trajectory_size_compatible", False) for r in records
        ),
        "training_eligible_pairs": 0,
        "quarantined_pairs": len(records),
        "trajectory_sensor_files_opened": 0,
        "semantic_remap_required_pairs": sum(
            r.get("semantic_remap_required", False) for r in records
        ),
        "label_schema": "Per-file instance indices mapped through labelMapping to fixed semantic classes",
        "limitations": [
            "Unknown dates/trajectories; neither folder IDs nor image size establish split membership.",
            "No transformed/cropped overlap search against trajectories; lack of exact native pairing is not leakage clearance.",
            "Payload integrity does not establish annotation accuracy. No samples used for training.",
        ],
        "required_release_evidence": "Verified source scene/collection and group-wise train/validation/test assignment, including transformed overlap review.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
