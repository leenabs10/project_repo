"""Prepare selected training annotations without changing an existing cohort.

Explicit AI polygon hints start from ignored pixels. Predicted semantic classes
are hashed for source lineage, never copied into new reference labels.
"""

import hashlib
from html import escape
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.adapters.tasks.goods import load_definition, validate_target
from src.pipeline.cache import validate_membership
from src.validation.artifacts import (
    artifact_hashes,
    verify_snapshot,
    write_json,
    write_implementation_snapshot,
)
from src.validation.guided_review import apply_polygon_edits, colour_overlay
from src.validation.reference_labels import _editor


def prepare_training_extension(
    adapter, task, previous, previous_sha, specification, output
):
    """Write a separate, unqualified native packet for unique new training IDs.

    Validate every selected split before opening sensor payloads. The specification
    declares review-grid coordinates and polygons with class IDs and reasons.
    Native categorical PNGs retain uncertain regions as 255. Reviewed exports
    still pass the existing source/provenance/coverage importer before admission.
    """
    task, previous, specification = map(Path, (task, previous, specification))
    parent = verify_snapshot(previous, previous_sha)
    definition = load_definition(task)
    if (
        parent["protocol_sha256"] != file_hash(task)
        or parent["manifest_sha256"] != adapter.manifest_sha256
    ):
        raise ValueError("Extension task/manifest differs from the prior cohort")
    if file_hash(adapter.manifest) != adapter.manifest_sha256:
        raise ValueError("Source manifest changed")
    existing = {
        json.loads(line)["sample_id"]
        for line in (previous / "references.jsonl").read_text().splitlines()
    }
    spec = json.loads(specification.read_text(encoding="utf-8"))
    selections = spec["samples"]
    records = list(adapter.records())
    validate_membership(records)
    by_id = {row["sample_id"]: row for row in records}
    ids = [row["sample_id"] for row in selections]
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(
            sample_id in existing
            or sample_id not in by_id
            or by_id[sample_id]["split"] != "train"
            for sample_id in ids
        )
    ):
        raise ValueError("Extension requires unique new training samples")
    for sample_id in ids:
        record = by_id[sample_id]
        if any(
            record["modalities"].get(name, {}).get("status") != "available"
            for name in ("rgb", "semantic")
        ):
            raise ValueError("Selected training source lacks RGB/semantic binding")
    output = reserve_output(
        Path(output), (adapter.root, adapter.manifest, task, previous, specification)
    )
    write_json(
        output / "profile.json", {"status": "running", "training_eligible": False}
    )
    try:
        shutil.copyfile(task, output / "task.json")
        shutil.copyfile(specification, output / "annotation_hints.json")
        template = (
            Path(__file__).parent / "templates/native_mask_editor.html"
        ).read_text(encoding="utf-8")
        entries, links, sources = [], [], {}
        for selection in selections:
            record = by_id[selection["sample_id"]]
            payloads = {}
            for name in ("rgb", "semantic"):
                reference = record["modalities"][name]
                payload = source_path(adapter.root, reference["path"]).read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                if reference.get("sha256", digest) != digest:
                    raise ValueError("Training source checksum mismatch")
                payloads[name] = (payload, digest)
                sources[reference["path"]] = digest
            with Image.open(io.BytesIO(payloads["rgb"][0])) as image:
                rgb = np.asarray(image)
            width, height = record["image_size"]
            if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
                raise ValueError("Native RGB geometry/type differs from its record")
            # Only explicit annotations resolve unknown pixels; semantic model
            # predictions never supply reference labels for this extension.
            blank = np.full((height, width), 255, dtype=np.uint8)
            draft = apply_polygon_edits(
                blank, selection["polygons"], spec["coordinate_size"]
            )
            validate_target(draft, (height, width), 1.0)
            identity = hashlib.sha256(record["sample_id"].encode()).hexdigest()[:24]
            folder = output / "samples" / identity
            folder.mkdir(parents=True)
            (folder / "rgb.png").write_bytes(payloads["rgb"][0])
            Image.fromarray(draft).save(folder / "candidate.png")
            with Image.open(folder / "candidate.png") as image:
                if not np.array_equal(np.asarray(image), draft):
                    raise ValueError("Draft PNG is not lossless")
            colour_overlay(rgb, draft).save(folder / "display_draft.png")
            binding = {
                "schema_version": "dubaidata/reference/1.0",
                "task_id": definition["task_id"],
                "protocol_sha256": file_hash(task),
                "source_manifest_sha256": adapter.manifest_sha256,
                "sample_id": record["sample_id"],
                "split": "train",
                "source_rgb_sha256": payloads["rgb"][1],
                "source_semantic_sha256": payloads["semantic"][1],
                "candidate_sha256": file_hash(folder / "candidate.png"),
                "parent_reference_profile_sha256": previous_sha,
                "annotation_hints_sha256": file_hash(specification),
            }
            (folder / "index.html").write_text(
                _editor(template, binding, definition, width, height), encoding="utf-8"
            )
            relative = folder.relative_to(output).as_posix()
            entries.append(
                {
                    "binding": binding,
                    "record": record,
                    "folder": relative,
                    "draft_ignore_fraction": float((draft == 255).mean()),
                }
            )
            links.append(
                f'<li><a href="{relative}/index.html">{escape(record["sample_id"])}</a> — Prepared annotation; validation pending</li>'
            )
        for relative, digest in sources.items():
            if file_hash(source_path(adapter.root, relative)) != digest:
                raise ValueError("Training source changed during preparation")
        if (
            file_hash(task) != file_hash(output / "task.json")
            or file_hash(specification) != file_hash(output / "annotation_hints.json")
            or file_hash(adapter.manifest) != adapter.manifest_sha256
        ):
            raise ValueError("Preparation input changed")
        write_json(output / "references.json", entries)
        (output / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Training reference extension</title><style>body{font:17px system-ui;max-width:1100px;margin:32px auto}li{padding:12px}</style><h1>Training reference extension</h1><p>Prepared annotation workspace. Existing reviewed references are preserved. Magenta: identifiable goods; cyan: background, including empty pallets and handling equipment; orange: ignored uncertainty. Complete the full mask before saving a reviewed version.</p><ul>'
            + "".join(links)
            + "</ul></html>",
            encoding="utf-8",
        )
        write_implementation_snapshot(output)
        profile = {
            "status": "prepared",
            "task_id": definition["task_id"],
            "protocol_sha256": file_hash(task),
            "manifest_sha256": adapter.manifest_sha256,
            "parent_reference_profile_sha256": previous_sha,
            "selected_frames": {"train": len(entries), "validation": 0},
            "qualified_reference_frames": 0,
            "annotation_origin": "ai_polygon_hints",
            "training_eligible": False,
            "test_sensor_files_opened": 0,
            "development_sensor_files_opened": 0,
            "source_files_verified": len(sources),
            "artifact_sha256": artifact_hashes(output),
        }
        write_json(output / "profile.json", profile)
        return profile
    except Exception as error:
        write_json(
            output / "profile.json",
            {"status": "failed", "training_eligible": False, "error": str(error)},
        )
        raise
