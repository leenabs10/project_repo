"""Prepare and import source-bound goods references without approving predictions."""

import hashlib
from datetime import datetime
from html import escape
import io
import json
import re
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.adapters.tasks.goods import candidate_target, load_definition, validate_target
from src.pipeline.cache import validate_membership
from src.validation.artifacts import (
    artifact_hashes,
    verify_snapshot,
    write_json,
    write_implementation_snapshot,
)


def decode_mask_runs(runs, width: int, height: int) -> np.ndarray:
    """Decode bounded native RLE; reject malformed values, overflow and truncation."""
    if (
        type(width) is not int
        or type(height) is not int
        or not 0 < width * height <= 16_777_216
    ):
        raise ValueError("Invalid or unsupported reference dimensions")
    if (
        width <= 0
        or height <= 0
        or not isinstance(runs, list)
        or not runs
        or len(runs) > width * height
    ):
        raise ValueError("Invalid reference mask runs")
    total = 0
    for run in runs:
        if (
            not isinstance(run, list)
            or len(run) != 2
            or any(type(value) is not int for value in run)
        ):
            raise ValueError("Mask runs require integer [class, count] pairs")
        value, count = run
        if value not in {0, 1, 255} or count <= 0:
            raise ValueError("Invalid class or run length")
        total += count
        if total > width * height:
            raise ValueError("Mask runs exceed native pixel count")
    if total != width * height:
        raise ValueError("Mask runs do not cover native pixel count")
    flat = np.empty(total, dtype=np.uint8)
    offset = 0
    for value, count in runs:
        flat[offset : offset + count] = value
        offset += count
    return flat.reshape(height, width)


def _editor(
    template: str, record: dict, definition: dict, width: int, height: int
) -> str:
    """Escape JSON embedded in script elements; labels themselves use textContent."""
    substitutions = {
        "__SOURCE_RECORD__": json.dumps(record).replace("<", "\\u003c"),
        "__TASK_DEFINITION__": json.dumps(definition).replace("<", "\\u003c"),
        "__WIDTH__": str(width),
        "__HEIGHT__": str(height),
    }
    # Replace once: literal placeholder-like text inside source IDs stays intact.
    return re.sub(
        r"__SOURCE_RECORD__|__TASK_DEFINITION__|__WIDTH__|__HEIGHT__",
        lambda match: substitutions[match.group(0)],
        template,
    )


def prepare_references(
    adapter,
    definition_path: Path,
    output: Path,
    *,
    required_training_ids: tuple[str, ...] = (),
) -> dict:
    """Export bounded train/development native examples; never open held-out sensors."""
    definition_path = Path(definition_path)
    definition = load_definition(definition_path)
    protocol_sha = file_hash(definition_path)
    records = list(adapter.records())
    validate_membership(records)
    if file_hash(adapter.manifest) != adapter.manifest_sha256:
        raise ValueError("Source manifest changed")
    selected = []
    by_id = {record["sample_id"]: record for record in records}
    if len(set(required_training_ids)) != len(required_training_ids) or any(
        sample_id not in by_id or by_id[sample_id]["split"] != "train"
        for sample_id in required_training_ids
    ):
        raise ValueError("Required reference sources must be unique training samples")
    for split, count in definition["minimum_reference_frames"].items():
        candidates = sorted(
            (record for record in records if record["split"] == split),
            key=lambda record: record["sample_id"],
        )
        if not candidates:
            raise ValueError(
                "Reference preparation requires train and development sources"
            )
        count = min(count, len(candidates))
        if split == "train" and required_training_ids:
            if len(required_training_ids) > count:
                raise ValueError(
                    "Required training sources exceed the reference budget"
                )
            indices = [
                index
                for index, record in enumerate(candidates)
                if record["sample_id"] in required_training_ids
            ]
            # Fill gaps in capture order instead of duplicating adjacent pilot frames.
            grid = np.arange(len(candidates))
            distance = np.full(len(candidates), len(candidates), dtype=np.int64)
            for index in indices:
                distance = np.minimum(distance, np.abs(grid - index))
            while len(indices) < count:
                index = int(np.argmax(distance))
                indices.append(index)
                distance = np.minimum(distance, np.abs(grid - index))
        else:
            indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
        selected.extend(candidates[index] for index in sorted(indices))
    for record in selected:
        if record["modalities"].get("semantic", {}).get("status") != "available":
            raise ValueError("Reference candidate semantics are unavailable")
    output = reserve_output(output, (adapter.root, adapter.manifest, definition_path))
    write_json(
        output / "profile.json", {"status": "running", "training_eligible": False}
    )
    try:
        write_implementation_snapshot(output)
        template = (Path(__file__).parent / "templates/mask_editor.html").read_text(
            encoding="utf-8"
        )
        entries, source_hashes, links = [], {}, []
        # Preserve the protocol's exact bytes: formatting is part of its checksum.
        shutil.copyfile(definition_path, output / "task.json")
        for record in selected:
            pair = adapter.read_rgb_depth(record)
            semantic_ref = record["modalities"]["semantic"]
            semantic_path = source_path(adapter.root, semantic_ref["path"])
            semantic_bytes = semantic_path.read_bytes()
            semantic_sha = hashlib.sha256(semantic_bytes).hexdigest()
            if semantic_ref.get("sha256", semantic_sha) != semantic_sha:
                raise ValueError("Semantic source checksum mismatch")
            with Image.open(io.BytesIO(semantic_bytes)) as image:
                semantic = np.array(image)
            if semantic.shape != pair.depth_raw.shape:
                raise ValueError("Semantic candidate geometry differs from native RGB")
            candidate = candidate_target(
                semantic, definition["source_candidate_class_id"]
            )
            identity = hashlib.sha256(record["sample_id"].encode()).hexdigest()[:24]
            folder = output / "samples" / identity
            folder.mkdir(parents=True)
            shutil.copyfile(
                source_path(adapter.root, record["modalities"]["rgb"]["path"]),
                folder / "rgb.png",
            )
            if file_hash(folder / "rgb.png") != pair.provenance["rgb_sha256"]:
                raise ValueError("Source image changed during export")
            Image.fromarray(candidate).save(folder / "candidate.png")
            with Image.open(folder / "candidate.png") as image:
                if not np.array_equal(np.array(image), candidate):
                    raise ValueError("Candidate mask export is not lossless")
            binding = {
                "schema_version": "dubaidata/reference/1.0",
                "task_id": definition["task_id"],
                "protocol_sha256": protocol_sha,
                "source_manifest_sha256": adapter.manifest_sha256,
                "sample_id": record["sample_id"],
                "split": record["split"],
                "source_rgb_sha256": pair.provenance["rgb_sha256"],
                "source_semantic_sha256": semantic_sha,
                "candidate_sha256": file_hash(folder / "candidate.png"),
            }
            width, height = record["image_size"]
            (folder / "index.html").write_text(
                _editor(template, binding, definition, width, height), encoding="utf-8"
            )
            entries.append(
                {
                    "binding": binding,
                    "record": record,
                    "folder": folder.relative_to(output).as_posix(),
                }
            )
            links.append(
                f'<li><a href="{folder.relative_to(output).as_posix()}/index.html">{escape(record["sample_id"])} ({record["split"]})</a></li>'
            )
            for name in ("rgb", "depth"):
                source_hashes[pair.provenance[name + "_path"]] = pair.provenance[
                    name + "_sha256"
                ]
            source_hashes[semantic_ref["path"]] = semantic_sha
        if (
            file_hash(definition_path) != protocol_sha
            or file_hash(adapter.manifest) != adapter.manifest_sha256
        ):
            raise ValueError("Definition or manifest changed during preparation")
        for relative, digest in source_hashes.items():
            if file_hash(source_path(adapter.root, relative)) != digest:
                raise ValueError("Reference source payload changed during preparation")
        write_json(output / "references.json", entries)
        (output / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Reference workspace</title><h1>Visible goods reference workspace</h1><p>Native candidates are model predictions. Complete source-bound corrections before dataset admission.</p><ul>'
            + "".join(links)
            + "</ul></html>",
            encoding="utf-8",
        )
        profile = {
            "status": "prepared",
            "task_id": definition["task_id"],
            "protocol_sha256": protocol_sha,
            "manifest_sha256": adapter.manifest_sha256,
            "selected_frames": {
                split: sum(entry["binding"]["split"] == split for entry in entries)
                for split in ("train", "validation")
            },
            "required_training_sample_ids": list(required_training_ids),
            "qualified_reference_frames": 0,
            "test_sensor_files_opened": 0,
            "training_eligible": False,
            "source_files_verified": len(source_hashes),
            "artifact_sha256": artifact_hashes(output),
        }
        write_json(output / "profile.json", profile)
        return profile
    except Exception as error:
        write_json(
            output / "profile.json",
            {"status": "failed", "error": str(error), "training_eligible": False},
        )
        raise


def import_references(
    packet: Path,
    submissions: Path,
    output: Path,
    expected_packet_sha256: str,
    *,
    source_root: Path,
) -> dict:
    """Validate explicit submissions and export corrected 0/1/255 reference masks.

    Empty submissions are a valid pending state, never sufficient training evidence.
    Every submitted mask must bind to the immutable packet and whole-image attestation.
    """
    packet, submissions = Path(packet).resolve(), Path(submissions).resolve()
    if not submissions.is_dir():
        raise ValueError("Submission directory must exist")
    profile = verify_snapshot(packet, expected_packet_sha256)
    if profile["status"] != "prepared":
        raise ValueError("Reference packet did not complete preparation")
    definition = load_definition(packet / "task.json")
    if file_hash(packet / "task.json") != profile["protocol_sha256"]:
        raise ValueError("Task definition checksum mismatch")
    entries = json.loads((packet / "references.json").read_text(encoding="utf-8"))
    by_id = {entry["binding"]["sample_id"]: entry for entry in entries}
    if len(by_id) != len(entries):
        raise ValueError("Duplicate packet reference identity")
    validated, seen = [], set()
    for path in sorted(submissions.glob("*.json")):
        if path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Submission path/size is unsupported")
        payload = path.read_bytes()
        if len(payload) > 64 * 1024 * 1024:
            raise ValueError("Reference submission exceeds the size limit")
        digest = hashlib.sha256(payload).hexdigest()
        submission = json.loads(payload)
        entry = by_id.get(submission.get("sample_id"))
        if entry is None or entry["binding"]["split"] not in {"train", "validation"}:
            raise ValueError("Submission is outside permitted reference membership")
        binding = entry["binding"]
        if set(submission) != set(binding) | {
            "reviewer",
            "reviewed",
            "reviewed_at",
            "width",
            "height",
            "mask_rle",
        }:
            raise ValueError("Unexpected or missing reference submission fields")
        if any(submission[key] != value for key, value in binding.items()):
            raise ValueError("Reference submission source binding mismatch")
        if binding["sample_id"] in seen:
            raise ValueError("Duplicate reference submission")
        seen.add(binding["sample_id"])
        if (
            submission["reviewed"] is not True
            or not isinstance(submission["reviewer"], str)
            or not submission["reviewer"].strip()
            or not isinstance(submission["reviewed_at"], str)
            or not submission["reviewed_at"].strip()
        ):
            raise ValueError(
                "Complete-image attestation and reviewer metadata are required"
            )
        if datetime.fromisoformat(submission["reviewed_at"]).tzinfo is None:
            raise ValueError("Reference attestation timestamp requires a timezone")
        for name, expected in (
            ("rgb", submission["source_rgb_sha256"]),
            ("semantic", submission["source_semantic_sha256"]),
        ):
            if (
                file_hash(
                    source_path(
                        source_root, entry["record"]["modalities"][name]["path"]
                    )
                )
                != expected
            ):
                raise ValueError("Original reference source checksum mismatch")
        if [submission["width"], submission["height"]] != entry["record"]["image_size"]:
            raise ValueError("Submitted mask geometry differs from source")
        mask = decode_mask_runs(
            submission["mask_rle"], submission["width"], submission["height"]
        )
        validate_target(
            mask,
            (submission["height"], submission["width"]),
            definition["maximum_ignore_fraction"],
        )
        # Keep metadata only; native masks/RLE are re-read one at a time at export.
        validated.append(
            (
                entry,
                {key: value for key, value in submission.items() if key != "mask_rle"},
                path,
                digest,
            )
        )
    output = reserve_output(output, (packet, submissions, source_root))
    write_json(
        output / "profile.json", {"status": "running", "training_eligible": False}
    )
    try:
        write_implementation_snapshot(output)
        references = []
        for entry, submission, path, digest in validated:
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("Reference submission changed during import")
            mask = decode_mask_runs(
                json.loads(payload)["mask_rle"],
                submission["width"],
                submission["height"],
            )
            identity = hashlib.sha256(submission["sample_id"].encode()).hexdigest()[:24]
            folder = output / "samples" / identity
            folder.mkdir(parents=True)
            rgb = source_path(packet, entry["folder"] + "/rgb.png")
            shutil.copyfile(rgb, folder / "rgb.png")
            Image.fromarray(mask).save(folder / "target.png")
            with Image.open(folder / "target.png") as image:
                if not np.array_equal(np.array(image), mask):
                    raise ValueError("Corrected reference export is not lossless")
            if (
                file_hash(path) != digest
                or file_hash(folder / "rgb.png") != submission["source_rgb_sha256"]
            ):
                raise ValueError("Reference input changed during import")
            for name, expected in (
                ("rgb", submission["source_rgb_sha256"]),
                ("semantic", submission["source_semantic_sha256"]),
            ):
                if (
                    file_hash(
                        source_path(
                            source_root, entry["record"]["modalities"][name]["path"]
                        )
                    )
                    != expected
                ):
                    raise ValueError("Original reference source changed during import")
            references.append(
                {
                    "sample_id": submission["sample_id"],
                    "split": submission["split"],
                    "group_id": entry["record"]["group_id"],
                    "task_id": definition["task_id"],
                    "protocol_sha256": profile["protocol_sha256"],
                    "source_rgb_sha256": submission["source_rgb_sha256"],
                    "source_semantic_sha256": submission["source_semantic_sha256"],
                    "rgb_path": (folder / "rgb.png").relative_to(output).as_posix(),
                    "target_path": (folder / "target.png")
                    .relative_to(output)
                    .as_posix(),
                    "target_sha256": file_hash(folder / "target.png"),
                    "reference_origin": definition["reference_origin"],
                    "reviewer": submission["reviewer"].strip(),
                    "submission_sha256": digest,
                    "image_size": entry["record"]["image_size"],
                    "ignore_fraction": float((mask == 255).mean()),
                    "goods_fraction": float((mask == 1).mean()),
                }
            )
        verify_snapshot(packet, expected_packet_sha256)
        (output / "references.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in references), encoding="utf-8"
        )
        counts = {
            split: sum(row["split"] == split for row in references)
            for split in ("train", "validation")
        }
        ready = all(
            counts[split] >= count
            for split, count in definition["minimum_reference_frames"].items()
        )
        result = {
            "status": "passed",
            "task_id": definition["task_id"],
            "protocol_sha256": profile["protocol_sha256"],
            "packet_profile_sha256": expected_packet_sha256,
            "manifest_sha256": profile["manifest_sha256"],
            "reference_counts": counts,
            "benchmark_reference_ready": ready,
            "reference_origin": definition["reference_origin"],
            "artifact_sha256": artifact_hashes(output),
        }
        write_json(output / "profile.json", result)
        return result
    except Exception as error:
        write_json(
            output / "profile.json",
            {
                "status": "failed",
                "error": str(error),
                "benchmark_reference_ready": False,
            },
        )
        raise
