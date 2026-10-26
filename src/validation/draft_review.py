"""Immutable native review packets preloaded from source-bound correction candidates."""

import hashlib
from html import escape
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.adapters.tasks.goods import load_definition, validate_target
from src.validation.artifacts import (
    artifact_hashes,
    verify_snapshot,
    write_implementation_snapshot,
    write_json,
)
from src.validation.reference_labels import _editor


def prepare_draft_review(
    packet, packet_sha, drafts, drafts_sha, task, output, source_root
):
    """Copy native drafts into a new review packet; original snapshots stay intact.

    The new binding retains the draft target/submission and parent packet hashes.
    Only train/development membership from the original packet is permitted.
    """
    packet, drafts, task, source_root = map(Path, (packet, drafts, task, source_root))
    original = verify_snapshot(packet, packet_sha)
    draft_profile = verify_snapshot(drafts, drafts_sha)
    definition = load_definition(task)
    if original["manifest_sha256"] != draft_profile["manifest_sha256"]:
        raise ValueError("Draft manifest differs from source packet")
    if definition["task_id"] != original["task_id"]:
        raise ValueError("Draft task identity differs from review task")
    entries = json.loads((packet / "references.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (drafts / "references.jsonl").read_text().splitlines()
    ]
    index = {row["sample_id"]: row for row in rows}
    if len(index) != len(rows) or set(index) != {
        item["binding"]["sample_id"] for item in entries
    }:
        raise ValueError("Draft membership must exactly match the source packet")
    for entry in entries:
        binding, draft = entry["binding"], index[entry["binding"]["sample_id"]]
        if (
            binding["split"] not in {"train", "validation"}
            or draft["split"] != binding["split"]
            or draft["group_id"] != entry["record"]["group_id"]
            or draft.get("reviewer_kind") != "ai"
            or draft["reference_origin"] != "ai_reviewed_model_assisted"
        ):
            raise ValueError("Expected matching train/development AI corrections")
        for name in ("rgb", "semantic"):
            digest = binding[f"source_{name}_sha256"]
            if (
                draft[f"source_{name}_sha256"] != digest
                or file_hash(
                    source_path(
                        source_root, entry["record"]["modalities"][name]["path"]
                    )
                )
                != digest
            ):
                raise ValueError("Draft/source payload binding changed")
        if (
            file_hash(source_path(drafts, draft["target_path"]))
            != draft["target_sha256"]
        ):
            raise ValueError("Draft target checksum mismatch")
    output = reserve_output(output, (packet, drafts, task, source_root))
    write_json(
        output / "profile.json", {"status": "running", "training_eligible": False}
    )
    try:
        shutil.copyfile(task, output / "task.json")
        template = (
            Path(__file__).parent / "templates/native_mask_editor.html"
        ).read_text(encoding="utf-8")
        prepared, links = [], []
        for entry in entries:
            original_binding = entry["binding"]
            draft = index[original_binding["sample_id"]]
            identity = hashlib.sha256(
                original_binding["sample_id"].encode()
            ).hexdigest()[:24]
            folder = output / "samples" / identity
            folder.mkdir(parents=True)
            source_rgb = source_path(packet, entry["folder"] + "/rgb.png")
            source_mask = source_path(drafts, draft["target_path"])
            with Image.open(source_mask) as image:
                target = np.array(image)
            width, height = entry["record"]["image_size"]
            validate_target(
                target, (height, width), definition["maximum_ignore_fraction"]
            )
            shutil.copyfile(source_rgb, folder / "rgb.png")
            shutil.copyfile(source_mask, folder / "candidate.png")
            binding = {
                **original_binding,
                "protocol_sha256": file_hash(task),
                "candidate_sha256": draft["target_sha256"],
                "parent_packet_profile_sha256": packet_sha,
                "parent_draft_profile_sha256": drafts_sha,
                "parent_submission_sha256": draft["submission_sha256"],
                "parent_candidate_sha256": original_binding["candidate_sha256"],
            }
            (folder / "index.html").write_text(
                _editor(template, binding, definition, width, height), encoding="utf-8"
            )
            prepared.append(
                {
                    "binding": binding,
                    "record": entry["record"],
                    "folder": folder.relative_to(output).as_posix(),
                }
            )
            links.append(
                f'<li data-sample="{escape(binding["sample_id"], quote=True)}"><a href="samples/{identity}/index.html">{escape(binding["sample_id"])} ({binding["split"]})</a> <span class="progress">not saved</span></li>'
            )
        write_json(output / "references.json", prepared)
        index_html = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Native goods reference editor</title>
<style>body{font:16px system-ui;margin:24px;max-width:1100px;background:#f5f6f8;color:#182230}li{padding:7px}a{color:#164ca0}.progress{margin-left:12px}button{font:inherit;padding:10px}</style>
<h1>Native goods reference editor</h1><p>36 native annotations: goods are magenta, background is cyan, ignored pixels are orange. Empty pallets and handling containers are background; identifiable loads are goods.</p>
<p>Drafts are preserved. Open an image, inspect/correct the complete native mask, enter your name and confirm before saving or exporting a reviewed version.</p><p id="summary">Validated reviewed versions: 0</p><button id="validate">Validate saved cohort</button><p id="result" role="status"></p><ul>"""
        index_html = index_html.replace(
            "36 native annotations", str(len(prepared)) + " native annotations"
        )
        index_html += "".join(links) + """</ul><script>
let token='';
async function progress(){try{const r=await fetch('/api/session');if(!r.ok)return;const s=await r.json();token=s.token;document.querySelector('#summary').textContent='Validated reviewed versions: '+s.reviewed.length+' / '+s.total;for(const e of document.querySelectorAll('[data-sample]'))e.querySelector('.progress').textContent=s.reviewed.includes(e.dataset.sample)?'saved and validated':'not saved';}catch{}}
document.querySelector('#validate').onclick=async()=>{try{const r=await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:'{}'});const x=await r.json();document.querySelector('#result').textContent=r.ok?'Saved cohort validated: '+JSON.stringify(x.reference_counts)+'. Reference ready: '+x.benchmark_reference_ready:x.error;}catch(e){document.querySelector('#result').textContent=e.message;}};
progress();</script></html>"""
        (output / "index.html").write_text(index_html, encoding="utf-8")
        write_implementation_snapshot(output)
        profile = {
            "status": "prepared",
            "task_id": definition["task_id"],
            "protocol_sha256": file_hash(task),
            "manifest_sha256": original["manifest_sha256"],
            "parent_packet_profile_sha256": packet_sha,
            "parent_draft_profile_sha256": drafts_sha,
            "selected_frames": {
                split: sum(item["binding"]["split"] == split for item in prepared)
                for split in ("train", "validation")
            },
            "starting_masks": "ai_corrected_drafts",
            "qualified_reference_frames": 0,
            "training_eligible": False,
            "test_sensor_files_opened": 0,
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
