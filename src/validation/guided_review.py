"""Prepare reproducible polygon refinements and an editable comparison workspace.

This tool applies explicit annotation decisions; it does not discover objects or
certify semantic accuracy. Original packets and saved review state stay immutable.
"""

import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.adapters.tasks.goods import load_definition, validate_target
from src.pipeline.cache import validate_membership
from src.validation.artifacts import (
    artifact_hashes,
    verify_snapshot,
    write_implementation_snapshot,
    write_json,
)
from src.validation.reference_labels import _editor, decode_mask_runs

PALETTE = {0: (0, 180, 210), 1: (235, 70, 210), 255: (255, 160, 0)}
PAGE_STYLE = """body{font:17px/1.55 system-ui;margin:24px auto;padding:0 20px;
max-width:1400px;background:#f5f6f8;color:#182230}a{color:#164ca0}
article,aside,details{background:white;padding:18px;border:1px solid #ccd3dc;
border-radius:10px;margin:16px 0}img{max-width:100%;height:auto}table{border-collapse:collapse}
td,th{border:1px solid #ccd3dc;padding:10px;text-align:left}.cards{display:grid;
grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.badge{font-weight:bold}
button{font:inherit;padding:10px}.progress{margin-left:12px}"""


def apply_polygon_edits(mask, edits, coordinate_size):
    """Return a uint8 copy with ordered polygon edits; preserve all other pixels.

    Coordinates refer to a declared [width, height] review grid. Rasterization is
    directly into native geometry, without interpolating categorical labels.
    Optional boundary widths are native pixels reserved as uncertain because a
    coarse annotation outline cannot establish exact pixel membership.
    Malformed IDs, geometry, coordinates, or unsupported boundary widths fail.
    """
    # A new annotation may start wholly unknown. Admission still requires the
    # protocol's coverage limit; editing must never force unknown pixels to background.
    if mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("Expected a uint8 HxW annotation with IDs 0/1/255")
    if (
        not isinstance(coordinate_size, list)
        or len(coordinate_size) != 2
        or any(type(value) is not int or value <= 0 for value in coordinate_size)
    ):
        raise ValueError("Coordinate size requires positive integer width and height")
    width, height = coordinate_size
    native_height, native_width = mask.shape
    output = Image.fromarray(mask.copy())
    drawing = ImageDraw.Draw(output)
    for edit in edits:
        value = edit.get("class_id")
        boundary = edit.get("uncertain_boundary_pixels", 0)
        polygon = np.asarray(edit.get("polygon"), dtype=float)
        if (
            type(value) is not int
            or value not in PALETTE
            or type(boundary) is not int
            or not 0 <= boundary <= 8
            or not edit.get("reason")
            or polygon.ndim != 2
            or polygon.shape[1] != 2
            or len(polygon) < 3
            or not np.isfinite(polygon).all()
            or np.any(polygon < 0)
            or np.any(polygon[:, 0] > width)
            or np.any(polygon[:, 1] > height)
        ):
            raise ValueError("Invalid annotation polygon, class, boundary, or reason")
        # Endpoint coordinates may lie on the image extent; clip to its last pixel.
        points = np.rint(polygon * [native_width / width, native_height / height])
        points = np.clip(points, [0, 0], [native_width - 1, native_height - 1])
        points = [tuple(point) for point in points.astype(int).tolist()]
        if len(set(points)) < 3:
            raise ValueError("Annotation polygon collapses at native geometry")
        drawing.polygon(points, fill=value)
        if boundary:
            drawing.line(points + points[:1], fill=255, width=boundary)
    return np.array(output)


def encode_runs(mask):
    """Encode row-major IDs and verify exact round-trip with the import decoder."""
    flat = mask.ravel()
    boundaries = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, flat.size]
    runs = [
        [int(flat[start]), int(end - start)]
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    height, width = mask.shape
    if not np.array_equal(mask, decode_mask_runs(runs, width, height)):
        raise ValueError("Refinement RLE round-trip mismatch")
    return runs


def colour_overlay(rgb, mask):
    """Visualize categorical targets; this RGB derivative is never a label export."""
    result = rgb.astype(float).copy()
    for value, colour in PALETTE.items():
        selected = mask == value
        strength = 0.22 if value == 0 else 0.48
        result[selected] = (
            result[selected] * (1 - strength) + np.array(colour) * strength
        )
    return Image.fromarray(np.rint(result).astype(np.uint8))


def write_comparison(folder, rgb, previous, refined, sample_id):
    """Save native overlays plus a display-only RGB/previous/refined contact panel."""
    before = colour_overlay(rgb, previous)
    after = colour_overlay(rgb, refined)
    before.save(folder / "previous_overlay.png")
    after.save(folder / "refined_overlay.png")
    sheet = Image.new("RGB", (1920, 410), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(
        (
            ("RGB photograph", Image.fromarray(rgb)),
            ("Previous annotation", before),
            ("Current annotation", after),
        )
    ):
        draw.text((index * 640 + 10, 8), label, fill="black")
        sheet.paste(image.resize((640, 360)), (index * 640, 30))
    draw.text(
        (10, 392),
        sample_id + " | magenta goods; cyan background; orange ignore",
        fill="black",
    )
    sheet.save(folder / "comparison.png")


def _sample_guidance(notes, previous_link, next_link):
    """Insert escaped, non-executable annotation notes beside the native editor."""
    items = "".join(f"<li>{escape(note)}</li>" for note in notes)
    return f"""<aside style="background:white;padding:16px;border:1px solid #a8b1bd">
<p><strong>Annotation workspace.</strong> Painting changes a working copy.
Source annotations and version history are preserved.</p>
<p><a href="../../guide.html">Colour guide and three real-image examples</a> |
<a href="comparison.png" target="_blank">RGB / previous / refined comparison</a> |
<a href="{previous_link}">Previous image</a> | <a href="{next_link}">Next image</a></p>
<details open><summary><strong>What to check in this image</strong></summary><ul>{items}</ul></details>
<p><strong>Pink:</strong> fill visible goods surfaces. <strong>Cyan:</strong> equipment,
people, empty pallets/handling containers and visible gaps. <strong>Orange:</strong>
unresolved identity or boundaries. Colour does not determine the real object class.</p>
<p>Start in RGB only. Compare the overlay, zoom to 200% for edges, use a small brush,
then inspect the whole image. Clear regions should be labelled; do not guess uncertain pixels.</p>
</aside>"""


def prepare_guided_review(packet, packet_sha, refinements, output, source_root):
    """Create a new source-bound draft packet with explicit edits and comparison UI.

    Only existing train/development members are allowed. RGB and semantic source
    hashes are verified without opening depth, LiDAR, or held-out payloads. Every
    new mask retains its parent mask hash, edit specification and annotation origin.
    This prepares no human attestations, admitted dataset, or training run.
    """
    packet, refinements, source_root = map(Path, (packet, refinements, source_root))
    parent = verify_snapshot(packet, packet_sha)
    definition = load_definition(packet / "task.json")
    specification = json.loads(refinements.read_text(encoding="utf-8"))
    if specification.get("reviewer_kind") != "ai":
        raise ValueError("Refinements must declare AI authorship")
    entries = json.loads((packet / "references.json").read_text(encoding="utf-8"))
    validate_membership([entry["record"] for entry in entries])
    expected = {entry["binding"]["sample_id"] for entry in entries}
    if set(specification["samples"]) != expected:
        raise ValueError("Refinement membership must exactly match the review packet")
    # Validate metadata before reading any sensor path, including rejected test entries.
    for entry in entries:
        binding, record = entry["binding"], entry["record"]
        if (
            binding["split"] not in {"train", "validation"}
            or record["split"] != binding["split"]
            or record["sample_id"] != binding["sample_id"]
            or binding["protocol_sha256"] != file_hash(packet / "task.json")
        ):
            raise ValueError("Only source-bound train/development entries are allowed")
    output = reserve_output(output, (packet, refinements, source_root))
    write_json(
        output / "profile.json", {"status": "running", "training_eligible": False}
    )
    try:
        shutil.copyfile(packet / "task.json", output / "task.json")
        shutil.copyfile(refinements, output / "refinements.json")
        template = (
            Path(__file__).parent / "templates/native_mask_editor.html"
        ).read_text(encoding="utf-8")
        prepared, statistics, cards = [], [], []
        for index, original_entry in enumerate(entries):
            entry = deepcopy(original_entry)
            binding = entry["binding"]
            sample_id = binding["sample_id"]
            record = entry["record"]
            for modality in ("rgb", "semantic"):
                if (
                    file_hash(
                        source_path(source_root, record["modalities"][modality]["path"])
                    )
                    != binding[f"source_{modality}_sha256"]
                ):
                    raise ValueError("Source payload binding changed")
            parent_folder = source_path(packet, entry["folder"])
            if (
                file_hash(parent_folder / "candidate.png")
                != binding["candidate_sha256"]
            ):
                raise ValueError("Parent candidate binding changed")
            if file_hash(parent_folder / "rgb.png") != binding["source_rgb_sha256"]:
                raise ValueError("Packet RGB differs from source")
            with Image.open(parent_folder / "candidate.png") as image:
                previous = np.array(image)
            width, height = record["image_size"]
            # Sparse correction hints may exceed review coverage requirements. Check their
            # format here; enforce the unchanged coverage limit on the new draft.
            validate_target(previous, (height, width), 1.0)
            selected = specification["samples"][sample_id]
            refined = apply_polygon_edits(
                previous, selected["edits"], specification["coordinate_size"]
            )
            validate_target(
                refined, (height, width), definition["maximum_ignore_fraction"]
            )
            folder = output / entry["folder"]
            folder.mkdir(parents=True)
            shutil.copyfile(parent_folder / "rgb.png", folder / "rgb.png")
            shutil.copyfile(
                parent_folder / "candidate.png", folder / "previous_candidate.png"
            )
            Image.fromarray(refined).save(folder / "candidate.png")
            with Image.open(folder / "candidate.png") as image:
                if not np.array_equal(refined, np.array(image)):
                    raise ValueError("Refinement PNG round-trip mismatch")
            binding.update(
                previous_review_packet_sha256=packet_sha,
                previous_candidate_sha256=binding["candidate_sha256"],
                candidate_sha256=file_hash(folder / "candidate.png"),
                refinement_specification_sha256=file_hash(refinements),
            )
            # Candidate provenance stays separate from confirmed saves and
            # their review state.
            write_json(
                folder / "ai_draft.json",
                {
                    **binding,
                    "reviewer_kind": "ai",
                    "reviewed": False,
                    "reference_origin": "ai_refined_draft",
                    "human_review_pending": True,
                    "width": width,
                    "height": height,
                    "mask_rle": encode_runs(refined),
                },
            )
            with Image.open(folder / "rgb.png") as image:
                rgb = np.array(image)
            write_comparison(folder, rgb, previous, refined, sample_id)
            neighbours = []
            for offset in (-1, 1):
                neighbour = entries[(index + offset) % len(entries)]["folder"]
                neighbours.append("../" + Path(neighbour).name + "/index.html")
            guidance = _sample_guidance(selected["notes"], *neighbours)
            html = _editor(template, binding, definition, width, height)
            (folder / "index.html").write_text(
                html.replace("<!-- REVIEW_GUIDANCE -->", guidance), encoding="utf-8"
            )
            changed = int((previous != refined).sum())
            statistics.append(
                {
                    "sample_id": sample_id,
                    "split": binding["split"],
                    "changed_pixels": changed,
                    "edit_count": len(selected["edits"]),
                    "goods_fraction": float((refined == 1).mean()),
                    "ignore_fraction": float((refined == 255).mean()),
                    "previous_ignore_fraction": float((previous == 255).mean()),
                    "candidate_sha256": binding["candidate_sha256"],
                    "previous_candidate_sha256": binding["previous_candidate_sha256"],
                    "human_review_pending": True,
                }
            )
            prepared.append(entry)
            link = escape(entry["folder"], quote=True)
            cards.append(
                f'<article data-sample="{escape(sample_id, quote=True)}"><h2>{escape(sample_id)}</h2><p>{escape(binding["split"])} · Reference candidate · <span class="progress">not saved</span></p><a href="{link}/index.html"><img src="{link}/refined_overlay.png" alt="Annotation overlay for {escape(sample_id, quote=True)}"></a><p>{escape(selected["notes"][0])}</p><a href="{link}/index.html">Open and correct this image</a> | <a href="{link}/comparison.png">Before / after</a></article>'
            )
        write_json(output / "references.json", prepared)
        write_json(output / "refinement_statistics.json", statistics)
        help_template = (
            Path(__file__).parent / "templates/goods_review_guide.html"
        ).read_text(encoding="utf-8")
        help_template = help_template.replace("36-image", f"{len(prepared)}-image")
        examples = []
        # Example identities are supplied by the local annotation specification.
        for item in specification["examples"]:
            entry = next(
                row
                for row in prepared
                if row["binding"]["sample_id"] == item["sample_id"]
            )
            examples.append(
                f'<article><h2>{escape(item["title"])}</h2><p>{escape(item["description"])}</p><img src="{escape(entry["folder"], quote=True)}/comparison.png" alt="RGB, previous candidate and refined candidate"><p><a href="{escape(entry["folder"], quote=True)}/index.html">Open this real example in the editor</a></p></article>'
            )
        (output / "guide.html").write_text(
            help_template.replace("<!-- EXAMPLES -->", "".join(examples)),
            encoding="utf-8",
        )
        index_page = (
            f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Goods annotation references</title><style>{PAGE_STYLE}</style><h1>Goods annotation references</h1><aside><p><strong>Reference candidates — qualification pending. Training disabled.</strong></p><p>Magenta: goods · Cyan: background · Orange: unresolved pixels.</p><p><a href="guide.html">Start here: colour rules, three real examples and painting steps</a></p><p id="summary">Saved reference versions: 0 / {len(prepared)}</p><button id="validate">Validate saved cohort</button><p id="result" role="status"></p></aside><div class="cards">'
            + "".join(cards)
            + "</div>"
        )
        # A sparse-hint packet may have no index script. Every generated workspace
        # needs its own progress and validation controls, independent of its parent.
        index_page += (
            Path(__file__).parent / "templates/review_progress.html"
        ).read_text(encoding="utf-8")
        (output / "index.html").write_text(index_page + "</html>", encoding="utf-8")
        write_implementation_snapshot(output)
        profile = {
            **{
                key: parent[key]
                for key in (
                    "task_id",
                    "manifest_sha256",
                    "protocol_sha256",
                    "selected_frames",
                )
            },
            "status": "prepared",
            "previous_review_packet_sha256": packet_sha,
            "refinement_specification_sha256": file_hash(refinements),
            "starting_masks": "ai_polygon_refined_drafts",
            "reviewer_kind": "ai",
            "frames_with_changed_pixels": sum(
                row["changed_pixels"] > 0 for row in statistics
            ),
            "qualified_reference_frames": 0,
            "training_eligible": False,
            "test_sensor_files_opened": 0,
            "all_png_and_rle_roundtrips_exact": True,
            "limitations": [
                "AI annotation retains semantic and boundary uncertainty.",
                "Coverage fractions are not accuracy scores.",
                "Training and development corrections need human review; no held-out evaluation was performed.",
            ],
            "artifact_sha256": artifact_hashes(output),
        }
        write_json(output / "profile.json", profile)
        verify_snapshot(packet, packet_sha)
        return profile
    except Exception as error:
        write_json(
            output / "profile.json",
            {"status": "failed", "error": str(error), "training_eligible": False},
        )
        raise


def main():
    """Prepare a new local review packet; never train or mark human approval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--refinements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    profile = prepare_guided_review(
        args.packet, args.packet_sha256, args.refinements, args.output, args.source_root
    )
    print(
        json.dumps(
            {key: value for key, value in profile.items() if key != "artifact_sha256"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
