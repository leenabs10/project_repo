"""Human-review evidence: explicitly labelled previews linked to lossless originals."""

from html import escape
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.adapters.storage.local import source_path


def _depth_preview(path: Path, reference: dict) -> Image.Image:
    """Fixed 0-8 metre preview; black means invalid, colour clipping is display only."""
    with Image.open(path) as image:
        raw = np.array(image)
    valid = (raw != reference["invalid_raw_value"]) & (raw > 0)
    fraction = np.clip(
        raw.astype(np.float32) * reference["scale_to_metres"] / 8.0, 0, 1
    )
    colours = np.stack(
        (fraction * 240, fraction * 200, 200 - fraction * 140), axis=-1
    ).astype(np.uint8)
    colours[~valid] = 0
    return Image.fromarray(colours)


def _thumbnail(image: Image.Image, nearest=False) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(
        (240, 135), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    )
    return image


def write_review(output: Path, clean_records: list[dict], attempts: list[dict]) -> None:
    """Export all accepted previews and a bounded overview, without approving them.

    Labels stay at their declared provenance. The generic preview does not assume
    goods class IDs or convert model predictions into reference annotations.
    """
    review = output / "review"
    review.mkdir(exist_ok=False)
    by_id = {record["sample_id"]: record for record in clean_records}
    passed = [
        attempt for attempt in attempts if attempt["automated_status"] == "passed"
    ]
    headings = (
        "Clean RGB",
        "Condition RGB",
        "Clean depth 0-8m",
        "Input depth 0-8m",
        "Observed (white)",
        "Hidden (white)",
    )
    panels, html_rows = [], []
    for index, attempt in enumerate(passed):
        clean, record = by_id[attempt["source_sample_id"]], attempt["record"]
        panel = Image.new("RGB", (6 * 250, 185), "white")
        draw = ImageDraw.Draw(panel)
        draw.text(
            (5, 5),
            f"{attempt['source_sample_id']} | {attempt['condition']['condition_id']} | qualification pending",
            fill="black",
        )
        paths = (
            clean["modalities"]["rgb"]["path"],
            record["modalities"]["rgb"]["path"],
            clean["modalities"]["depth"]["path"],
            record["modalities"]["depth"]["path"],
            attempt["masks"]["observed_valid"],
            attempt["masks"]["hidden_valid"],
        )
        for column, (heading, relative) in enumerate(zip(headings, paths)):
            path = source_path(output, relative)
            if column in (2, 3):
                image = _depth_preview(path, clean["modalities"]["depth"])
            else:
                with Image.open(path) as original:
                    image = original.copy()
            panel.paste(_thumbnail(image, nearest=column >= 2), (column * 250 + 5, 45))
            draw.text((column * 250 + 5, 27), heading, fill="black")
        filename = attempt["variant_id"] + ".png"
        panel.save(review / filename)
        if index < 12:
            panels.append(panel)
        links = " ".join(
            f'<a href="../{escape(relative, quote=True)}">{escape(heading)}</a>'
            for heading, relative in zip(headings, paths)
        )
        semantic = clean["modalities"].get("semantic", {})
        if semantic.get("status") == "available":
            links += f' <a href="../{escape(semantic["path"], quote=True)}">Unchanged semantic mask (declared source labels)</a>'
        html_rows.append(
            f'<section><h2>{escape(attempt["source_sample_id"])} / {escape(attempt["condition"]["condition_id"])}</h2><img loading="lazy" src="{filename}" alt="Paired review preview"><p>{links}</p><pre>{escape(json.dumps(attempt["metrics"], indent=2))}</pre></section>'
        )
    if panels:
        overview = Image.new("RGB", (1500, 185 * len(panels)), "white")
        for index, panel in enumerate(panels):
            overview.paste(panel, (0, index * 185))
        overview.save(review / "overview.png")
    rejected = [
        {
            "source": attempt["source_sample_id"],
            "condition": attempt["condition"]["condition_id"],
            "reasons": attempt["rejection_reasons"],
        }
        for attempt in attempts
        if attempt["automated_status"] == "rejected"
    ]
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Sensor condition review</title>
<style>body{font:16px system-ui;margin:24px;background:#f6f6f6;color:#171717}section{background:white;padding:16px;margin:20px 0}img{max-width:100%;height:auto}a{display:inline-block;margin-right:14px}pre{white-space:pre-wrap}</style>
<h1>Sensor condition review - approval pending</h1>
<p>Check native images through the links. Preview depth uses a fixed 0-8 m display range; black is invalid. White masks mean true. Native payloads are unchanged by display resizing.</p>
<p>Review RGB appearance/clipping, spatial mask effects and source-label correctness separately. Transferred missingness is a controlled stress pattern. LiDAR is unchanged; no physical or joint sensor fidelity is certified.</p>
<p>All variants remain training-ineligible. Existing predicted labels need an independent reference review for the goods benchmark.</p>"""
    html += "".join(html_rows)
    html += (
        "<h2>Automated rejections</h2><pre>"
        + escape(json.dumps(rejected, indent=2))
        + "</pre></html>"
    )
    (review / "index.html").write_text(html, encoding="utf-8")
    with (review / "decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "variant_id",
                "source_sample_id",
                "appearance_decision",
                "mask_decision",
                "notes",
            ],
        )
        writer.writeheader()
        for attempt in passed:
            writer.writerow(
                {
                    "variant_id": attempt["variant_id"],
                    "source_sample_id": attempt["source_sample_id"],
                    "appearance_decision": "pending",
                    "mask_decision": "pending",
                    "notes": "",
                }
            )
