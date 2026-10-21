"""Render current annotation pages without rewriting immutable packet artifacts."""

from html import escape
import json
from pathlib import Path

from src.validation.reference_labels import _editor


def reference_page(packet, entries, relative, reviewer_name=""):
    """Return a professional current view, or None for a non-page asset.

    Source/task bindings come from the verified packet. Presentation comes from
    current templates; original HTML, masks and provenance hashes stay unchanged.
    A displayed name never sets the confirmation checkbox or saves a reference.
    """
    packet = Path(packet)
    definition = json.loads((packet / "task.json").read_text(encoding="utf-8"))
    if relative == "index.html":
        links = []
        for entry in sorted(
            entries.values(),
            key=lambda item: (item["binding"]["split"], item["binding"]["sample_id"]),
        ):
            binding = entry["binding"]
            links.append(
                f'<li data-sample="{escape(binding["sample_id"], quote=True)}">'
                f'<a href="{escape(entry["folder"], quote=True)}/index.html">{escape(binding["sample_id"])}</a>'
                f' ({escape(binding["split"])}) <span class="progress">Loading status</span></li>'
            )
        progress = (Path(__file__).parent / "templates/review_progress.html").read_text(
            encoding="utf-8"
        )
        return (
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Annotation references</title>'
            "<style>body{font:17px system-ui;max-width:1100px;margin:32px auto;line-height:1.6}li{padding:8px}.progress{margin-left:12px}button{font:inherit;padding:10px}</style>"
            "<h1>Annotation references</h1><p>Native-resolution annotations with source-bound version history. "
            "Magenta: goods; cyan: background; orange: ignored pixels. Empty pallets and handling containers "
            'are background; identifiable loads are goods.</p><p><a href="guide.html">Annotation standard</a></p>'
            '<p id="summary">Loading saved references</p><button id="validate">Validate reference dataset</button>'
            '<p id="result" role="status"></p><ul>'
            + "".join(links)
            + "</ul>"
            + progress
            + "</html>"
        )
    if relative == "guide.html":
        return (
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Annotation standard</title>'
            "<style>body{font:18px system-ui;max-width:1000px;margin:32px auto;line-height:1.7}li{padding:8px}</style>"
            '<a href="index.html">All references</a><h1>Annotation standard</h1><ul>'
            "<li>Goods (1): visible identifiable loads, cartons, crates and stored goods.</li>"
            "<li>Background (0): equipment, supports, empty pallets and handling containers, people and building surfaces.</li>"
            "<li>Ignored (255): unresolved identity, occlusion boundaries or unresolvable blur.</li></ul>"
            "<p>Use native geometry, preserve visible openings and distinguish loads from their supports. "
            "Saving validates source integrity, geometry, categorical IDs and version metadata.</p></html>"
        )
    entry = next(
        (
            item
            for item in entries.values()
            if relative == item["folder"] + "/index.html"
        ),
        None,
    )
    if entry is None:
        return None
    width, height = entry["record"]["image_size"]
    template = (Path(__file__).parent / "templates/native_mask_editor.html").read_text(
        encoding="utf-8"
    )
    html = _editor(template, entry["binding"], definition, width, height)
    if reviewer_name:
        html = html.replace(
            '<input id="reviewer" autocomplete="name">',
            '<input id="reviewer" autocomplete="name" value="'
            + escape(reviewer_name, quote=True)
            + '">',
        )
    return html
