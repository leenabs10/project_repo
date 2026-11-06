"""Render deployment-selected report navigation without changing evidence.

All report paths are POSIX paths relative to the workspace's reports directory.
Only generated reports/mvp/index.html is written. Labels and descriptions are
HTML escaped; missing files, unsupported fields and escaping paths fail before I/O.
"""

import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote

from src.adapters.storage.local import source_path
from src.core.paths import raw_data_roots


def render_navigation(configuration, template):
    """Return HTML for existing local reports; never infer a selected experiment.

    ``configuration`` is a decoded deployment dictionary. ``template`` is text
    containing exactly one summary and navigation placeholder. Raises ValueError
    for invalid configuration, and FileNotFoundError for unavailable selections.
    No files are written by this function.
    """
    workspace = Path(configuration["workspace_root"])
    if not workspace.is_absolute():
        raise ValueError("workspace_root must be absolute")
    root = workspace.resolve() / "reports"
    view = configuration.get("report_workspace", {})
    if not isinstance(view, dict) or set(view) - {
        "port",
        "scenes",
        "predictions",
        "summary",
        "links",
    }:
        raise ValueError("Unsupported report_workspace configuration")
    port = view.get("port", 8810)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Report port must be an integer from 1 to 65535")
    summary = view.get("summary", "No report selection is configured.")
    if not isinstance(summary, str):
        raise ValueError("Report summary must be text")
    links = view.get("links", [])
    if not isinstance(links, list):
        raise ValueError("Report links must be a list")

    def report_path(value):
        if not isinstance(value, str):
            raise ValueError("Report paths must be relative POSIX strings")
        path = source_path(root, value)
        if not path.is_file():
            raise FileNotFoundError("Configured report is unavailable: " + value)
        return "../" + quote(value, safe="/")

    for name in ("scenes", "predictions"):
        if name in view:
            report_path(view[name])
    items = []
    for link in links:
        if not isinstance(link, dict) or set(link) != {"label", "path", "description"}:
            raise ValueError("Each report link needs label, path and description")
        if not all(isinstance(link[key], str) for key in ("label", "description")):
            raise ValueError("Report labels and descriptions must be text")
        href = report_path(link["path"])
        items.append(
            "    <li>\n"
            f'      <a href="{html.escape(href, quote=True)}">'
            f'{html.escape(link["label"])}</a>\n'
            f'      <p>{html.escape(link["description"])}</p>\n'
            "    </li>"
        )
    if not items:
        items.append("    <li>Configure report_workspace in configs/local.json.</li>")
    for placeholder in ("{{summary}}", "{{navigation}}"):
        if template.count(placeholder) != 1:
            raise ValueError("Report template requires one " + placeholder)
    return template.replace("{{navigation}}", "\n".join(items)).replace(
        "{{summary}}", html.escape(summary)
    )


def main():
    """Refresh only generated navigation after validating all local selections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()
    configuration = json.loads(args.configuration.read_text(encoding="utf-8-sig"))
    rendered = render_navigation(
        configuration, args.template.read_text(encoding="utf-8-sig")
    )
    destination = Path(configuration["workspace_root"]) / "reports/mvp/index.html"
    if any(
        destination.resolve().is_relative_to(raw.resolve()) for raw in raw_data_roots()
    ):
        raise ValueError("Report navigation cannot be written inside raw data")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
