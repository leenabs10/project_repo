"""Resolve local storage separately from the portable source repository.

Only deployment paths are relocated. Frozen sample records, calibration, labels,
generation specifications and historical profiles retain their original bytes.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def workspace_root(project_root=None):
    """Return the configured absolute local workspace, or the legacy project root.

    The fallback keeps isolated fixtures and unconfigured checkouts compatible.
    A malformed existing configuration fails explicitly instead of falling back.
    """
    project = Path(project_root or PROJECT_ROOT).resolve()
    configuration = project / "configs/local.json"
    if not configuration.is_file():
        return project
    value = json.loads(configuration.read_text(encoding="utf-8-sig"))
    path = value.get("workspace_root")
    if not isinstance(path, str) or not path.strip() or not Path(path).is_absolute():
        raise ValueError("configs/local.json requires an absolute workspace_root")
    return Path(path).resolve()


def raw_data_roots(project_root=None):
    """Protect both the original and relocated raw-data locations from outputs."""
    project = Path(project_root or PROJECT_ROOT).resolve()
    return tuple(
        dict.fromkeys((project / "data/raw", workspace_root(project) / "data/raw"))
    )


def resolve_legacy_path(path, project_root=None):
    """Map recorded deployment paths without editing the original metadata.

    Relative paths are interpreted against the source checkout. Already relocated
    paths and unrelated absolute paths are preserved. Consumers still enforce
    their own input/output containment and source checksum requirements.
    """
    project = Path(project_root or PROJECT_ROOT).resolve()
    original = Path(path)
    absolute = (
        (project / original).resolve()
        if not original.is_absolute()
        else original.resolve()
    )
    if not absolute.is_relative_to(project):
        return absolute
    relative = absolute.relative_to(project)
    workspace = workspace_root(project)
    redirects = (
        (Path(".tools/scene_assets"), workspace / "assets/scene_assets"),
        (Path(".tools/materials"), workspace / "assets/materials"),
        (Path(".tools"), workspace / "archive/tools"),
        (Path("data"), workspace / "data"),
        (Path("reports"), workspace / "reports"),
        (Path("models"), workspace / "models"),
        (Path("outputs"), workspace / "outputs"),
    )
    if workspace != project:
        for old_prefix, new_prefix in redirects:
            if relative == old_prefix or relative.is_relative_to(old_prefix):
                return (new_prefix / relative.relative_to(old_prefix)).resolve()
    archived = project / "archive" / relative
    if not absolute.exists() and archived.exists():
        return archived.resolve()
    return absolute
