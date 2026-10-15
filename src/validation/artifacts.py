"""Read-only verification of source-bound artifact snapshots."""

import json
import importlib.metadata
import sys
from pathlib import Path

from src.adapters.storage.local import file_hash, source_path


def verify_snapshot(root: Path, expected_profile_sha256: str | None = None) -> dict:
    """Verify profile identity and every listed artifact using contained paths."""
    root = Path(root).resolve()
    path = root / "profile.json"
    if expected_profile_sha256 and file_hash(path) != expected_profile_sha256:
        raise ValueError("Snapshot profile checksum mismatch")
    profile = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(profile.get("artifact_sha256"), dict)
        or not profile["artifact_sha256"]
    ):
        raise ValueError("Snapshot has no artifact checksums")
    for relative, digest in profile["artifact_sha256"].items():
        if file_hash(source_path(root, relative)) != digest:
            raise ValueError("Snapshot artifact checksum mismatch: " + relative)
    return profile


def write_json(path: Path, value) -> None:
    """Write finite JSON metadata; runtime artifacts never overwrite raw sources."""
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def artifact_hashes(root: Path) -> dict:
    """List immutable payload hashes; the containing profile is hashed by its caller."""
    return {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "profile.json"
    }


def write_implementation_snapshot(output: Path) -> None:
    """Capture source/template versions separately from historical evidence."""
    source_root = Path(__file__).parents[1]
    paths = set(source_root.rglob("*.py"))
    paths.update((source_root / "core/schemas").glob("*.json"))
    paths.update((source_root / "validation/templates").glob("*.html"))
    write_json(
        output / "implementation.json",
        {
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "Pillow", "jsonschema")
            },
            "files": {
                path.relative_to(source_root.parent).as_posix(): file_hash(path)
                for path in sorted(paths)
            },
        },
    )
