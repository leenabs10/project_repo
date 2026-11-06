"""Read-only TorWIC inventory. Run from your project root; no packages needed."""

import argparse
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.paths import workspace_root


def main():
    """Inventory raw files and ZIP contents, save a report, and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=workspace_root() / "data/raw/torwic_slam"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=workspace_root() / "reports/dataset_inventory.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    print(f"Python: {sys.executable}")
    print(f"Virtual environment: {sys.prefix != sys.base_prefix}")
    print(f"Dataset: {root}")
    if not root.is_dir():
        parser.exit(
            1, "Dataset folder missing. Run from the amr-synthetic-mvp project root.\n"
        )

    extensions = Counter()
    folder_counts = Counter()
    examples = defaultdict(list)
    metadata = []
    archives = []
    errors = []
    total_bytes = 0
    total_files = 0

    def walk_error(error):
        """Keep unreadable-directory errors in the report while scanning other folders."""
        errors.append(str(error))

    for parent, dirs, names in os.walk(root, onerror=walk_error, followlinks=False):
        # Sort traversal and filenames so report ordering is reproducible.
        dirs.sort()
        for name in sorted(names):
            path = Path(parent) / name
            if path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            try:
                size = path.stat().st_size
            except OSError as error:
                errors.append(f"{relative}: {error}")
                continue
            total_files += 1
            total_bytes += size
            suffix = path.suffix.lower() or "[no extension]"
            extensions[suffix] += 1
            folder_counts[path.parent.relative_to(root).as_posix()] += 1
            if len(examples[suffix]) < 5:
                examples[suffix].append(relative)
            lower = name.lower()
            # These are candidates for later inspection, not validated metadata.
            if suffix in {".txt", ".csv", ".yaml", ".yml", ".json"} or any(
                token in lower
                for token in ("timestamp", "frame_time", "traj", "pose", "imu", "calib")
            ):
                metadata.append(relative)
            if suffix == ".zip":
                try:
                    # Read only the member listing; never extract into the raw dataset.
                    with zipfile.ZipFile(path) as archive:
                        members = [
                            member
                            for member in archive.infolist()
                            if not member.is_dir()
                        ]
                        archives.append(
                            {
                                "path": relative,
                                "file_count": len(members),
                                "uncompressed_bytes": sum(
                                    member.file_size for member in members
                                ),
                                "extension_counts": dict(
                                    Counter(
                                        Path(member.filename).suffix.lower()
                                        or "[no extension]"
                                        for member in members
                                    )
                                ),
                                "first_20_members": [
                                    member.filename for member in members[:20]
                                ],
                            }
                        )
                except (OSError, zipfile.BadZipFile) as error:
                    errors.append(f"{relative}: {error}")

    report = {
        "root": str(root),
        "python": sys.executable,
        "virtual_environment": sys.prefix != sys.base_prefix,
        "support_files": {
            name: (root / name).is_file()
            for name in ("calibrations.txt", "groundtruth_map.ply")
        },
        "top_level_entries": sorted(p.name for p in root.iterdir()),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "extension_counts": dict(sorted(extensions.items())),
        "files_per_folder": dict(sorted(folder_counts.items())),
        "example_paths_by_extension": dict(sorted(examples.items())),
        "metadata_candidates": metadata,
        "archives": archives,
        "errors": errors,
        "note": "Inventory only. File presence does not prove valid contents, date, modality, or synchronization. ZIP members were listed, not extracted or integrity-tested.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Files: {total_files:,}; size: {total_bytes / (1024 ** 3):.2f} GiB")
    for name, present in report["support_files"].items():
        print(f"{name}: {'present' if present else 'MISSING'}")
    for extension, count in sorted(extensions.items()):
        print(f"{extension}: {count:,}; example: {examples[extension][0]}")
    print(f"Report: {args.report.resolve()}")
    if errors:
        # Preserve the completed inventory but signal that some paths could not be read.
        print(f"WARNING: {len(errors)} read errors; see report.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
