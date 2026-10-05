"""Read-only TorWIC inventory. Run from your project root; no packages needed."""
import argparse
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/torwic_slam"))
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_inventory.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    print(f"Python: {sys.executable}")
    print(f"Virtual environment: {sys.prefix != sys.base_prefix}")
    print(f"Dataset: {root}")
    if not root.is_dir():
        parser.exit(1, "Dataset folder missing. Run from the amr-synthetic-mvp project root.\n")

    extensions = Counter()
    folder_counts = Counter()
    examples = defaultdict(list)
    metadata = []
    archives = []
    errors = []
    total_bytes = 0
    total_files = 0

    def walk_error(error):
        errors.append(str(error))

    for parent, dirs, names in os.walk(root, onerror=walk_error, followlinks=False):
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
            if suffix in {".txt", ".csv", ".yaml", ".yml", ".json"} or any(
                token in lower for token in ("timestamp", "frame_time", "traj", "pose", "imu", "calib")
            ):
                metadata.append(relative)
            if suffix == ".zip":
                try:
                    with zipfile.ZipFile(path) as archive:
                        members = [m for m in archive.infolist() if not m.is_dir()]
                        archives.append({
                            "path": relative,
                            "file_count": len(members),
                            "uncompressed_bytes": sum(m.file_size for m in members),
                            "extension_counts": dict(Counter(Path(m.filename).suffix.lower() or "[no extension]" for m in members)),
                            "first_20_members": [m.filename for m in members[:20]],
                        })
                except (OSError, zipfile.BadZipFile) as error:
                    errors.append(f"{relative}: {error}")

    report = {
        "root": str(root),
        "python": sys.executable,
        "virtual_environment": sys.prefix != sys.base_prefix,
        "support_files": {name: (root / name).is_file() for name in ("calibrations.txt", "groundtruth_map.ply")},
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
        print(f"WARNING: {len(errors)} read errors; see report.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
