"""TorWIC-compatible research CLI backed by shared adapter-based cache preparation."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.adapters.datasets.torwic import TorwicAdapter
from src.core.contracts import SCHEMA_VERSION
from src.pipeline.cache import prepare_cache
from src.training.runtime import require_output


def main():
    """Preserve the legacy flags, BOX/NEAREST policy and uint16-mm research layout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    args = parser.parse_args()
    output = require_output(args.output, sources=[args.root, args.manifest])
    adapter = TorwicAdapter(args.root, args.manifest)
    job = {
        "schema_version": SCHEMA_VERSION,
        "job_id": "torwic_depth_cache",
        "kind": "prepare_rgb_depth_cache",
        "dataset_id": adapter.dataset_id,
        "source_manifest_sha256": adapter.manifest_sha256,
        "parameters": {
            "resolution": [args.width, args.height],
            "splits": ["train", "validation"],
            "depth_max_metres": 20,
            "cache_format": "legacy_mm",
        },
    }
    print(json.dumps(prepare_cache(adapter, job, output), indent=2), flush=True)


if __name__ == "__main__":
    main()
