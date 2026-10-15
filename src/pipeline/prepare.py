"""Local worker: resolve dataset adapters from deployment configuration, then prepare."""

import argparse
import json
from pathlib import Path

from src.core.contracts import validate_job
from src.adapters.datasets.registry import dataset_adapter
from src.pipeline.cache import prepare_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = validate_job(json.loads(args.job.read_text(encoding="utf-8")))
    runtime = json.loads(args.runtime.read_text(encoding="utf-8"))
    configuration = runtime["dataset"]
    if job["kind"] != "prepare_rgb_depth_cache":
        raise ValueError("Preparation worker requires a cache job")
    adapter = dataset_adapter(configuration, job["dataset_id"])
    report = prepare_cache(adapter, job, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "dataset_id": report["dataset_id"],
                "splits": report["splits"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
