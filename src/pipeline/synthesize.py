"""Fixed local worker for controlled real-sample conditions; no simulator startup."""

import argparse
import json
from pathlib import Path

from src.adapters.datasets.registry import dataset_adapter
from src.core.contracts import validate_job
from src.pipeline.sensor_conditions import generate_conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = validate_job(json.loads(args.job.read_text(encoding="utf-8")))
    if job["kind"] != "generate_sensor_conditions":
        raise ValueError("Sensor worker requires a sensor-condition job")
    runtime = json.loads(args.runtime.read_text(encoding="utf-8"))
    adapter = dataset_adapter(runtime["dataset"], job["dataset_id"])
    report = generate_conditions(
        adapter,
        job,
        Path(runtime["quality_policy"]),
        args.output,
        runtime["evidence_files"],
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "accepted_pending_review",
                    "rejected_variants",
                    "training_eligible",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
