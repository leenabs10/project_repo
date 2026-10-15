"""Run a supported portable job with separate local deployment configuration."""

import argparse
import copy
import json
from pathlib import Path
import sys

from src.adapters.execution.local import LocalExecutor


def main():
    """Resolve deployment paths outside the job and dispatch the fixed local worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--worker-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    runtime = copy.deepcopy(json.loads(args.runtime.read_text(encoding="utf-8")))
    for key in ("root", "manifest"):
        runtime["dataset"][key] = str((workspace / runtime["dataset"][key]).resolve())
    if "quality_policy" in runtime:
        runtime["quality_policy"] = str(
            (workspace / runtime["quality_policy"]).resolve()
        )
    if "evidence_files" in runtime:
        runtime["evidence_files"] = {
            key: str((workspace / value).resolve())
            for key, value in runtime["evidence_files"].items()
        }
    result = LocalExecutor(args.worker_python, workspace).run(job, runtime, args.output)
    summary = {key: value for key, value in result.items() if key != "artifact_sha256"}
    summary["verified_artifacts"] = len(result["artifact_sha256"])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
