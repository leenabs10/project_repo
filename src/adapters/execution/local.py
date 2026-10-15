"""Launch a fixed worker without shell interpolation or engine imports."""

import json
from pathlib import Path
import subprocess

from src.adapters.storage.local import file_hash, reserve_output
from src.core.contracts import validate_job


class LocalExecutor:
    """Keep machine executables/roots out of portable jobs; record fresh attempts."""

    def __init__(self, python_executable: Path, project_root: Path):
        self.python_executable = Path(python_executable).resolve()
        self.project_root = Path(project_root).resolve()

    def run(
        self, job: dict, runtime: dict, output: Path, *, timeout_seconds=1800
    ) -> dict:
        """Run a supported fixed worker and verify artifact hashes before success.

        The worker is fixed application code. Customer requests cannot choose a
        shell command, import module or executable. Timeout/failure remains an
        explicit failed attempt, never a successful or training-ready dataset.
        """
        validate_job(job)
        worker = {
            "prepare_rgb_depth_cache": "src.pipeline.prepare",
            "generate_sensor_conditions": "src.pipeline.synthesize",
        }[job["kind"]]
        dataset = runtime["dataset"]
        if file_hash(Path(dataset["manifest"])) != job["source_manifest_sha256"]:
            raise ValueError("Runtime manifest does not match the portable job")
        evidence_paths = tuple(
            Path(path) for path in runtime.get("evidence_files", {}).values()
        )
        if runtime.get("quality_policy"):
            evidence_paths += (Path(runtime["quality_policy"]),)
        output = reserve_output(
            output,
            (
                Path(dataset["root"]),
                Path(dataset["manifest"]),
                self.project_root / "data/raw",
                *evidence_paths,
            ),
        )
        job_path, runtime_path = output / "job.json", output / "runtime.json"
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
        command = [
            str(self.python_executable),
            "-m",
            worker,
            "--job",
            str(job_path),
            "--runtime",
            str(runtime_path),
            "--output",
            str(output / "dataset"),
        ]
        try:
            with (output / "worker.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=self.project_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                    shell=False,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"Worker failed with exit code {completed.returncode}; inspect worker.log"
                )
            profile = json.loads(
                (output / "dataset/profile.json").read_text(encoding="utf-8")
            )
            if (
                profile["status"] != "passed"
                or profile["manifest_sha256"] != job["source_manifest_sha256"]
            ):
                raise ValueError("Worker returned an unsuccessful or mismatched result")
            if job["kind"] == "prepare_rgb_depth_cache":
                artifacts = {
                    f"{split}.npz": digest
                    for split, digest in profile["cache_sha256"].items()
                }
            else:
                artifacts = profile["artifact_sha256"]
                if profile.get("training_eligible") is not False:
                    raise ValueError("Sensor batch must remain quarantined for review")
            from src.adapters.storage.local import source_path

            for relative, digest in artifacts.items():
                if file_hash(source_path(output / "dataset", relative)) != digest:
                    raise ValueError("Worker artifact checksum mismatch")
            result = {
                "status": "passed",
                "job_id": job["job_id"],
                "executor": "local_subprocess",
                "worker_profile": "dataset/profile.json",
                "artifact_sha256": artifacts,
                "worker_profile_sha256": file_hash(output / "dataset/profile.json"),
            }
            if job["kind"] == "prepare_rgb_depth_cache":
                result["cache_sha256"] = profile["cache_sha256"]
            else:
                result.update(
                    training_eligible=False,
                    human_review="pending",
                    accepted_pending_review=profile["accepted_pending_review"],
                    rejected_variants=profile["rejected_variants"],
                )
        except Exception as error:
            result = {
                "status": "failed",
                "job_id": job["job_id"],
                "executor": "local_subprocess",
                "error": str(error),
            }
            (output / "result.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            raise
        (output / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result
