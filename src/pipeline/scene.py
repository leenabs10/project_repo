"""Launch a fixed isolated scene worker and validate its canonical outputs.

Portable scene intent selects capabilities, not arbitrary executable commands.
Local executable and evidence paths are deployment configuration only.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from src.adapters.generation.capabilities import BACKENDS
from src.adapters.generation.material_assets import verify_material_assets
from src.adapters.storage.local import file_hash, reserve_output
from src.core.scene import validate_scene_job
from src.adapters.tasks.goods_scene import validate_goods_scene
from src.validation.scene_dataset import export_dataset


def run_scene(job, runtime, output, workspace, *, timeout_seconds=1800):
    """Preflight before allocating output; retain failed attempts and worker logs."""
    validate_scene_job(job)
    # The first scene application is goods segmentation; new applications need
    # their own explicit task policy, without changing the shared scene core.
    validate_goods_scene(job)
    if job["backend"] not in BACKENDS:
        raise NotImplementedError("Unknown generation backend")
    BACKENDS[job["backend"]].require(set(job["required_modalities"]))
    if job["backend"] != "blender_cycles":
        raise NotImplementedError(
            "This local scene runner has no qualified worker for that backend"
        )
    workspace = Path(workspace).resolve()
    executable = Path(runtime["blender_executable"]).resolve()
    evidence = {
        key: Path(runtime[key]).resolve()
        for key in ("training_profile", "source_manifest")
    }
    if not executable.is_file():
        raise FileNotFoundError("Configured Blender executable is unavailable")
    for key, binding in (
        ("training_profile", "profile_sha256"),
        ("source_manifest", "manifest_sha256"),
    ):
        if file_hash(evidence[key]) != job["conditioning"][binding]:
            raise ValueError("Scene conditioning evidence checksum mismatch")
    profile = json.loads(evidence["training_profile"].read_text(encoding="utf-8"))
    if (
        profile.get("split") != "train"
        or profile.get("status") != "passed"
        or profile.get("manifest_sha256") != job["conditioning"]["manifest_sha256"]
    ):
        raise ValueError("Scene conditioning profile is not training-bound")
    asset_root = (
        Path(runtime["asset_root"]).resolve() if runtime.get("asset_root") else None
    )
    asset_hashes = verify_material_assets(job, asset_root)
    worker = workspace / "src/adapters/generation/blender_scene_worker.py"
    sources = [
        worker,
        workspace / "src/synthesis/warehouse_renderer.py",
        workspace / "src/synthesis/renderer_geometry.py",
        workspace / "src/adapters/generation/material_assets.py",
    ]
    source_hashes = {str(path): file_hash(path) for path in sources}
    output = reserve_output(
        Path(output),
        (
            workspace / "data/raw",
            executable,
            *evidence.values(),
            *sources,
            *((asset_root,) if asset_root else ()),
        ),
    )
    job_path = output / "job.json"
    job_path.write_text(json.dumps(job, indent=2, allow_nan=False), encoding="utf-8")
    (output / "runtime.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )
    implementation = output / "implementation"
    implementation.mkdir()
    for path in sources:
        shutil.copyfile(path, implementation / path.name)
    command = [
        str(executable),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "2",
        "--python",
        str(worker),
        "--",
        "--job",
        str(job_path),
        "--job-sha256",
        file_hash(job_path),
        "--output",
        str(output / "rendered"),
    ]
    if asset_root:
        command.extend(["--asset-root", str(asset_root)])
    result = dict(
        status="failed",
        job_id=job["job_id"],
        executor="local_subprocess",
        backend=job["backend"],
        training_eligible=False,
    )
    try:
        with (output / "worker.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=workspace,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"Scene worker exited {completed.returncode}; inspect worker.log"
            )
        if any(
            file_hash(Path(path)) != expected
            for path, expected in source_hashes.items()
        ):
            raise ValueError("Worker implementation changed during execution")
        if verify_material_assets(job, asset_root) != asset_hashes:
            raise ValueError("Texture assets changed during rendering")
        worker_profile = json.loads(
            (output / "rendered/worker_profile.json").read_text(encoding="utf-8")
        )
        if worker_profile["job_sha256"] != file_hash(job_path):
            raise ValueError("Worker returned a different scene binding")
        verified = export_dataset(
            job, output / "rendered", output / "dataset", asset_root=asset_root
        )
        result.update(
            status=verified["status"],
            captures=verified["captures"],
            dataset_profile_sha256=file_hash(output / "dataset/profile.json"),
            executable_sha256=file_hash(executable),
            worker_source_sha256=source_hashes,
            job_sha256=file_hash(job_path),
            material_asset_sha256=asset_hashes,
        )
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        (output / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime.read_text(encoding="utf-8"))
    for key in ("blender_executable", "training_profile", "source_manifest"):
        runtime[key] = str((args.workspace / runtime[key]).resolve())
    if runtime.get("asset_root"):
        runtime["asset_root"] = str((args.workspace / runtime["asset_root"]).resolve())
    print(json.dumps(run_scene(job, runtime, args.output, args.workspace), indent=2))


if __name__ == "__main__":
    main()
