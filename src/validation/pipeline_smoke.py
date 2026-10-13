"""Verify integrated depth CLIs and existing checkpoints without full retraining."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.runtime import PROJECT_ROOT, require_cached_ids, require_output
from src.training.depth_baseline import Refiner, inputs
from src.training.depth_context import ContextModel, LocalModel, build_input
from src.training.depth_quality import QualityModel, corrupt
from src.inference.depth_quality import restore
from src.inference.depth_context import complete


def file_hash(path):
    """Hash binary cache/checkpoint bytes without loading them all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cli_check(relative, working_directory):
    """Test a script from an unrelated cwd; --help must perform no workflow I/O."""
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / relative), "--help"],
        cwd=working_directory,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    if result.returncode != 0 or "usage:" not in result.stdout:
        raise RuntimeError(
            f"CLI import/path failure: {relative}: {result.stderr[-1200:]}"
        )
    return {"module_path": relative, "outside_project_cwd": True, "help_exit_code": 0}


def main():
    """Check all checkpoint architectures and discard three tiny optimizer steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = require_output(
        args.output, sources=[PROJECT_ROOT / "models", PROJECT_ROOT / "data/caches"]
    )
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("The documented training environment must execute CUDA")
    device = torch.device("cuda")
    matrix = torch.arange(256, device=device, dtype=torch.float32).reshape(16, 16)
    torch.testing.assert_close((matrix @ matrix.T).cpu(), matrix.cpu() @ matrix.cpu().T)
    torch.cuda.synchronize()
    scripts = [
        "src/ingestion/prepare_depth_cache.py",
        "src/training/depth_baseline.py",
        "src/training/depth_context.py",
        "src/training/depth_quality.py",
        "src/inference/depth_context.py",
        "src/inference/depth_quality.py",
        "src/validation/depth_baseline_diagnostics.py",
        "src/validation/depth_context_assessment.py",
        "src/validation/depth_stress_audit.py",
        "src/validation/training_environment.py",
    ]
    with ThreadPoolExecutor(max_workers=2) as workers:
        cli_results = list(workers.map(lambda path: cli_check(path, output), scripts))
    registry = []
    for path in sorted((PROJECT_ROOT / "models").rglob("*.pt")):
        before = file_hash(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        family = path.relative_to(PROJECT_ROOT / "models").parts[0]
        if family == "depth_baseline":
            model = Refiner()
            channels, resolution = 6, [160, 90]
        elif family == "depth_context":
            kind = checkpoint["kind"]
            model = LocalModel() if kind == "local" else ContextModel(kind == "rgb")
            channels, resolution = 8, [160, 90]
        elif family == "depth_quality":
            model = QualityModel()
            channels, resolution = 8, [320, 180]
        else:
            raise ValueError("Unknown checkpoint family")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        width, height = resolution
        sample = torch.zeros((1, channels, height, width), device=device)
        sample[:, 5] = 0.5
        with torch.no_grad():
            prediction = model(sample)
        if (
            prediction.shape != (1, 1, height, width)
            or not torch.isfinite(prediction).all()
        ):
            raise ValueError("Checkpoint shape or finite inference failed")
        assert file_hash(path) == before
        registry.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": before,
                "family": family,
                "resolution": resolution,
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "seed": checkpoint.get("seed"),
                "epoch": checkpoint.get("epoch"),
                "denoising_allowed": bool(checkpoint.get("denoising_allowed", False)),
                "strict_state_loading": "passed",
                "cuda_forward": "passed",
                "production_release": False,
            }
        )
        del model, checkpoint, prediction, sample
    samples = {}
    for name, resolution in (("depth_160", [160, 90]), ("depth_320", [320, 180])):
        directory = PROJECT_ROOT / "data/caches" / name
        for split in ("train", "validation"):
            with np.load(directory / f"{split}.npz", allow_pickle=False) as cached:
                ids = require_cached_ids(cached["sample_ids"].tolist(), split)
                if split == "train":
                    rgb = cached["rgb"][0].copy()
                    depth = cached["depth_mm"][0].astype(np.float32) * 0.001
                    donors = (
                        cached["depth_mm"][:4].copy() if name == "depth_320" else None
                    )
                    samples[name] = (rgb, depth, donors)
        if (rgb.shape[1], rgb.shape[0]) != tuple(resolution):
            raise ValueError("Cache resolution differs from its model contract")
    optimizer_checks = []
    for family in ("depth_baseline", "depth_context", "depth_quality"):
        if family == "depth_quality":
            rgb, depth, donors = samples["depth_320"]
            x, _, mask, _, _ = corrupt(rgb, depth, donors, 0, 4242, "blocks")
            model, normalization = QualityModel().to(device), 5
        else:
            rgb, depth, _ = samples["depth_160"]
            valid = (depth > 0) & (depth <= 20)
            observed = valid & (np.random.default_rng(4242).random(depth.shape) > 0.6)
            mask = valid & ~observed
            if family == "depth_baseline":
                x = inputs(rgb, depth, observed)
                model, normalization = Refiner().to(device), 20
            else:
                x = build_input(rgb, depth, observed)[0]
                model, normalization = ContextModel(False).to(device), 5
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        prediction = model(torch.from_numpy(x)[None].to(device))[0, 0]
        target = torch.from_numpy(depth / normalization).to(device)
        hidden = torch.from_numpy(mask).to(device)
        loss = (prediction - target).abs()[hidden].mean()
        if not torch.isfinite(loss):
            raise ValueError("Training smoke loss is nonfinite")
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ValueError("Training smoke gradient is nonfinite")
        optimizer.step()
        optimizer_checks.append(
            {
                "family": family,
                "optimizer_step": "passed",
                "normalized_loss": float(loss.detach()),
                "weights_saved": False,
            }
        )
    candidate = torch.load(
        PROJECT_ROOT / "models/depth_quality/research_candidate.pt",
        map_location=device,
        weights_only=True,
    )
    model = QualityModel().to(device).eval()
    model.load_state_dict(candidate["state_dict"], strict=True)
    _, depth, _ = samples["depth_320"]
    completed, observed = restore(model, depth)
    np.testing.assert_allclose(completed[observed], depth[observed], atol=1e-5)
    for entry in registry:
        assert file_hash(PROJECT_ROOT / entry["path"]) == entry["sha256"]
    report = {
        "status": "passed",
        "python_executable": sys.executable,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_matrix_kernel": "passed",
        "cli_checks": cli_results,
        "checkpoint_registry": registry,
        "optimizer_smoke_checks": optimizer_checks,
        "quality_inference_observed_preservation": "passed",
        "existing_checkpoints_unchanged": True,
        "full_retraining_performed": False,
        "test_sensor_files_opened": 0,
        "code_sha256": file_hash(Path(__file__)),
    }
    (output / "pipeline_smoke.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "cli_checks": len(cli_results),
                "checkpoints_checked": len(registry),
                "optimizer_smoke_steps": len(optimizer_checks),
                "full_retraining": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
