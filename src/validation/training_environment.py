"""Check imports, CPU processing and an actual CUDA kernel."""

import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.runtime import PROJECT_ROOT, require_output, require_cached_ids
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = require_output(args.output)
    cache = args.output.resolve().parent / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    # Load PyTorch DLLs before Open3D on Windows.
    import torch, torchvision
    import numpy as np
    import PIL, scipy, pandas, matplotlib, cv2, open3d as o3d, yaml, tqdm, sklearn, pytest

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    )
    assert len(cloud.points) == 2
    assert cv2.resize(np.zeros((8, 8, 3), dtype=np.uint8), (4, 4)).shape == (4, 4, 3)
    gpu = {"available": torch.cuda.is_available()}
    if gpu["available"]:
        gpu.update(
            name=torch.cuda.get_device_name(0),
            cuda_runtime=torch.version.cuda,
            capability=list(torch.cuda.get_device_capability(0)),
        )
        x = torch.arange(256, device="cuda", dtype=torch.float32).reshape(16, 16)
        assert torch.allclose((x @ x.T).cpu(), x.cpu() @ x.cpu().T)
        torch.cuda.synchronize()
        gpu["matrix_kernel_passed"] = True
    else:
        raise RuntimeError("CUDA is unavailable; GPU training verification failed.")
    names = [
        "numpy",
        "Pillow",
        "scipy",
        "pandas",
        "matplotlib",
        "opencv-python",
        "open3d",
        "PyYAML",
        "tqdm",
        "scikit-learn",
        "pytest",
        "torch",
        "torchvision",
    ]
    report = {
        "python": sys.executable,
        "python_version": sys.version,
        "packages": {n: md.version(n) for n in names},
        "cuda": gpu,
        "core_smoke_test": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
