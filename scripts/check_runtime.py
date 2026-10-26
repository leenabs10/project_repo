"""Inspect the selected Python runtime without installs or training artifacts.

Run with an explicitly selected interpreter. Exit 1 for unsupported Python,
missing core packages, or a failed requested CUDA check. Optional ML packages
may be absent for a core-only runtime. Package versions are reported, not changed.
"""

import argparse
import importlib.metadata
import json
import sys


def inspect_runtime(require_cuda=False):
    """Return interpreter/package metadata and an optional CUDA kernel result.

    This checks availability, not agreement with dependency pins, model memory
    capacity, backend qualification, or dataset admission. No files are written.
    """
    report = {
        "python_executable": sys.executable,
        "python": sys.version,
        "packages": {},
    }
    for name in ("numpy", "Pillow", "jsonschema", "pytest", "torch", "torchvision"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        import torch

        available = torch.cuda.is_available()
        report["cuda"] = {"available": available, "runtime": torch.version.cuda}
        if available:
            report["cuda"]["device"] = torch.cuda.get_device_name(0)
        if require_cuda:
            if not available:
                raise RuntimeError("CUDA is required but unavailable")
            values = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
            result = (values @ values.T).cpu()
            if not torch.allclose(result, values.cpu() @ values.cpu().T):
                raise RuntimeError("CUDA matrix check failed")
            report["cuda"]["matrix_kernel_passed"] = True
    except Exception as error:
        report["cuda_error"] = str(error)
    failed = (
        sys.version_info[:2] != (3, 12)
        or any(
            report["packages"][name] is None
            for name in ("numpy", "Pillow", "jsonschema")
        )
        or (require_cuda and "cuda_error" in report)
    )
    report["status"] = "failed" if failed else "passed"
    return report


def main():
    """Print a finite JSON diagnostic and return a process-friendly status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = inspect_runtime(args.require_cuda)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
