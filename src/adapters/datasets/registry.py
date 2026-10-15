"""Explicit dataset adapter selection from trusted deployment configuration."""

from pathlib import Path


def dataset_adapter(configuration: dict, dataset_id: str):
    """Construct supported adapters lazily; never import a module named by a job."""
    arguments = Path(configuration["root"]), Path(configuration["manifest"])
    if configuration["adapter"] == "torwic":
        from src.adapters.datasets.torwic import TorwicAdapter

        return TorwicAdapter(*arguments)
    if configuration["adapter"] == "paired_png_manifest":
        from src.adapters.datasets.paired_images import PairedPngAdapter

        return PairedPngAdapter(*arguments, dataset_id)
    raise ValueError("Dataset adapter is not implemented")
