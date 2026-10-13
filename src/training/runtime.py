"""Shared project paths and input/output protection for depth research CLIs."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def require_output(path, *, sources=(), allow_existing=False):
    """Resolve a derivative destination and reject raw/input/checkpoint overwrites.

    Existing files/directories require explicit reuse permission. Training,
    prediction and audit outputs are research derivatives outside raw data.
    """
    destination = Path(path).resolve()
    protected = [
        PROJECT_ROOT / "data/raw",
        *[Path(source).resolve() for source in sources],
    ]
    if any(
        destination == source or destination.is_relative_to(source)
        for source in protected
    ):
        raise ValueError("Output must be outside raw data and input/checkpoint sources")
    if destination.exists() and not allow_existing:
        raise ValueError(
            "Output already exists; preserve it and choose a fresh run path"
        )
    return destination


def require_cached_ids(sample_ids, split):
    """Reject duplicate/cross-trajectory IDs in a train or development cache."""
    sequence = {"train": "Aisle_CCW_Run_1", "validation": "Aisle_CW_Run_1"}.get(split)
    if sequence is None:
        raise ValueError("Only train and development-validation caches are supported")
    ids = list(sample_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Empty cache or duplicate sample IDs")
    if any(
        not isinstance(item, str) or not item.startswith(sequence + "/") for item in ids
    ):
        raise ValueError("Cache includes an unexpected trajectory or held-out sample")
    return ids
