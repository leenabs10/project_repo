"""Contained local artifacts and fresh output reservation; no cloud assumptions."""

import hashlib
from pathlib import Path, PurePosixPath


def file_hash(path: Path) -> str:
    """Hash bytes in bounded blocks, including large caches and source files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(root: Path, relative: str) -> Path:
    """Resolve portable metadata paths; reject traversal, drives and symlink escapes."""
    portable = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or ":" in relative
        or portable.is_absolute()
        or ".." in portable.parts
    ):
        raise ValueError("Artifact paths must be contained relative POSIX paths")
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Artifact path escapes the dataset root")
    return result


def reserve_output(output: Path, protected: tuple[Path, ...]) -> Path:
    """Reserve a new derivative directory outside all input trees and their parents."""
    output = Path(output).resolve()
    for source in protected:
        source = Path(source).resolve()
        if (
            output == source
            or output.is_relative_to(source)
            or source.is_relative_to(output)
        ):
            raise ValueError("Output overlaps a protected input")
    output.mkdir(parents=True, exist_ok=False)
    return output
