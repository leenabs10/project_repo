"""Validate and load registered triangle meshes without simulator dependencies."""

import hashlib
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np


def load_normalized_mesh(binding, asset_root):
    """Load a checksum-bound NPZ mesh from a contained asset-root path.

    Return float32 vertices (N, 3), int32 triangles (M, 3), and float32 UVs
    (N, 2). Vertices span centered unit bounds; scene dimensions set metric scale.
    UVs use the Blender/OpenGL bottom-left convention. Pickle loading is disabled.
    Invalid paths, checksums, archive sizes, arrays, or triangles raise ValueError;
    filesystem and archive errors propagate. No renderer objects are created.
    """
    relative = binding["path"]
    portable = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or ":" in relative
        or portable.is_absolute()
        or ".." in portable.parts
        or portable.suffix != ".npz"
    ):
        raise ValueError("Mesh paths must be contained relative NPZ references")
    root = Path(asset_root).resolve()
    path = (root / relative).resolve()
    # Bound on-disk allocation before hashing, independently of the archive's
    # declared uncompressed sizes. Large or malformed files must fail early.
    if path.is_relative_to(root) and path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Mesh file exceeds its supported byte budget")
    if (
        not path.is_relative_to(root)
        or hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]
    ):
        raise ValueError("Mesh path/checksum verification failed")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if (
            len(entries) != 3
            or {entry.filename for entry in entries}
            != {"vertices.npy", "triangles.npy", "uv.npy"}
            or sum(entry.file_size for entry in entries) > 32 * 1024 * 1024
        ):
            raise ValueError("Mesh archive exceeds its supported array budget")
    with np.load(path, allow_pickle=False) as data:
        vertices, triangles, uv = data["vertices"], data["triangles"], data["uv"]
    if (
        vertices.dtype != np.float32
        or vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not 3 <= len(vertices) <= 100000
        or not np.isfinite(vertices).all()
        or triangles.dtype != np.int32
        or triangles.ndim != 2
        or triangles.shape[1:] != (3,)
        or not 1 <= len(triangles) <= 200000
        or triangles.min() < 0
        or triangles.max() >= len(vertices)
        or uv.dtype != np.float32
        or uv.shape != (len(vertices), 2)
        or not np.isfinite(uv).all()
        or uv.min() < -1e-5
        or uv.max() > 1.00001
    ):
        raise ValueError("Invalid normalized mesh arrays, indices or UV coordinates")
    if not np.allclose(
        vertices.min(axis=0), -0.5, atol=5e-5, rtol=0
    ) or not np.allclose(vertices.max(axis=0), 0.5, atol=5e-5, rtol=0):
        raise ValueError("Mesh must retain exact centered unit bounds")
    points = vertices[triangles]
    if np.any(
        np.linalg.norm(
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
        )
        <= 1e-12
    ):
        raise ValueError("Degenerate triangles are unsupported")
    return dict(vertices=vertices, triangles=triangles, uv=uv)
