"""Explicit canonical-manifest adapter for RGB/uint16 depth PNG datasets.

This is a supported input format, not automatic arbitrary-format recognition.
Image dimensions, depth units and invalid sentinels come from each record.
"""

from dataclasses import dataclass
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.contracts import validate_record
from src.adapters.storage.local import file_hash, source_path


@dataclass
class DecodedPair:
    """Native arrays plus source hashes; raw uint16 depth retains its declared units."""

    rgb: np.ndarray
    depth_raw: np.ndarray
    scale_to_metres: float
    invalid_raw_value: int
    provenance: dict


def read_pair(root: Path, record: dict) -> DecodedPair:
    """Decode and hash exact bytes; never read unrelated modalities or other splits."""
    validate_record(record)
    payloads = {}
    provenance = {
        "dataset_id": record["dataset_id"],
        "sample_id": record["sample_id"],
        "split": record["split"],
    }
    for modality in ("rgb", "depth"):
        reference = record["modalities"].get(modality)
        if reference is None or reference["status"] != "available":
            raise ValueError("RGB/depth preparation requires both modalities")
        path = source_path(root, reference["path"])
        payloads[modality] = path.read_bytes()
        provenance[modality + "_path"] = reference["path"]
        digest = hashlib.sha256(payloads[modality]).hexdigest()
        if reference.get("sha256", digest) != digest:
            raise ValueError("Source artifact hash differs from the manifest")
        provenance[modality + "_sha256"] = digest
        with Image.open(io.BytesIO(payloads[modality])) as image:
            image.load()
            payloads[modality] = np.array(image)
    width, height = record["image_size"]
    rgb, depth = payloads["rgb"], payloads["depth"]
    if rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
        raise ValueError("RGB shape/dtype differs from declared native geometry")
    if depth.dtype != np.uint16 or depth.shape != (height, width):
        raise ValueError("Depth must be native uint16 PNG matching the camera geometry")
    reference = record["modalities"]["depth"]
    if "invalid_raw_value" not in reference:
        raise ValueError("Depth invalid sentinel must be explicitly declared")
    return DecodedPair(
        rgb,
        depth,
        float(reference["scale_to_metres"]),
        reference["invalid_raw_value"],
        provenance,
    )


class PairedPngAdapter:
    """Read a declared canonical JSONL manifest with configurable native units."""

    def __init__(self, root: Path, manifest: Path, dataset_id: str):
        self.root = Path(root).resolve()
        self.manifest = Path(manifest).resolve()
        self.dataset_id = dataset_id
        self.manifest_sha256 = file_hash(self.manifest)
        self._records = [
            validate_record(json.loads(line))
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(record["dataset_id"] != dataset_id for record in self._records):
            raise ValueError("Manifest contains another dataset")
        self._by_sample = {record["sample_id"]: record for record in self._records}
        # Validate all declared paths before opening any sensor payload.
        for record in self._records:
            for reference in record["modalities"].values():
                if reference["status"] == "available":
                    source_path(self.root, reference["path"])

    def records(self):
        yield from copy.deepcopy(self._records)

    def read_rgb_depth(self, record):
        if self._by_sample.get(record["sample_id"]) != record:
            raise ValueError("Record differs from its source manifest")
        return read_pair(self.root, record)
