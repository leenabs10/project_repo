"""TorWIC conventions isolated behind the shared metadata and RGB/depth interface."""

import json
import copy
from pathlib import Path

from src.adapters.datasets.paired_images import read_pair
from src.adapters.storage.local import file_hash, source_path
from src.core.contracts import SCHEMA_VERSION, validate_record
from src.ingestion.torwic_loader import calibration, load_sample
from src.profiling.task_dataset_review import evaluation_records


class TorwicAdapter:
    """Reuse existing split/date/path rules and native calibration without rewriting raw data."""

    dataset_id = "torwic_slam"

    def __init__(self, root: Path, manifest: Path):
        self.root = Path(root).resolve()
        self.manifest = Path(manifest).resolve()
        self.manifest_sha256 = file_hash(self.manifest)
        self._source = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        evaluation_records(self._source)
        self._source.sort(key=lambda row: (row["sequence_id"], int(row["frame_id"])))
        self.calibration = calibration(self.root)
        self.calibration_sha256 = file_hash(self.root / "calibrations.txt")
        self._records = []
        for row in self._source:
            modalities = {
                "rgb": {
                    "status": "available",
                    "path": row["rgb_path"],
                    "units": "uint8_srgb",
                    "frame": "azure_left_camera",
                },
                "depth": {
                    "status": "available",
                    "path": row["depth_path"],
                    "units": "millimetres",
                    "scale_to_metres": row["depth_scale_to_metres"],
                    "invalid_raw_value": 0,
                    "frame": "azure_left_camera",
                },
                "lidar": {
                    "status": "available",
                    "path": row["lidar_path"],
                    "units": "metres",
                    "frame": "os_sensor",
                    "payload_type": "binary_float32_xyz",
                },
                "semantic": {
                    "status": "available",
                    "path": row["semantic_path"],
                    "label_origin": row["semantic_source"],
                },
            }
            if row["depth_scale_to_metres"] != 0.001:
                raise ValueError("Unexpected TorWIC depth unit convention")
            for reference in modalities.values():
                source_path(self.root, reference["path"])
            record = {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": self.dataset_id,
                "sample_id": row["sample_id"],
                "group_id": row["sequence_id"],
                "split": row["split"],
                "timestamp_seconds": row["timestamp_seconds"],
                "timestamp_basis": row["timestamp_basis"],
                "image_size": [
                    int(self.calibration["Camera1.width"]),
                    int(self.calibration["Camera1.height"]),
                ],
                "camera": {
                    "sensor_id": "azure_left_camera",
                    "image_convention": "real_export_unverified",
                    "intrinsics": [
                        float(self.calibration["Camera1." + key])
                        for key in ("fx", "fy", "cx", "cy")
                    ],
                },
                "modalities": modalities,
                "source_metadata": {
                    "publisher_record": row,
                    "calibration_sha256": self.calibration_sha256,
                    "registration_status": "real_image_convention_unverified",
                },
            }
            self._records.append(validate_record(record))
        self._by_sample = {record["sample_id"]: record for record in self._records}

    def records(self):
        """Expose metadata for all groups without opening RGB/depth/LiDAR payloads."""
        yield from copy.deepcopy(self._records)

    def read_rgb_depth(self, record):
        if self._by_sample.get(record["sample_id"]) != record:
            raise ValueError("Record differs from its source manifest")
        return read_pair(self.root, record)

    def read_multimodal(self, record):
        """Delegate full native sensor decoding to the existing verified loader."""
        if self._by_sample.get(record["sample_id"]) != record:
            raise ValueError("Record differs from its source manifest")
        return load_sample(
            self.root, record["source_metadata"]["publisher_record"], self.calibration
        )
