"""Versioned portable contracts; importing them never initializes a simulator.

Only JSON-compatible metadata belongs here. Pixel arrays are decoded by dataset
adapters and processed outside this module; source-specific assumptions stay there.
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterator, Protocol

from jsonschema import Draft202012Validator

SCHEMA_VERSION = "dubaidata/1.0"


@lru_cache(maxsize=4)
def _validator(name: str) -> Draft202012Validator:
    schema = json.loads(
        (Path(__file__).parent / "schemas" / f"{name}.json").read_text()
    )
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_record(record: dict) -> dict:
    """Validate canonical metadata, rejecting nonfinite/non-JSON declarations."""
    json.dumps(record, allow_nan=False)
    _validator("sample").validate(record)
    depth = record["modalities"].get("depth")
    if depth and depth["status"] == "available":
        if (
            depth.get("scale_to_metres", 0) <= 0
            or not depth.get("units")
            or "invalid_raw_value" not in depth
        ):
            raise ValueError(
                "Depth requires declared units, scale and invalid sentinel"
            )
    camera = record.get("camera")
    if camera and any(value <= 0 for value in camera["intrinsics"][:2]):
        raise ValueError("Camera focal lengths must be positive")
    return record


def validate_job(job: dict) -> dict:
    """Validate supported portable jobs; deployment paths never belong here."""
    json.dumps(job, allow_nan=False)
    if job.get("kind") == "generate_sensor_conditions":
        _validator("sensor_job").validate(job)
        identifiers = [condition["condition_id"] for condition in job["conditions"]]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Duplicate condition IDs")
        return job
    _validator("job").validate(job)
    if len(set(job["parameters"]["splits"])) != len(job["parameters"]["splits"]):
        raise ValueError("Duplicate requested cache splits")
    return job


def validate_sensor_policy(policy: dict) -> dict:
    """Check explicit training-derived acceptance bounds, without certifying realism."""
    json.dumps(policy, allow_nan=False)
    _validator("sensor_policy").validate(policy)
    for key in ("rgb_gain_range", "mean_luma_range", "valid_depth_fraction_range"):
        low, high = policy["bounds"][key]
        if low > high:
            raise ValueError("Reversed quality-policy bounds")
    if (
        policy["bounds"]["hidden_valid_fraction_min"]
        > policy["bounds"]["hidden_valid_fraction_max"]
    ):
        raise ValueError("Reversed hidden-depth bounds")
    return policy


class DatasetAdapter(Protocol):
    """Normalize source metadata and decode RGB/depth with explicit provenance."""

    dataset_id: str
    root: Path
    manifest: Path
    manifest_sha256: str

    def records(self) -> Iterator[dict]:
        """Yield validated metadata for every split without opening sensor files."""
        ...

    def read_rgb_depth(self, record: dict) -> Any:
        """Decode one pair, preserving raw depth and the declared unit scale."""
        ...
