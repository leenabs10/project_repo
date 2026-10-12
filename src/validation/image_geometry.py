"""Require explicit image convention and matching calibration before sensor fusion."""

import hashlib
from pathlib import Path


def require_projection_policy(contract, calibration_path):
    """Return distortion application only for an evidence-confirmed convention.

    This guard is for fusion consumers, not diagnostic comparison of hypotheses.
    Rectified images require the corresponding adjusted pinhole intrinsics.
    """
    actual_hash = hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest()
    if contract.get("fusion_enabled") is not True:
        raise ValueError("Camera-LiDAR fusion is disabled by the geometry contract")
    if contract.get("calibration_sha256") != actual_hash:
        raise ValueError("Image convention is not tied to this calibration version")
    if contract.get("evidence_status") != "confirmed" or not contract.get(
        "evidence_reference"
    ):
        raise ValueError(
            "Image convention requires confirmed export or camera metadata evidence"
        )
    convention = contract.get("source_image_convention")
    if convention == "raw_distorted":
        return True
    if (
        convention == "rectified_pinhole"
        and contract.get("rectified_intrinsics_confirmed") is True
    ):
        return False
    raise ValueError("Unknown image convention or unverified rectified intrinsics")
