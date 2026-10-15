"""Portable, nongeometric stress transforms; no dataset or simulator assumptions."""

import hashlib
import json

import numpy as np


def condition_seed(
    seed: int, dataset_id: str, sample_id: str, condition_id: str
) -> int:
    """Bind randomness to identity so partitioning/order cannot change a sample."""
    identity = json.dumps(
        [seed, dataset_id, sample_id, condition_id], separators=(",", ":")
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")


def choose_donor(condition: dict, sample_id: str, seed: int) -> str:
    """Select one declared donor, excluding self; never retry against quality scores."""
    candidates = sorted(set(condition["donor_sample_ids"]) - {sample_id})
    if not candidates:
        raise ValueError("Depth-mask transfer requires a different training donor")
    return candidates[int(np.random.default_rng(seed).integers(len(candidates)))]


def transform_pair(pair, condition: dict, donor=None) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 RGB/uint16 depth copies at native size and original depth units.

    Donors contribute only their invalid-pixel pattern. No donor measurements,
    semantic values, geometric warps or newly inferred depth values are introduced.
    """
    rgb, depth = pair.rgb.copy(), pair.depth_raw.copy()
    if condition["method"] == "rgb_gain":
        gain = condition["gain"]
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("RGB gain must be positive and finite")
        rgb = np.rint(np.clip(rgb.astype(np.float32) * gain, 0, 255)).astype(np.uint8)
    elif condition["method"] == "depth_mask_transfer":
        if donor is None or donor.depth_raw.shape != depth.shape:
            raise ValueError("Donor mask must match native depth geometry")
        donor_invalid = (donor.depth_raw == donor.invalid_raw_value) | (
            donor.depth_raw == 0
        )
        depth[donor_invalid] = pair.invalid_raw_value
    else:
        raise NotImplementedError("Sensor-condition method is not implemented")
    return rgb, depth
