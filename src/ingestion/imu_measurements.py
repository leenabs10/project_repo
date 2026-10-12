"""Evidence-gated IMU alignment in the verified measurement frame, without fusion."""

import hashlib
from pathlib import Path

import numpy as np

from imu_timing import bracket_frames, load_imu

EVIDENCE_FIELDS = (
    "column_mapping",
    "units",
    "coordinate_frame",
    "clock_basis",
    "gap_policy",
)


def source_hash(path):
    """Hash the source without retaining its payload in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_measurement_policy(root, relative_path, contract):
    """Return a verified source path and stream policy or raise ValueError.

    Column indices refer to the seven-column text export. All six measurement
    columns must occur exactly once across the two ordered XYZ vectors. Only SI
    units and an explicitly confirmed shared frame-clock basis are supported.
    Evidence references are review records, not automatic proof of their truth.
    The contract must stay disabled until those records have been verified.
    """
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported IMU measurement contract version")
    if contract.get("interpolation_enabled") is not True:
        raise ValueError("IMU measurement interpolation is disabled")
    root = Path(root).resolve()
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("IMU source must be a path relative to the dataset root")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root):
        raise ValueError("IMU source path escapes the dataset root")
    key = path.relative_to(root).as_posix()
    policy = contract.get("streams", {}).get(key)
    if not isinstance(policy, dict) or policy.get("evidence_status") != "confirmed":
        raise ValueError("This IMU stream lacks confirmed measurement metadata")
    references = policy.get("evidence_references", {})
    if not isinstance(references, dict) or any(
        not isinstance(references.get(field), str) or not references[field].strip()
        for field in EVIDENCE_FIELDS
    ):
        raise ValueError(
            "IMU columns, units, frame, clock and gap policy need evidence"
        )
    columns = []
    for field in ("angular_velocity_columns_xyz", "linear_acceleration_columns_xyz"):
        vector = policy.get(field)
        if (
            not isinstance(vector, list)
            or len(vector) != 3
            or any(
                not isinstance(index, int) or isinstance(index, bool)
                for index in vector
            )
        ):
            raise ValueError(
                "IMU vector columns must be ordered lists of three integers"
            )
        columns.extend(vector)
    if sorted(columns) != [1, 2, 3, 4, 5, 6]:
        raise ValueError(
            "Every measurement column must be assigned once; time is column 0"
        )
    if (
        policy.get("angular_velocity_units") != "rad/s"
        or policy.get("linear_acceleration_units") != "m/s^2"
    ):
        raise ValueError(
            "Only verified rad/s and m/s^2 measurement units are supported"
        )
    frame = policy.get("measurement_frame")
    if not isinstance(frame, str) or not frame.strip():
        raise ValueError("A verified measurement frame is required")
    if policy.get("timestamps_match_frame_basis") is not True:
        raise ValueError("IMU and query timestamps must have a confirmed common basis")
    gap = policy.get("max_gap_seconds")
    if (
        not isinstance(gap, (int, float))
        or isinstance(gap, bool)
        or not np.isfinite(gap)
        or gap <= 0
    ):
        raise ValueError("An evidence-backed positive finite gap limit is required")
    if policy.get("source_sha256") != source_hash(path):
        raise ValueError("IMU measurement metadata does not match this source version")
    return path, policy


def align_verified_measurements(
    root, relative_path, frame_times, contract, *, sort=False
):
    """Linearly align verified XYZ measurements without extrapolation or rotation.

    Inputs are frame timestamps in seconds on the verified common clock. Output
    vectors are float64 (N, 3), in rad/s and m/s^2, in ``measurement_frame``. Invalid
    queries remain all NaN with ``valid=False`` and -1 bracket indices. Exact
    matches reproduce original readings. Original streams are never rewritten.
    Linear resampling is an estimate, not ground truth, IMU preintegration, or a
    sensor-fusion result. Frame conversion/extrinsics remain a separate contract.
    """
    path, policy = require_measurement_policy(root, relative_path, contract)
    rows, ordering = load_imu(path, sort=sort)
    if source_hash(path) != policy["source_sha256"]:
        raise ValueError("IMU source changed while loading")
    queries = np.asarray(frame_times, dtype=np.float64)
    left, right, valid, gaps = bracket_frames(
        rows[:, 0], queries, max_gap_seconds=policy["max_gap_seconds"]
    )
    gyro = np.full((len(queries), 3), np.nan)
    acceleration = np.full((len(queries), 3), np.nan)
    indices = np.flatnonzero(valid)
    # Exact matches have zero-width brackets: alpha=0 reproduces that row.
    widths = rows[right[indices], 0] - rows[left[indices], 0]
    fractions = np.divide(
        queries[indices] - rows[left[indices], 0],
        widths,
        out=np.zeros(len(indices)),
        where=widths > 0,
    )
    for output, columns in (
        (gyro, policy["angular_velocity_columns_xyz"]),
        (acceleration, policy["linear_acceleration_columns_xyz"]),
    ):
        before = rows[left[indices]][:, columns]
        after = rows[right[indices]][:, columns]
        output[indices] = before + fractions[:, None] * (after - before)
        if not np.isfinite(output[indices]).all():
            raise ValueError("IMU interpolation produced nonfinite measurements")
    reasons = np.full(len(queries), "outside_span", dtype="<U20")
    reasons[np.isfinite(gaps) & ~valid] = "gap_exceeds_limit"
    reasons[valid] = "interpolated"
    reasons[valid & (left == right)] = "exact_match"
    return {
        "timestamp_seconds": queries.copy(),
        "angular_velocity_rad_s": gyro,
        "linear_acceleration_m_s2": acceleration,
        "valid": valid,
        "reason": reasons,
        "bracket_indices": np.column_stack([left, right]),
        "bracket_gap_seconds": gaps,
        "measurement_frame": policy["measurement_frame"],
        "source_sha256": policy["source_sha256"],
        "ordering_report": ordering,
        "method": "linear_resampling_in_verified_measurement_frame",
        "frame_conversion_applied": False,
        "sensor_fusion_applied": False,
    }
