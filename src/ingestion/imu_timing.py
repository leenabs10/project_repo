"""Read-only IMU ordering and frame-bracketing helpers; no measurement interpolation."""

from pathlib import Path
import numpy as np


def load_imu(path, *, sort=False):
    """Return seven-column IMU rows and an explicit ordering report.

    Column zero is time in seconds. The remaining publisher columns are retained
    unchanged; this module does not assign units or rotate measurement vectors.
    Sorting is opt-in, stable, and performed on a copy. Duplicate timestamps are
    rejected because their measurement-selection policy has not been specified.
    """
    rows = np.loadtxt(Path(path), ndmin=2)
    if rows.shape[1] != 7 or len(rows) < 2 or not np.isfinite(rows).all():
        raise ValueError("Expected at least two finite, seven-column IMU rows")
    times = rows[:, 0]
    backward = int(np.sum(np.diff(times) < 0))
    duplicates = len(times) - len(np.unique(times))
    if duplicates:
        raise ValueError(
            "Duplicate IMU timestamps require an explicit resolution policy"
        )
    if backward and not sort:
        raise ValueError("Backward IMU timestamps: request sort=True explicitly")
    ordered = rows[np.argsort(times, kind="stable")].copy() if sort else rows.copy()
    report = {
        "rows": len(rows),
        "backward_raw_intervals": backward,
        "duplicate_timestamps": int(duplicates),
        "sorting_requested": sort,
        "rows_reordered": bool(backward and sort),
        "time_span_seconds": [float(ordered[0, 0]), float(ordered[-1, 0])],
        "interval_seconds_p50_p99_max": np.quantile(
            np.diff(ordered[:, 0]), [0.5, 0.99, 1]
        ).tolist(),
        "measurements_interpolated": False,
    }
    return ordered, report


def bracket_frames(imu_times, frame_times, *, max_gap_seconds=None):
    """Return bracketing indices, validity and gaps without extrapolating.

    Invalid queries receive -1 indices and NaN gaps. Exact timestamp matches use
    the same row on both sides, including endpoints. A gap limit is optional and
    must be supplied explicitly; rejected gaps remain reported for diagnosis.
    """
    times = np.asarray(imu_times, dtype=float)
    queries = np.asarray(frame_times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("IMU times must be finite and strictly increasing")
    if queries.ndim != 1 or not np.isfinite(queries).all():
        raise ValueError("Frame times must be a finite one-dimensional array")
    if max_gap_seconds is not None and (
        not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0
    ):
        raise ValueError("Gap limit must be a positive finite number")
    right = np.searchsorted(times, queries, side="left")
    clipped = np.minimum(right, len(times) - 1)
    exact = (right < len(times)) & (times[clipped] == queries)
    left = np.where(exact, right, right - 1)
    valid = (queries >= times[0]) & (queries <= times[-1])
    gaps = np.full(len(queries), np.nan)
    gaps[valid] = times[right[valid]] - times[left[valid]]
    if max_gap_seconds is not None:
        valid &= gaps <= max_gap_seconds
    return np.where(valid, left, -1), np.where(valid, right, -1), valid, gaps
