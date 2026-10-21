"""TorWIC-SLAM manifest builder and sample loader. Requires numpy and Pillow.

Raw data is read only. Paths in manifests are relative to the dataset root.
Example:
  python torwic_loader.py --root C:/amr-synthetic-mvp/data/raw/torwic_slam --output reports/ingestion
"""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Support both the existing standalone CLI and imports through dataset adapters.
if __package__ in (None, ""):
    from imu_timing import bracket_frames, load_imu
else:
    from .imu_timing import bracket_frames, load_imu

# Manifest field names map to the publisher's folders and file extensions.
MODALITIES = {
    "rgb_path": ("image_left", ".png"),
    "depth_path": ("depth_left", ".png"),
    "lidar_path": ("lidar", ".pcd"),
    "semantic_path": ("segmentation_greyscale_left", ".png"),
    "rgb_right_path": ("image_right", ".png"),
    "depth_right_path": ("depth_right", ".png"),
    "semantic_right_path": ("segmentation_greyscale_right", ".png"),
}
# Applied only when the caller explicitly confirms the collection dates.
CONFIRMED_SPLITS = {
    "Aisle_CCW_Run_1": ("train", "2022-06-15"),
    "Aisle_CW_Run_1": ("validation", "2022-06-23"),
    "Aisle_CCW": ("test", "2022-10-12"),
}
# List positions are the integer IDs stored in the grayscale semantic masks.
CLASSES = [
    "Background",
    "Driveable Ground",
    "Ceiling",
    "Ego Vehicle",
    "Wall/Fence/Pillar",
    "Miscellaneous Static Feature",
    "Rack/Shelf",
    "Goods Materials",
    "Fixed Machinery",
    "Cart/Pallet Jack",
    "Pylons",
    "Text Region",
    "Miscellaneous Non-Static Feature",
    "Person",
    "Forklift/Truck",
    "Miscellaneous Dynamic Feature",
]


def calibration(root):
    """Parse supplied key/value text; preserve transform names without inverting them."""
    result = {}
    for line in (root / "calibrations.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=" if "=" in line else ":", 1)
        try:
            # Accept Python literals (including lists), without executing file contents.
            result[key.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            raise ValueError(f"Unsupported calibration entry: {key}")
    for camera in ("Camera1", "Camera2"):
        for name in ("fx", "fy", "cx", "cy", "width", "height", "D"):
            if f"{camera}.{name}" not in result:
                raise ValueError(f"Missing calibration: {camera}.{name}")
    return result


def read_xyz(path):
    """Return filtered XYZ points and a validity mask over the original PCD rows.

    Only this dataset's binary float32 XYZ layout is supported. The mask retains
    the original point count even though the returned XYZ array drops invalid rows.
    """
    header = {}
    with path.open("rb") as stream:
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"Incomplete PCD header: {path}")
            line = raw.decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            header[key] = values
            if key == "DATA":
                break
        expected = {
            "FIELDS": ["x", "y", "z"],
            "SIZE": ["4"] * 3,
            "TYPE": ["F"] * 3,
            "COUNT": ["1"] * 3,
            "DATA": ["binary"],
        }
        if any(header.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Unsupported PCD layout: {path}")
        payload = stream.read()
    count = int(header["POINTS"][0])
    # Each point contains three 4-byte floats; '<f4' explicitly selects little endian.
    if len(payload) != count * 12:
        raise ValueError(f"PCD payload/header mismatch: {path}")
    xyz = np.frombuffer(payload, dtype="<f4").reshape(count, 3)
    valid = np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)
    return xyz[valid].copy(), valid


def lidar_to_camera(xyz, cal, side="left"):
    """Use p_camera = R.T @ (p_os - t).

    The supplied pose describes the camera in the os_sensor frame. This inverse
    interpretation is supported by beginning/middle/end training-frame depth
    comparisons for both cameras; it is not a calibration accuracy guarantee.
    """
    if side not in ("left", "right"):
        raise ValueError("Camera side must be left or right")
    prefix = "T_cam1_os" if side == "left" else "T_cam2_os"
    q = np.array([cal[f"{prefix}.q{a}"] for a in "xyzw"], dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("Invalid calibration quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    rotation_matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    translation = np.array([cal[f"{prefix}.p{a}"] for a in "xyz"])
    # Points are row vectors, so right-multiplication by R implements R.T @ p.
    return (np.asarray(xyz) - translation) @ rotation_matrix


class SequenceImu:
    """Read both IMU files once and expose safe per-frame timestamp brackets.

    ``max_gap_seconds`` is a caller-selected timing tolerance, not a calibrated
    fusion policy. ``sort=True`` explicitly permits sorting copies of raw rows.
    Column zero is seconds; columns 1..6 remain opaque publisher measurements.
    Invalid brackets have -1 indices and NaN arrays, never fabricated readings.
    No interpolation, clock correction, or coordinate conversion is performed.
    """

    def __init__(self, root, record, *, max_gap_seconds, sort=False):
        if (
            max_gap_seconds is None
            or not np.isfinite(max_gap_seconds)
            or max_gap_seconds <= 0
        ):
            raise ValueError("An explicit positive finite IMU gap limit is required")
        self.root = Path(root).resolve()
        self.sequence_id = record["sequence_id"]
        self.max_gap_seconds = float(max_gap_seconds)
        self.paths = self._record_paths(record)
        self.rows = {}
        self.reports = {}
        for side, path in self.paths.items():
            # Hash before and after loading so the provenance describes these rows.
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            rows, report = load_imu(path, sort=sort)
            if hashlib.sha256(path.read_bytes()).hexdigest() != before:
                raise ValueError("IMU source changed during loading")
            rows.setflags(write=False)
            self.rows[side] = rows
            self.reports[side] = {
                **report,
                "source_path": path.relative_to(self.root).as_posix(),
                "source_sha256": before,
                "measurement_columns": [1, 2, 3, 4, 5, 6],
                "measurement_units_and_frames": "unverified",
            }

    def _record_paths(self, record):
        """Reject escaped, cross-trajectory, or incorrectly named IMU paths."""
        if record["sequence_id"] != self.sequence_id:
            raise ValueError("IMU reader cannot be reused across sequences")
        rgb_path = (self.root / record["rgb_path"]).resolve()
        if not rgb_path.is_relative_to(self.root):
            raise ValueError("Manifest path escapes dataset root")
        sequence_directory = rgb_path.parent.parent
        if sequence_directory.name != self.sequence_id:
            raise ValueError("Sensor paths disagree with the sequence ID")
        paths = {}
        for side in ("left", "right"):
            path = (self.root / record[f"imu_{side}_path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Manifest path escapes dataset root")
            if path != sequence_directory / f"imu_{side}.txt":
                raise ValueError("IMU path disagrees with the sensor sequence or side")
            paths[side] = path
        return paths

    def bracket_record(self, record):
        """Return both sides' two-row brackets at the record's frame time.

        Each ``measurement_rows`` is float64 (2, 6), copied from original rows
        when valid, otherwise all NaN. Exact matches repeat the same observation.
        Check ``valid`` before using these arrays; a bracket is not an estimate.
        """
        if self._record_paths(record) != self.paths:
            raise ValueError("Record uses different IMU streams from this reader")
        timestamp = float(record["timestamp_seconds"])
        result = {}
        for side, rows in self.rows.items():
            left, right, valid, gaps = bracket_frames(
                rows[:, 0], [timestamp], max_gap_seconds=self.max_gap_seconds
            )
            indices = np.array([left[0], right[0]], dtype=np.int64)
            bracket_times = np.full(2, np.nan)
            measurements = np.full((2, 6), np.nan)
            if valid[0]:
                bracket_times = rows[indices, 0].copy()
                measurements = rows[indices, 1:].copy()
                reason = "exact_match" if left[0] == right[0] else "bracketed"
            else:
                reason = "outside_span" if np.isnan(gaps[0]) else "gap_exceeds_limit"
            result[side] = {
                "valid": bool(valid[0]),
                "reason": reason,
                "timestamp_seconds": timestamp,
                "bracket_indices": indices,
                "bracket_times_seconds": bracket_times,
                "measurement_rows": measurements,
                "bracket_gap_seconds": float(gaps[0]),
                "max_gap_seconds": self.max_gap_seconds,
                "ordering_report": copy.deepcopy(self.reports[side]),
                "measurements_interpolated": False,
            }
        return result


def load_sample(root, record, cal=None, *, imu_sequence=None):
    """Load left sensor data; retain a validity mask and raw depth for reproducibility.

    Pass an explicitly configured SequenceImu to also return ``imu`` brackets
    for both sides. The default remains sensor loading without IMU file reads.
    No distortion correction, projection, resampling, or IMU interpolation is
    applied. Semantic arrays retain class IDs; they are not normalized like RGB.
    """
    root = Path(root).resolve()
    imu = None
    if imu_sequence is not None:
        if imu_sequence.root != root:
            raise ValueError("IMU reader and sample must use the same dataset root")
        # Fail on inconsistent IMU provenance before decoding image payloads.
        imu = imu_sequence.bracket_record(record)
    cal = calibration(root) if cal is None else cal
    arrays = {}
    for key in ("rgb_path", "depth_path", "semantic_path"):
        path = (root / record[key]).resolve()
        # Resolve before checking containment so '..' and symlinks cannot escape root.
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes dataset root")
        with Image.open(path) as image:
            image.load()
            arrays[key] = np.array(image)
    rgb, depth, semantic = (
        arrays[k] for k in ("rgb_path", "depth_path", "semantic_path")
    )
    shape = (int(cal["Camera1.height"]), int(cal["Camera1.width"]))
    if rgb.shape != (*shape, 3) or depth.shape != shape or semantic.shape != shape:
        raise ValueError("Image shapes do not match left camera calibration")
    if rgb.dtype != np.uint8 or depth.dtype != np.uint16 or semantic.dtype != np.uint8:
        raise ValueError("Unexpected RGB, depth, or semantic dtype")
    if semantic.max() >= len(CLASSES):
        raise ValueError("Unknown semantic class ID")
    lidar_path = (root / record["lidar_path"]).resolve()
    if not lidar_path.is_relative_to(root):
        raise ValueError("Manifest path escapes dataset root")
    xyz, valid_xyz = read_xyz(lidar_path)
    # Zero is the depth sensor's invalid sentinel; preserve raw values alongside metres.
    depth_valid = depth > 0
    depth_m = depth.astype(np.float32) * 0.001
    depth_m[~depth_valid] = np.nan
    sample = {
        "sample_id": record["sample_id"],
        "sequence_id": record["sequence_id"],
        "timestamp_seconds": record["timestamp_seconds"],
        "rgb": rgb,
        "depth_raw": depth,
        "depth_m": depth_m,
        "depth_valid": depth_valid,
        "lidar_xyz": xyz,
        "lidar_valid": valid_xyz,
        "semantic_ids": semantic,
        "pose_xyz_qxyzw": np.array(record["pose_xyz_qxyzw"], dtype=np.float64),
        "calibration": cal,
        "split": record["split"],
    }
    if imu is not None:
        sample["imu"] = imu
    return sample


def build_manifest(root, confirm_dates=False):
    """Validate sequence metadata and return frame records, summaries, and calibration.

    Modality filenames must cover every frame exactly. Image and PCD contents are
    decoded separately by load_sample; this scan checks metadata and file presence.
    Splits remain unassigned unless confirm_dates is enabled.
    """
    root = Path(root).resolve()
    cal = calibration(root)
    if not (root / "groundtruth_map.ply").is_file():
        raise ValueError("Ground-truth map is missing")
    records, summaries = [], []
    for time_file in sorted(root.rglob("frame_times.txt")):
        seq = time_file.parent
        times = np.loadtxt(time_file, ndmin=1)
        poses = np.loadtxt(seq / "traj_gt.txt", ndmin=2)
        # Pose columns are timestamp, xyz, then quaternion xyzw.
        if len(times) < 2 or poses.shape != (len(times), 8):
            raise ValueError(f"Timestamp/pose row count or format mismatch: {seq}")
        if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise ValueError(f"Invalid frame timestamps: {seq}")
        if not np.isfinite(poses).all() or not np.allclose(
            poses[:, 0], times, rtol=0, atol=1e-8
        ):
            raise ValueError(f"Invalid pose values or timestamp association: {seq}")
        if not np.allclose(np.linalg.norm(poses[:, 4:8], axis=1), 1, atol=1e-5):
            raise ValueError(f"Invalid pose quaternion norms: {seq}")
        ids = [f"{i:06d}" for i in range(len(times))]
        for folder, ext in MODALITIES.values():
            actual = {p.stem for p in (seq / folder).glob(f"*{ext}")}
            if actual != set(ids):
                raise ValueError(f"Missing or extra modality indices: {seq / folder}")
        split, date = (
            CONFIRMED_SPLITS.get(seq.name, ("unassigned", None))
            if confirm_dates
            else ("unassigned", None)
        )
        imu_bounds, imu_warnings = {}, {}
        # Report ordering issues without sorting or changing the raw IMU streams.
        for side in ("left", "right"):
            rows = np.loadtxt(seq / f"imu_{side}.txt", ndmin=2)
            if rows.shape[1] != 7 or len(rows) < 2 or not np.isfinite(rows).all():
                raise ValueError(f"Invalid IMU rows: {seq}, {side}")
            imu_bounds[side] = (rows[:, 0].min(), rows[:, 0].max())
            imu_warnings[side] = {
                "backward_timestamp_intervals": int(np.sum(np.diff(rows[:, 0]) < 0)),
                "duplicate_timestamp_intervals": int(np.sum(np.diff(rows[:, 0]) == 0)),
                "duplicate_timestamps": int(len(rows) - len(np.unique(rows[:, 0]))),
            }
            if imu_warnings[side]["duplicate_timestamps"]:
                raise ValueError(
                    f"Duplicate IMU timestamps require a resolution policy: {seq}, {side}"
                )
        for i, frame in enumerate(ids):
            # Store portable paths relative to the dataset root, including nested folders.
            row = {
                "schema_version": 1,
                "sample_id": f"{seq.name}/{frame}",
                "sequence_id": seq.name,
                "frame_id": frame,
                "collection_date": date,
                "split": split,
                "timestamp_seconds": float(times[i]),
                "timestamp_basis": "sequence_relative",
                "pose_xyz_qxyzw": poses[i, 1:].tolist(),
                "pose_sensor_frame": "os_sensor",
                "calibration_id": "torwic_azure_os_sensor",
                "calibration_path": "calibrations.txt",
                "source": "TorWIC-SLAM",
                "semantic_source": "provided_model_prediction",
                "depth_scale_to_metres": 0.001,
                "imu_left_path": (seq / "imu_left.txt").relative_to(root).as_posix(),
                "imu_right_path": (seq / "imu_right.txt").relative_to(root).as_posix(),
                "imu_time_span_contains_frame": {
                    side: bool(low <= times[i] <= high)
                    for side, (low, high) in imu_bounds.items()
                },
            }
            row.update(
                {
                    key: (seq / folder / f"{frame}{ext}").relative_to(root).as_posix()
                    for key, (folder, ext) in MODALITIES.items()
                }
            )
            records.append(row)
        summaries.append(
            {
                "sequence_id": seq.name,
                "relative_directory": seq.relative_to(root).as_posix(),
                "frames": len(times),
                "split": split,
                "collection_date": date,
                "duration_seconds": float(times[-1] - times[0]),
                "frame_intervals_over_0_15_seconds": int(np.sum(np.diff(times) > 0.15)),
                "imu_order_warnings": imu_warnings,
            }
        )
    if not records or len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("Empty manifest or duplicate sample IDs")
    return records, summaries, cal


def main():
    """Build the manifest, smoke-test one training frame, and write ingestion reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/ingestion"))
    parser.add_argument(
        "--confirm-planned-dates",
        action="store_true",
        help="Use only after confirming the downloaded trajectory dates",
    )
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError("Write ingestion reports outside the immutable raw root")
    records, summaries, cal = build_manifest(args.root, args.confirm_planned_dates)
    # Smoke-test a frame in the planned training sequence; do not tune on the test sequence.
    demo_record = next(
        row for row in records if row["sample_id"] == "Aisle_CCW_Run_1/000010"
    )
    sample = load_sample(args.root, demo_record, cal)
    if not np.array_equal(np.isnan(sample["depth_m"]), ~sample["depth_valid"]):
        raise ValueError("Depth invalid-value mask does not match NaNs")
    if not np.isfinite(sample["lidar_xyz"]).all() or np.any(
        np.all(sample["lidar_xyz"] == 0, axis=1)
    ):
        raise ValueError("Invalid LiDAR points survived filtering")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.jsonl"
    # Replace only after writing every record, avoiding a partially written manifest.
    temporary = manifest.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(manifest)
    with manifest.open(encoding="utf-8") as stream:
        reloaded = [json.loads(line) for line in stream]
    if len(reloaded) != len(records):
        raise ValueError("Manifest round-trip count mismatch")
    summary = {
        "total_samples": len(records),
        "split_dates_confirmed_by_caller": args.confirm_planned_dates,
        "sequences": summaries,
        "class_ids": dict(enumerate(CLASSES)),
        "loader_smoke_test": {
            "sample_id": sample["sample_id"],
            "rgb_shape": list(sample["rgb"].shape),
            "depth_dtype": str(sample["depth_raw"].dtype),
            "positive_depth_fraction": float(sample["depth_valid"].mean()),
            "stored_lidar_points": len(sample["lidar_valid"]),
            "filtered_lidar_points": len(sample["lidar_xyz"]),
            "semantic_class_ids": np.unique(sample["semantic_ids"]).tolist(),
            "invalid_depth_mask_check": "passed",
            "lidar_filter_check": "passed",
            "manifest_round_trip": "passed",
        },
        "limitations": [
            "Only demo image and PCD payloads decoded; whole-dataset decoding remains to do.",
            "Transform inverse is implemented; use representative_ingestion_checks.py for spatial diagnostics. Calibration accuracy and distortion handling need application-specific validation.",
            "The default smoke test omits IMU; SequenceImu optionally returns explicitly sorted, gap-limited brackets without measurement interpolation.",
            "Zero XYZ points are filtered; nonzero outliers need a separate documented rule.",
            "Dates and splits remain unassigned unless explicitly confirmed.",
        ],
        "publisher_documentation": "https://github.com/Viky397/TorWICDataset",
    }
    (args.output / "ingestion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "sample_record.json").write_text(
        json.dumps(demo_record, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
