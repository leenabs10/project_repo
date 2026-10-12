# AMR Synthetic MVP

TorWIC-SLAM dataset inspection and ingestion. Raw data stays unchanged.

Run from the project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable); print(sys.prefix != sys.base_prefix)"
python -m pip --version
.\.venv\Scripts\python.exe src\ingestion\inspect_torwic.py --root data\raw\torwic_slam
```

The inventory is saved to reports/dataset_inventory.json.

Build the confirmed trajectory manifest and run representative checks:

```powershell
.\.venv\Scripts\python.exe src\ingestion\torwic_loader.py --root data\raw\torwic_slam --output data\manifests --confirm-planned-dates
.\.venv\Scripts\python.exe src\validation\week1_checks.py --root data\raw\torwic_slam --manifest data\manifests\manifest.jsonl --output reports\week1
```

Use `--confirm-planned-dates` only for the collection dates already confirmed in
`docs/dataset_notes.txt`. To decode every trajectory payload and check both-camera
image shapes/types, semantic IDs, and binary LiDAR contents:

```powershell
.\.venv\Scripts\python.exe src\validation\full_ingestion_checks.py --root data\raw\torwic_slam --manifest data\manifests\manifest.jsonl --output reports\week1\full_payload_checks.json
```

This audit reads raw data without modifying it and performs no model evaluation.

## Week 1 status

The trajectory ingestion gate passed on October 9, 2026: all 3,088 frames
and 21,616 sensor payloads passed the full audit, with nine representative
checks and a byte-identical rebuilt manifest. See
[the portable validation summary](docs/week1_validation.json) and
[dataset notes](docs/dataset_notes.txt) for scope and limitations.

Raw datasets and local reports remain excluded from Git. This completes ingestion
for the depth-only prototype; sensor registration and IMU handling must be
validated before adding fusion. October was inspected only for data integrity.

## Dataset characterization

Development takes place on `dev`. Integration into `main` always requires the
user's explicit permission; see [development rules](AGENTS.md).

The training-only profiler covers both cameras, measured depth, semantic
predictions, LiDAR usability/range, trajectory intervals and IMU span/order.
The alignment review compares native-resolution projection with and without
the supplied distortion coefficients. It does not fit calibration parameters.

```powershell
.\.venv\Scripts\python.exe src\profiling\dataset_profile.py --root data\raw\torwic_slam --manifest data\manifests\manifest.jsonl --output reports\dataset_profile --workers 4
.\.venv\Scripts\python.exe src\validation\sensor_alignment_review.py --root data\raw\torwic_slam --manifest data\manifests\manifest.jsonl --output reports\sensor_alignment --samples 16
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See [characterization findings](docs/dataset_characterization.txt) and
[portable evidence](docs/dataset_characterization.json). Reports and overlays are
derived artifacts and remain excluded from Git. Full profiling decoded 7,952
training payloads; validation and test sensors were not opened.

`imu_timing.load_imu` rejects backward order by default. Explicit `sort=True`
returns a sorted copy with anomaly information; duplicates are rejected.
`bracket_frames` returns validity masks and never extrapolates. No vector
interpolation or implicit timestamp correction is performed.

Production camera-LiDAR registration and the exported RGB distortion convention
remain unresolved. Native overlap residuals are diagnostics, not independent
calibration accuracy. Semantic masks remain pseudo-labels.
