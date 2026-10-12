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

## Data-quality gates and augmentation pilot

The development pipeline includes training-derived augmentation proposals,
native-resolution paired export checks, image-geometry preflight, and an audit of
extra human annotations. See [review findings](docs/data_quality_review.txt).

```powershell
.\.venv\Scripts\python.exe src\synthesis\augmentation_policy.py --profile reports\dataset_profile\summary.json --root data\raw\torwic_slam --manifest data\manifests\manifest.jsonl --output configs\augmentation_policy.json
.\.venv\Scripts\python.exe src\synthesis\paired_augmentation_review.py --root data\raw\torwic_slam --manifest data\manifests\manifest.jsonl --policy configs\augmentation_policy.json --output reports\augmentation_pilot_next --samples 6
.\.venv\Scripts\python.exe src\validation\annotation_provenance.py --root data\raw\torwic_slam --annotations data\raw\torwic_slam\AnnotatedSemanticSet_Finetuning\Labelled_Test_Data --output reports\annotation_provenance_review
```

Choose a fresh pilot output directory; existing runs are never overwritten.
The extra indexed annotations use instance IDs and require their per-file
`labelMapping` to produce semantic IDs. Their unknown split provenance remains
quarantined. Trajectory masks retain their existing semantic-ID interpretation.

`configs/image_geometry.json` intentionally disables fusion until the Azure
export convention is confirmed. RGB gain and missing-depth mask transfer are
controlled stress augmentations, not certified physical sensor simulation.
Publish completed development batches only when the user authorizes the target branch.


## IMU loader integration

See [the timing review](docs/imu_ingestion_review.txt) and
[portable evidence](docs/imu_ingestion_review.json). `SequenceImu` reads both
streams once per trajectory; pass it as `load_sample(..., imu_sequence=reader)`.
Its constructor requires `max_gap_seconds`; `sort=True` explicitly sorts a copy.
Check each side's `valid` and `reason` before using `measurement_rows` (2 x 6).
Column meanings/units remain unverified, and no vector interpolation is applied.

Run a read-only timing integrity review into a fresh report path:

```powershell
.\.venv\Scripts\python.exe src\validation\imu_ingestion_review.py --root data\raw\torwic_slam --manifest data\manifests\manifest.jsonl --output reports\imu_review_next\timing_review.json --sort-imu --max-gap-seconds 0.02
```

The 20 ms value is an explicit inspection tolerance, not a certified fusion
setting. The report separates successful loader-contract checks from incomplete
coverage. Invalid brackets return NaN measurements and -1 indices. Keep raw
streams and split membership unchanged; do not fill unavailable IMU readings.


## Verified IMU measurement alignment

The [metadata review](docs/sensor_metadata_review.txt) records the source
investigation and remaining requirements. `imu_measurements.align_verified_measurements`
supports linear resampling only with reviewed per-stream column, unit, frame,
clock and gap evidence bound to the source hash. It returns masked SI vectors in
the recorded sensor frame; no coordinate conversion or fusion is applied.

`configs/imu_measurements.json` keeps real TorWIC interpolation disabled. Preserve
unknown fields as null; do not populate them from the separate Mapping exporter
or assume a 20 ms inspection tolerance is certified. The existing bracket-only
SequenceImu loader remains available without assigning vector meanings.

```powershell
.\.venv\Scripts\python.exe src\validation\sensor_metadata_review.py --root data\raw\torwic_slam --imu-contract configs\imu_measurements.json --geometry-contract configs\image_geometry.json --annotations data\raw\torwic_slam\AnnotatedSemanticSet_Finetuning\Labelled_Test_Data --output reports\sensor_metadata_review_next\review.json
```

Use a fresh output path. `protections_verified_metadata_pending` confirms the
current disabled gates, not readiness for interpolation, fusion or annotation
training. Once evidence is obtained, review and test it before enabling a gate.


## Warehouse goods task and renderer pilot

See [MVP scope](docs/mvp_scope.txt) and [current readiness](docs/dataset_readiness.txt).
The chosen starting task is left-camera goods segmentation. Binary target 1
means publisher semantic class 7, not driveable-ground class 1. The real targets
are pseudo-labels; [visual findings](docs/label_quality_findings.json) identify
possible errors requiring reference review before training claims.

```powershell
.\.venv\Scripts\python.exe src\profiling\task_dataset_review.py --root data\raw\torwic_slam --manifest data\manifests\manifest.jsonl --profile reports\dataset_profile\summary.json --output reports\task_review_next
$renderer = (Get-Content .tools\renderer_runtime.json -Raw | ConvertFrom-Json).blender_executable
& $renderer --background --factory-startup -t 8 --python-exit-code 1 --python src\synthesis\warehouse_renderer.py -- --config configs\warehouse_renderer.json --root data\raw\torwic_slam --output reports\warehouse_render_next
.\.venv\Scripts\python.exe src\validation\renderer_pilot_checks.py --render-root reports\warehouse_render_next --raw-root data\raw\torwic_slam --output reports\warehouse_package_next
```

Choose fresh output directories. The pilot creates native RGB, clean float32 axial
depth, geometry semantic IDs, binary goods targets, poses, settings and a local ZIP.
It uses an independent zero-distortion synthetic camera with the reference left
camera K. It does not certify the real exported-image convention. An analytic
plane and independent scene rays verify depth/class alignment before packaging.

Shared cuboid geometry improves construction efficiency without reducing resolution
or maximum sampling quality. Pilot appearance is a prototype; clean depth has no
calibrated Azure failure model. Training eligibility remains false until reference
labels, camera/domain appearance and downstream usefulness are reviewed. No IMU
or spinning-LiDAR simulation is claimed. Keep all development local until the
user authorizes publication.

The local `.tools/renderer_runtime.json` records the installed executable and official
archive checksum. It stays excluded from Git. On another machine, install the same
portable Blender version and create this local runtime file with its executable path.

Validation summaries under `docs` are snapshots taken before publication. Their
`committed`/`pushed` fields describe the review-time state, not live GitHub status.
Use Git history and branch tracking for the current publication state. Generated
reports, datasets, renderer installations and machine paths remain local.
