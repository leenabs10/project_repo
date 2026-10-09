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
