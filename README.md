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
