# DubaiDATA Synthetic Data MVP

DubaiDATA is developing a customer-adaptable, simulator-independent synthetic-data
pipeline. It ingests customer datasets through adapters, standardizes modalities
and metadata, identifies coverage requirements, generates controlled sensor and
scene variations, validates quality, and exports traceable datasets for
customer-specific ML applications.

The architecture follows three principles:

- **Dataset-independent core**
- **Simulator-independent generation contracts**
- **Infrastructure-independent execution**

[MVP scope](docs/mvp_scope.txt) · [Architecture](docs/pipeline_architecture.txt) ·
[Delivery milestones](docs/mvp_delivery_plan.txt) · [Capability assessment](docs/dataset_readiness.txt)

## Reference application

TorWIC-SLAM warehouse AMR data is the reference integration. It includes RGB,
depth, LiDAR, calibration, trajectory information and predicted semantic labels.
Goods segmentation is the primary benchmark; depth restoration is a conditional
secondary benchmark. Dataset conventions and task semantics belong in adapters,
not in the reusable pipeline core.

The workflow is:

```text
Customer data -> adapter -> standardized profile -> synthesis specification
-> supported generation backend -> quality acceptance -> dataset export
-> optional hybrid training and independent real-data evaluation
```

## Capabilities and integration status

| Area | Evidence or integration status |
| --- | --- |
| Data integrity | 3,088 frames and 21,616 sensor payloads audited |
| Depth caches | Both resolutions verified against manifests and source derivatives |
| Research workflows | Preparation, training, prediction and audit scripts integrated |
| Scene pilot | Custom Blender RGB/depth/label geometry and export checks passed |
| Shared preparation | Canonical contracts, two dataset input formats and local worker/executor verified |
| Real-sample conditions | Native RGB gain/depth-mask jobs verified; 11 passed automated checks, 7 rejected; release qualification in progress |
| Goods baseline | Real-only training, strict reload and native evaluation verified on known fixtures; dataset qualification remains separate |
| BlenderProc | Tools prepared; generation integration and acceptance pending |
| Goods references | Native annotation, source-bound import and provenance checks implemented; reference accuracy requires separate qualification |
| Simulated LiDAR | Dedicated generation unavailable; camera point clouds are not LiDAR scans |
| Downstream utility | Controlled goods comparison and independent evaluation required |

BlenderProc is the intended camera-generation backend. Isaac Sim is an additional
candidate when customer sensor, physics or fidelity requirements justify the work.
Local and cloud execution are separate deployment choices. Neither a simulator nor
a cloud provider is a predetermined production destination.

## Data and quality rules

- Preserve raw data and clean references; write derivatives to separate locations.
- Keep complete-trajectory splits and protect held-out real data from tuning.
- Record source IDs, parameters, seeds, calibration, versions and artifact hashes.
- Exclude invalid depth and uncertain labels from supervision.
- Check geometry, labels, requested conditions, fidelity, coverage and lineage.
- Treat hypothetical corruption as controlled stress synthesis until calibrated.
- Require reference-label accuracy and appearance qualification before dataset release.

## Local development commands

Run from the project root using the appropriate Python environment. Explicit
interpreter paths avoid dependence on shell activation scripts.

Inspect inputs and verify native payloads:

```powershell
.\.venv\Scripts\python.exe src\ingestion\inspect_torwic.py --root data\raw\torwic_slam
.\.venv\Scripts\python.exe src\validation\full_ingestion_checks.py --root data\raw\torwic_slam --manifest data\manifests\manifest.jsonl --output reports\ingestion_review_next\full_payload_checks.json
```

Use the CUDA-enabled research interpreter recorded in the local runtime configuration:

```powershell
$trainingPython = (Get-Content .tools\training_runtime.json -Raw | ConvertFrom-Json).python_executable
& $trainingPython -m unittest discover -s tests
& $trainingPython -m src.validation.depth_cache_review --output reports\depth_cache_review_next
& $trainingPython -m src.training.depth_quality --help
& $trainingPython -m src.inference.depth_quality --help
```

Use fresh output paths. Full payload audits and cache-source comparisons are
intentionally comprehensive; reuse matching evidence when inputs and code have
not changed. Do not retrain merely to verify documentation or formatting changes.

Dataset profiling and task preparation:

```powershell
.\.venv\Scripts\python.exe src\profiling\dataset_profile.py --root data\raw\torwic_slam --manifest data\manifests\manifest.jsonl --output reports\dataset_profile_next --workers 4
.\.venv\Scripts\python.exe src\profiling\task_dataset_review.py --root data\raw\torwic_slam --manifest data\manifests\manifest.jsonl --profile reports\dataset_profile\summary.json --output reports\task_review_next
```

Existing renderer pilot:

```powershell
$renderer = (Get-Content .tools\renderer_runtime.json -Raw | ConvertFrom-Json).blender_executable
& $renderer --background --factory-startup -t 8 --python-exit-code 1 --python src\synthesis\warehouse_renderer.py -- --config configs\warehouse_renderer.json --root data\raw\torwic_slam --output reports\warehouse_render_next
.\.venv\Scripts\python.exe src\validation\renderer_pilot_checks.py --render-root reports\warehouse_render_next --raw-root data\raw\torwic_slam --output reports\warehouse_package_next
```

The renderer command launches generation; installation alone does not establish
quality acceptance. The existing pilot uses its documented synthetic camera
convention and does not certify the real exported-image distortion convention.

## Evidence and reproducibility

[Ingestion validation](docs/ingestion_validation.json) records payload counts,
split membership and the immutable original audit record.
The representative diagnostic CLI is `python -m src.validation.representative_ingestion_checks`;
new runs write `representative_ingestion_checks.json` in their selected output directory.

The pinned `configs/augmentation_policy.json` uses dimensionless RGB gains,
8-bit luminance limits and depth-validity fractions between zero and one.
Its method descriptions distinguish controlled stress from calibrated sensor
simulation. JSON remains strict JSON; explanatory comments belong in its generator,
`src/synthesis/augmentation_policy.py`, so configuration hashes remain stable.

[Depth verification](docs/depth_cache_verification.txt) records cache contracts,
mask handling, donor isolation and baseline metrics.
[Pipeline organization](docs/pipeline_integration.txt) records research environment
and checkpoint compatibility evidence. [Checkpoint inventory](configs/checkpoint_registry.json)
identifies local weights and their limitations. Dependency snapshots are in
`requirements/`; source calibration and label provenance remain attached to datasets.

Raw data, generated reports, environments, credentials, runtime paths and model
weights remain excluded from version control. See [development instructions](AGENTS.md)
for repository workflow and evidence-preservation rules.

## Adapter-based preparation and technology stack

[Verified stack](docs/tech_stack.txt) and [adapter evidence](docs/adapter_foundation.json)
describe implemented preparation boundaries and remaining generation integrations.
The host package includes the core JSON schemas; existing research CLIs remain available.

```powershell
& $trainingPython -m pip install -r requirements/pipeline_core.txt -r requirements/development.txt
& $trainingPython -m pip install --no-deps --no-build-isolation -e .
& $trainingPython -m src.pipeline.run --job configs/jobs/torwic_cache.json --runtime configs/runtime/torwic_local.example.json --output reports/adapter_cache_next
```

The portable job contains dataset identity, manifest hash and task parameters. Local
paths and the selected dataset adapter belong to separate runtime configuration.
This command prepares RGB/depth caches; it does not launch simulation or training.
Scene backends fail explicitly until their requested capabilities are integrated.

## Controlled real-sample synthesis

[Condition-generation evidence](docs/sensor_condition_generation.txt) documents the
implemented RGB gain and training-donor depth-mask methods, portable customer controls,
source binding and shared quality checks. LiDAR and labels are preserved unchanged.
The native-resolution pilot passed 11 of 18 variants through automated checks;
dataset release qualification and complete requested coverage remain outstanding.

```powershell
& $trainingPython -m src.pipeline.run --job configs/jobs/torwic_sensor_conditions.json --runtime configs/runtime/torwic_sensor_conditions.example.json --output reports/sensor_conditions_next
```

The runtime requires the exact training profile, parent policy and calibration files
referenced by their hashes. Use a fresh output directory. Generated variants remain
excluded from training until dataset acceptance criteria are satisfied.

## Benchmark preparation and qualification

The native reference editor preserves source bindings, categorical masks and
annotation version history. The importer retains original submission bytes and
distinguishes model-assisted corrections from confirmed reference versions.
Technical validation does not establish annotation accuracy or benchmark readiness.

The real-only training/evaluation CLI uses explicit configurations and rejects
unqualified targets, held-out membership and changed payload checksums. Ignored
pixels remain excluded from supervision. Checkpoints reload strictly, and logits
are evaluated at native reference geometry. Known-fixture execution does not
establish customer accuracy, model improvement or synthetic-data utility.

```powershell
& $trainingPython -m src.pipeline.prepare_benchmark --help
& $trainingPython -m src.training.goods_benchmark train --help
& $trainingPython -m src.training.goods_benchmark evaluate --help
& $trainingPython -m src.training.goods_baseline --help
```

Experimental training extensions, dataset-specific model results and pending
annotation revisions are excluded from this verified software release.
