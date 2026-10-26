# DubaiDATA Synthetic Data MVP

A customer-adaptable synthetic-data pipeline for inspecting datasets, identifying
coverage gaps, generating controlled variations, qualifying outputs, and measuring
their value on real data. Warehouse goods segmentation on TorWIC-SLAM is the first
reference application; depth restoration is a separate conditional benchmark.

## Pipeline

```text
Customer dataset -> dataset adapter -> canonical records and coverage profile
-> generation specification -> supported backend -> quality and admission
-> traceable dataset export -> matched task evaluation
```

The core is dataset-independent, generation contracts are simulator-independent,
and execution is separated from deployment infrastructure. These boundaries do
not imply that every dataset, simulator, or cloud is supported.

## Current capabilities

| Area | Implemented capability and limit |
| --- | --- |
| Ingestion | TorWIC adapter and declared paired-PNG input; 3,088 TorWIC frames and 21,616 payloads audited |
| Preparation | Versioned contracts, source hashes, split checks, RGB/depth caches, local execution |
| Sensor conditions | Bounded RGB gain and depth missingness; calibrated sensor noise remains unsupported |
| Scene generation | Isolated Blender Cycles RGB, axial depth, semantic and instance labels; bounded accepted batches |
| Goods evaluation | Source-bound 30-training/12-development reference cohort and matched multi-seed comparisons |
| Synthetic utility | Overall segmentation benefit remains unestablished; no automatic model promotion |
| Other backends | BlenderProc integration, simulated LiDAR, physics, and remote execution remain unavailable |

The completed equipment-pair experiment gives real-only versus real + synthetic
goods IoU **51.63% -> 51.42%**, precision **55.86% -> 54.40%**, and recall
**87.42% -> 90.40%**. It fails its declared utility rule. A later threshold
diagnostic explores tradeoffs on the reused development cohort; it does not
reverse that result. The locked test remains unevaluated.

See [current status](docs/goods_benchmark_status.txt),
[equipment results](docs/equipment_pair_results.txt), and
[threshold diagnostic](docs/goods_threshold_diagnostic_results.txt).

## Start here

1. Read the [MVP scope](docs/mvp_scope.txt) and [architecture](docs/pipeline_architecture.txt).
2. Follow the [development guide](docs/development.md) to select and check an interpreter.
3. Use the [documentation index](docs/README.md) to find contracts and experiment evidence.
4. Read [AGENTS.md](AGENTS.md) before changing code or publishing.

With a verified interpreter selected as `$pythonExe`, inspect the preparation CLI:

```powershell
& $pythonExe -m src.pipeline.run --help
```

Jobs require explicit configuration and fresh output paths. See the
[configuration guidance](docs/development.md#configuration-and-reproducibility)
before execution.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/core/` | Portable contracts and JSON schemas |
| `src/adapters/` | Dataset, task, renderer, storage, and execution integrations |
| `src/pipeline/` | Workflow orchestration and comparisons |
| `src/ingestion/`, `src/profiling/` | Source decoding, characterization, and audits |
| `src/synthesis/` | Generation planning and controlled variation methods |
| `src/training/`, `src/inference/` | Task models, training, and prediction |
| `src/validation/` | Reference verification, qualification, admission, and review tooling |
| `configs/`, `tests/`, `requirements/` | Specifications, contract checks, and pinned environments |
| `docs/` | Guidance and small source-bound evidence |
| `scripts/` | Local development diagnostics |

Datasets, reports, outputs, environments, assets, runtime paths, and weights stay
outside Git. Preserve raw inputs and historical references, checkpoints, and reports.
Goods references retain their recorded annotation/review provenance; publisher
trajectory masks are predictions. Camera-depth point clouds are not simulated LiDAR.
