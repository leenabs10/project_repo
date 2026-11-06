# Pipeline architecture

DubaiDATA separates dataset conventions, generation engines and execution
infrastructure. TorWIC and warehouse goods are the reference application.

```text
Customer inputs → dataset adapter → canonical records → profile and coverage
→ generation specification → qualified backend → validation and admission
→ traceable dataset → optional matched real-data evaluation
```

| Component | Location | Responsibility |
| --- | --- | --- |
| Contracts | `src/core/` | Versioned data/scene schemas and local path resolution |
| Adapters | `src/adapters/` | Dataset, task, backend, storage and execution boundaries |
| Source decoding | `src/ingestion/` | RGB, depth, point clouds, calibration and timing |
| Profiling | `src/profiling/` | Quality, coverage and saved-error diagnostics |
| Orchestration | `src/pipeline/` | Preparation, generation and matched experiments |
| Scene planning | `src/synthesis/` | Supported warehouse geometry and conditions |
| Validation | `src/validation/` | Integrity, geometry, reference reviews and admission |
| Benchmark | `src/training/` | Task recipes, sampling and evaluation |

## Code and local artifacts

The repository contains maintained source, tests and portable specifications.
`configs/local.json` selects the external workspace and Python interpreter. It
is ignored by Git. Use `configs/local.example.json` when configuring another machine.

The workspace holds `data`, `reports`, `models`, `assets`, `services` and archived
local operators. Frozen payloads keep their original bytes and profile hashes.
`src/core/paths.py` resolves recorded deployment paths where a maintained boundary
needs them; it does not rewrite calibration, labels or generation specifications.

## Capability boundaries

The working backend is Blender Cycles with RGB, axial depth, validity, semantic
and instance outputs. Bounded RGB gain and depth missingness are supported
conditions. Qualified simulated LiDAR, calibrated sensor noise, physics and remote
execution remain open. Camera-depth point clouds do not satisfy LiDAR requirements.
Additional backends require asset, camera, sensor, label and appearance qualification.

Depth restoration remains a conditional secondary benchmark. Existing module
paths are retained for compatibility; the source map identifies the primary flow.
