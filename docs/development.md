# Development guide

## Select a runtime

Use Python 3.12 and the existing pinned environments. The project-local `.venv`
must not be assumed to work or contain CUDA packages. Machine-local runtime paths
belong in ignored `.tools/training_runtime.json`, outside portable job files.

From the repository root, resolve and inspect the configured interpreter:

```powershell
$runtime = Get-Content .tools/training_runtime.json -Raw | ConvertFrom-Json
$pythonExe = $runtime.python_executable
& $pythonExe scripts/check_runtime.py
```

If it is unavailable, explicitly select an existing environment. A historical
execution report records an earlier runtime; check that environment before use:

```powershell
$pythonExe = 'C:\path\to\verified-environment\Scripts\python.exe'
& $pythonExe scripts/check_runtime.py --require-cuda
```

For a different runtime JSON, explicitly read its `python_executable` field using
the same procedure. The diagnostic checks Python/core-package availability and
reports package versions. It does not compare versions against the pinned files.
`--require-cuda` also exercises a small CUDA matrix operation. No packages are
installed and no training artifacts are written. This does not certify model
memory capacity, renderer availability, or dataset qualification.

## Environment specifications

| File | Purpose |
| --- | --- |
| `requirements/pipeline_core.txt` | Portable host contracts and basic array/image handling |
| `requirements/development.txt` | Tests and formatting |
| `requirements/mvp_tools.txt` | Ingestion and broader data-processing snapshot |
| `requirements/depth_research.txt` | Depth research dependencies |
| `requirements/torch_cu128.txt` | CUDA PyTorch/torchvision pins and package index |
| `requirements/blenderproc_host.txt` | Separate BlenderProc host snapshot |

Do not install all files into one environment. Keep renderer and training
dependencies isolated. Preserve verified versions; updates require their own
validation. Install the host package into a prepared environment using:

```powershell
& $pythonExe -m pip install --no-deps --no-build-isolation -e .
```

This does not install missing dependencies or qualify a backend.

## Entry points and verification

```powershell
& $pythonExe -m src.pipeline.run --help
& $pythonExe -m src.pipeline.prepare_benchmark --help
& $pythonExe -m src.training.goods_benchmark --help
& $pythonExe -m src.pipeline.scene_benchmark --help
& $pythonExe -m src.profiling.goods_failure_audit --help
& $pythonExe -m pytest -q -p no:cacheprovider
```

The installed `dubaidata-prepare` command invokes `src.pipeline.run`. Keep module
entry points compatible with existing workflows. Tests include small known-label
training fixtures, not customer benchmark runs. Compile changed modules and run
appropriate checks. Do not retrain completed experiments to verify documentation.
Repeat full ingestion only when relevant source/loader changes justify it.

## Code organization

- `core`: portable contracts, with no renderer APIs.
- `adapters/datasets`: source conventions and split-membership checks.
- `validation/goods_references.py`: qualified reference loading shared by training
  and diagnostic consumers.
- `training/goods_model.py`: model/optimizer construction, loss, and benchmark
  configuration validation; PyTorch imports remain lazy.
- `training/goods_benchmark.py`: training/evaluation orchestration and the existing
  CLI; moved public helpers remain available as compatibility imports.
- `pipeline`: job execution and experiment orchestration.
- `synthesis`: reusable builders with separate planning responsibilities.

Shared dataset verification should not require importing profiling or a training
entry point. Document public contracts, units, frames, shapes, masks, provenance,
and failure behavior.

## Configuration and reproducibility

Portable jobs, task specifications, and runtime configuration serve different
purposes. Keep executable paths, storage roots, and credentials in local runtime
configuration. `configs/runtime/*.example.json` provides structural examples;
resolve local paths and exact source bindings before execution.

Frozen benchmark configurations and historical profiles remain unchanged. Similar
JSON files may intentionally describe different budgets, losses, or cohort versions.
Future shared defaults must resolve to a complete validated configuration with a
checksum; configuration inheritance is not currently implemented.

Use fresh output destinations. Preserve raw inputs, accepted datasets, reports,
checkpoints, and source/task/cohort/code bindings. The ignored local report index
provides navigation and does not grant acceptance.

## Repository and editor hygiene

Keep generated artifacts ignored. Hide caches and build artifacts in the editor
rather than deleting evidence. Historical documents retain their existing paths.

Develop on `dev` or a feature branch based on it. Preserve unrelated edits.
Review explicit paths and complete staged diffs before a commit; publication and
integration into `main` require user authorization. See [AGENTS.md](../AGENTS.md).
