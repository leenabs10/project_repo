# Source map

Follow the workflow below when finding code. Existing Python module paths stay
stable because CLIs, renderer workers and frozen experiments import them.

| Step | Start with | Responsibility |
| --- | --- | --- |
| Read customer data | `adapters/datasets/registry.py`, `adapters/datasets/torwic.py` | Dataset selection and format-specific conventions |
| Validate shared records | `core/contracts.py`, `core/scene.py` | Canonical data, generation and scene contracts |
| Profile coverage | `profiling/dataset_profile.py`, `profiling/goods_failure_audit.py` | Input quality, coverage and saved-error evidence |
| Execute preparation | `pipeline/run.py`, `pipeline/cache.py` | Supported local jobs and derivatives |
| Plan warehouse scenes | `synthesis/equipment_pairs.py`, `synthesis/grouped_cargo.py` | Bounded geometry and cargo arrangements |
| Render | `pipeline/scene.py`, `adapters/generation/blender_scene_worker.py` | Isolated backend execution and export |
| Qualify and admit | `validation/scene_quality.py`, `validation/scene_admission.py` | Geometry checks and source-bound dataset eligibility |
| Review labels | `validation/review_server.py`, `validation/reference_labels.py` | Versioned reference editing and import validation |
| Train and compare | `pipeline/scene_benchmark.py`, `training/goods_benchmark.py` | Matched multi-seed execution and evaluation |
| Inspect predictions | `validation/goods_predictions.py` | Native exports and reporting |

## Supporting code

`ingestion/` decodes source payloads and timing; `adapters/` owns dataset, task,
backend, execution and storage boundaries. `training/goods_model.py` contains
model/loss configuration, while sampling lives in `training/auxiliary_inputs.py`
and `training/scene_inputs.py`.

Depth-restoration training, inference and diagnostics remain a conditional
secondary benchmark. Earlier warehouse planners remain dependencies of retained
experiments; a historical name alone does not make a module unused.

## Change rules

Keep public imports and command interfaces compatible. Put executable jobs behind
entry points, document units/shapes/frames, and explain nonobvious assumptions.
Use focused contract tests. Do not rerun training for navigation or documentation
changes. See the root README and docs/run.md for execution.
