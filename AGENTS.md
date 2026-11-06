# Repository development policy

## Architecture and scope

- Maintain a dataset-independent core, simulator-independent generation contracts
  and infrastructure-independent execution. Current specifications are in
  `docs/architecture.md` and `docs/scope.md`.
- Keep dataset formats, split assignments and sensor conventions in dataset
  adapters. Keep task semantics in task adapters and engine APIs in isolated
  renderer workers. Executable paths, credentials and storage roots are deployment
  configuration, outside portable customer specifications.
- Check required modalities and fidelity before execution. Reject unsupported
  versions, fields, assets and backends instead of silently approximating them.
- Goods segmentation is the primary benchmark. Depth restoration is a separate,
  conditional benchmark. Camera point clouds do not satisfy simulated LiDAR.
- Select additional backends when sensor, physics or fidelity requirements justify
  them. Isaac Sim and cloud execution are candidates, not mandatory destinations.
  Every backend needs asset/material conversion and camera, geometry, depth, label
  and appearance fixtures before its outputs can be qualified.
- Distinguish verified capabilities, experimental implementations and target
  architecture. Report remaining limits explicitly.

## Git and publication

- Develop on `dev` or a feature branch based on `dev`. Check the active branch,
  working tree, staging area and remote divergence before Git mutations.
- Publish only the reviewed scope requested by the user. Earlier publication does
  not authorize subsequent pushes. Stage explicit paths and inspect the complete
  staged diff, file list and whitespace checks before committing.
- Require explicit permission for each integration into `main`. A passing test
  suite or a development push does not authorize a main merge or push.
- Preserve unrelated edits, local experiments, checkpoints and report history.
  Use an isolated checkout when publishing an earlier verified snapshot. Never
  discard current work to assemble that snapshot.
- Do not force-push or rewrite shared history. Stop and inspect unexpected remote
  divergence before proceeding.
- Keep datasets, generated reports, environments, weights, downloaded assets,
  machine-local runtime files and credentials out of Git. Small aggregate evidence
  belongs under `docs` with its source/version binding.
- Use professional commit titles describing behavior and validation. Confirm the
  destination branch and resulting commit after an authorized push.

## Readability and presentation

- Name modules, reports and configurations by purpose. Use capability milestones
  and acceptance criteria in stakeholder documents; keep private scheduling and
  conversational history outside the repository presentation.
- Write focused functions with explicit imports and descriptive variable names.
  Expand dense control flow and apply consistent formatting.
- Document public interfaces, units, coordinate frames, array shapes/dtypes,
  validity masks, return values and failure behavior. Comments should explain
  assumptions, decisions and nonobvious constraints.
- Put executable I/O, training and report generation behind entry points.
  Importing reusable modules must not execute jobs or modify artifacts.
- For comment/docstring-only edits, compare executable ASTs and compile the
  changed modules. Do not retrain solely because documentation changed.
- Use neutral interface labels such as reference candidate, qualification status
  and saved version. Preserve actual annotation origin and assessor provenance in
  metadata; presentation changes must not imply independent ground truth.
- Keep actionable review links and operator procedures in local ignored reports
  and conversation. Public documents should describe capabilities, results,
  qualification criteria and limitations.

## Source data and provenance

- Treat `data/raw` as immutable. Write caches, derivatives, corrections, predictions
  and reports outside that root.
- Preserve sample and trajectory IDs, timestamps, sensor side, calibration,
  source paths, units, masks and annotation provenance through every stage.
- Validate native shape, dtype, finite values, class IDs, payload layout and counts.
  Enforce path containment and agreement between split labels and source paths.
- Native TorWIC RGB is uint8 at 1280x720. Measured depth is uint16 millimetres
  scaled by 0.001 to metres; zero is invalid and excluded from supervision.
- Binary XYZ LiDAR uses the verified publisher layout. Preserve stored/usable
  counts when filtering nonfinite or zero points. Do not invent intensity values.
- Retain clean references separately from corrupted inputs. Predictions and inferred
  measurements must not silently become ground-truth targets.
- Trajectory semantic masks are model predictions. Additional indexed annotations
  require their per-file label mapping and verified collection/split provenance.
  Keep unqualified extra pairs quarantined.
- Bind sources, manifests, calibration, configurations, code and checkpoints with
  checksums when producing evidence. Preserve historical version/hash relationships.
  Verify lossless cache and categorical export round-trips.
- Document resampling as a derivative. Update intrinsics when changing image
  geometry; avoid depth interpolation across discontinuities and semantic-ID
  interpolation.

## Splits and evaluation isolation

- Training trajectory: `Aisle_CCW_Run_1`, 1,136 frames.
- Development trajectory: `Aisle_CW_Run_1`, 1,037 frames.
- Locked test trajectory: `Aisle_CCW`, 915 frames.
- Split by complete trajectory. Derive profiles, condition parameters, donor masks
  and coverage requirements from training data only.
- Development supports model selection and comparisons. Disclose dependent video
  frames and repeated development use; they do not establish unbiased generalization.
- Freeze the candidate and protocol before final test evaluation. Integrity checks
  on test files do not authorize model evaluation or tuning.
- The current qualified reference cohort has 32 training and 12 development
  samples, bound to profile SHA256
  `37ab46daec17a248cc5dc42016113159d42798e64fafbfcafe9ea88a71a36be8`.
  Explicitly verify the intended revision for each experiment.
- Keep empty pallets, empty handling containers and transport equipment background.
  Identifiable loads and stored goods are foreground. Unresolved real regions remain
  ignored with ID 255; do not infer hidden contents or surfaces.
- Preserve existing annotations and retained models. Reannotation requires a
  demonstrated target error, not a lower score or an unsuccessful experiment.

## Sensor timing and geometry

- Preserve the supported inverse camera-pose convention:
  `p_camera = R.T @ (p_os - t)` with xyzw quaternions.
- Camera depth comparisons use axial Z in metres. Use nearest-surface projection
  and report edge-filtered and full-overlap residuals separately.
- Keep distortion alternatives and registration limitations explicit. Projection
  residuals alone do not prove calibration or physical sensor accuracy.
- Strict IMU loading rejects backward ordering. Explicit stable sorting produces
  a copy and records anomalies; raw streams remain unchanged.
- Reject duplicate timestamps until a resolution policy is defined. Require
  explicit gap limits, flag queries outside the stream span and never extrapolate.
- Timestamp bracketing is separate from measurement interpolation, vector-frame
  conversion, preintegration and fusion. Real measurement mapping remains disabled
  until source-bound unit/frame/clock evidence is verified.

## Synthesis quality and admission

- Data quality takes priority over volume and execution speed. Keep failed or
  unqualified outputs quarantined with their reasons and source bindings.
- Preserve clean RGB/depth/label correspondence. Record algorithm, seed, resolved
  parameters, donor identity, masks, scene geometry and source lineage.
- Separate controlled stress tests from sensor-calibrated simulation. Do not infer
  sensor noise from spatial depth changes or projection residuals without resolving
  geometry, edges, occlusion and timing effects.
- Verify metric units, poses, intrinsics, extrinsics, timing, visible labels and
  invalid masks before expanding a generated batch.
- Qualify appearance, geometry, labels and downstream utility separately. A
  histogram match or a passing renderer fixture does not establish realism.
- Do not treat silence, automated checks or a prepared gallery as human approval.
  Bind explicit acceptance to the exact dataset/job and preserve assessor provenance.
- Dataset admission must verify source artifacts, decisions, task/cohort versions
  and split isolation. Eligibility is separate from successful generation.
- Keep retained measurements unchanged in clean mode. Any denoising mode requires
  observed-pixel nonregression evidence; rejected modes remain blocked.
- Measure accepted-sample throughput, rejection rate, coverage, memory/storage and
  task utility before making scalability claims. Preserve IDs, seeds and acceptance
  semantics under partitioning or parallel execution.

## Training and comparison

- Declare target origin, input channels, normalization, valid-target mask, loss,
  optimizer, budget, seeds and checkpoint-selection policy before a run.
- Prevent withheld-target leakage through caches, derived features, preprocessing,
  donor selection or nearest/smoothed inputs.
- Compare real-only and hybrid runs with matched initialization, update budgets,
  selection policies and evaluation references. Disclose changes in data counts
  or real/synthetic presentations.
- Use multiple seeds for improvement claims. Distinguish pixel-weighted from
  frame-weighted metrics and report both gains and regressions.
- Preserve ignored targets during supervision and scoring. Reject nonfinite
  losses, gradients and outputs. Reload saved checkpoints strictly for evaluation.
- Report depth errors in metres and the evaluated regions. Aggregate gains must
  not conceal damage to retained measurements, boundaries or important equipment.
- Keep depth-restoration, goods segmentation and sensor-fidelity conclusions
  separate. None alone establishes navigation, SLAM or production safety.
- A configuration, fixture run or generated dataset does not demonstrate synthetic
  utility. Retain rejected candidates and do not promote a model automatically.

## Verification and environment

- Test changed contracts meaningfully: paths, splits, masks, units, timing,
  coordinate transforms, label preservation, admission and checkpoint loading.
- Compile changed Python modules and run checks on the exact publication snapshot.
  Reuse verified caches; repeat full ingestion only when source/loader changes
  justify the cost.
- Preserve the verified package versions. Resolve deployment interpreters through
  `configs/local.json`; the ingestion environment is not assumed to
  include CUDA training packages.
- Bound CPU workers and GPU batches. Report actual resolution, precision, hardware,
  measured peak memory and timing scope when benchmarking.
- Current capability/status evidence is in `docs/status.md` and
  `archive/docs/scene_generation.txt`. Keep published evidence consistent with published
  code; later local experiments remain separate.


## Repository and workspace layout

- Keep maintained code, tests, portable configurations and concise active docs in
  the source repository. The external workspace is selected in ignored
  `configs/local.json`; its data, assets, models, reports and local operators are
  not source files and must remain outside Git.
- Preserve archived documentation/configuration bytes under `archive/`. Preserve
  original metadata path strings; resolve deployment paths at maintained entry
  points and record relocations in the workspace migration ledger.
- Protect both original and relocated raw-data locations. Migration does not
  authorize changes to dataset contents, saved reference pixels or past scores.
- Portable run commands belong in `docs/run.md`. Machine-specific review links,
  service state and credentials belong in the ignored external workspace.
