# MVP development instructions

These rules apply to the repository and its subdirectories. Follow the user's
current instructions and the verified MVP scope. Distinguish completed evidence
from proposed work; do not invent requirements from an unavailable planning document.

## Branches, commits, and publishing

- Implement on `dev` or on a feature branch created from `dev`. Check branch and
  working-tree status before editing, staging, committing, or publishing.
- Never merge, rebase, cherry-pick, reset, or push development changes into `main`
  without the user's explicit permission for that specific integration. Tests
  passing or a task finishing do not grant merge permission.
- Batch work locally during the current task/day. **Do not push while today's
  work is in progress.** Push only after the task batch is complete and the user
  requests or confirms publication. Earlier pushes do not authorize another push.
- Local commits may capture verified work on the development branch. Keep them
  focused, with professional titles explaining the resulting behavior.
- Preserve the user's changes. Do not discard edits, force-push, rewrite shared
  history, or resolve conflicts by blindly taking one side.
- Before a commit, inspect both the working diff and staged diff. Stage explicit
  reviewed paths, run `git diff --check`, and inspect the staged file list for
  datasets, generated artifacts, credentials, or unrelated changes.
- Before an authorized push, check the remote/branch and divergence. Push the
  named development branch; never use a command that publishes every branch.
- Before a main integration, present the concrete changes, test evidence, and
  remaining risks, then ask for permission. Leave main untouched while waiting.

## Code readability and naming

- Readability is a requirement. Use descriptive names based on purpose, such as
  `dataset_profile`, `sensor_alignment_review`, and `imu_timing`; avoid calendar
  week or phase-number names for new code, reports, and commits.
- Keep functions focused and expand dense statements. Avoid compressed control
  flow, multiple actions on one line, and unexplained numeric constants.
- Add docstrings for public functions and important data contracts: purpose,
  arguments, units, array shape/dtype, masks, return values, and failure behavior.
- Comments should explain decisions, assumptions, and limitations rather than
  restate obvious assignments. State coordinate conventions and normalization
  scales where they are used.
- Use consistent formatting and explicit imports. Put executable I/O, training,
  and report generation behind a main entry point; importing a reusable module
  must not launch a run or overwrite artifacts.
- Preserve historical evidence names and hashes unless a migration is explicitly
  requested. For readability-only edits, verify executable AST equivalence and
  syntax compilation; do not retrain merely because comments changed.

## Data integrity and provenance

- Raw files under `data/raw` are immutable. Reports, caches, corrected copies,
  augmentations, and model predictions must be written outside the raw root.
- Retain sample IDs, trajectory IDs, timestamps, sensor side, source paths,
  calibration references, units, and label provenance across every stage.
- Validate shape, dtype, finite values, class IDs, payload layout, and counts.
  Reject paths escaping the dataset root and split/path inconsistencies.
- Native RGB is 1280x720 uint8. Measured depth is uint16 millimetres converted by
  0.001 to metres; zero is invalid and must remain excluded from supervision.
- Binary XYZ LiDAR uses the verified publisher layout. Preserve stored-versus-
  usable counts when filtering nonfinite/zero points; do not invent intensity.
- Keep original clean references separate from corrupted inputs. Model predictions
  and inferred unknown depths must never silently become ground-truth labels.
- Trajectory semantic masks are model predictions. Audit the source, class mapping,
  and split provenance of extra human annotations before using them for training.
- Hash sources, manifests, calibration, code, and checkpoints when creating
  reproducible evidence. Keep hashes tied to their actual version, not rewritten
  to imply an old experiment used new code. Verify lossless cache/export round-trips.
- Resampling is a documented derivative, not a replacement for native data. Avoid
  blending depth across discontinuities or interpolating semantic class IDs.
  Update camera intrinsics when changing image geometry.
- Exclude datasets, environments, model weights, generated reports, and secrets
  from Git. Small reviewed aggregate evidence may be committed under `docs`.

## Dataset splits and leakage prevention

- Confirmed training: `Aisle_CCW_Run_1`, June 15, 2022, 1,136 frames.
- Development validation: `Aisle_CW_Run_1`, June 23, 2022, 1,037 frames.
- Held-out test: `Aisle_CCW`, October 12, 2022, 915 frames.
- Split by complete trajectory, never randomly across adjacent video frames.
- Derive profiling statistics, augmentation proposals, and donor masks from
  training data only. Validate record paths as well as split labels and IDs.
- Validation supports development comparisons and model selection; report that
  reuse honestly. It is not an unbiased estimate after repeated selection.
- Do not tune on October. Read-only integrity inspection is distinct from model
  evaluation. Freeze the candidate and evaluation protocol before final testing;
  report failures without changing the model against the same test results.

## Sensor timing and alignment

- Preserve the inverse camera-pose convention already supported by training
  diagnostics: `p_camera = R.T @ (p_os - t)`; quaternions are xyzw.
- Camera depth comparisons use axial Z in metres, not Euclidean LiDAR range.
  Use nearest-surface projection and separate edge-filtered from full overlap.
- Keep distortion alternatives and limitations visible. Overlap residuals are
  not independent calibration accuracy or proof of production registration.
- Strict IMU loading rejects backward ordering by default. Explicit stable
  sorting returns a copy and logs anomalies; never silently rewrite raw streams.
- Reject duplicate timestamps until a resolution policy is defined. Flag queries
  outside the IMU span, never extrapolate, and require explicit gap limits.
- Timestamp bracketing does not implement measurement interpolation, vector-frame
  conversion, or hardware timing verification. Audit these before adding fusion.

## Synthetic-data quality

- Preserve clean RGB/depth/semantic references and their geometric relationship.
  Record every augmentation's algorithm, seed, parameters, donor ID, and masks.
- Separate controlled stress tests from sensor-calibrated simulation. Gaussian
  noise or transferred missing-depth masks are not physically realistic merely
  because a rendered example looks plausible.
- Do not estimate sensor noise directly from spatial depth variation or projection
  residuals without accounting for scene geometry, edges, occlusions, and timing.
- Evaluate appearance, geometry, labels, and downstream usefulness separately.
  Rendered camera point clouds are not equivalent to a spinning OS1-128 scan.
- Keep clean-mode observed measurements unchanged. Denoising that changes them
  requires an observed-pixel non-regression check; keep rejected modes blocked.
- Verify units, poses, intrinsics, sensor extrinsics, timing, and label alignment
  for rendered samples before scaling generation. Benchmark a small batch first.

## Training and comparison rules

- Declare the task, target provenance, input channels, normalization, valid-target
  mask, loss, optimizer, budget, selection metric, and seed before a run.
- Test that withheld target values cannot leak through nearest/smooth features,
  derived inputs, caches, donor selection, or preprocessing.
- Use fixed validation corruptions across comparisons. Vary training corruptions
  reproducibly and separate model seeds from data-generation seeds.
- Compare against appropriate simple baselines and modality ablations. Keep data
  counts, initialization policy, and compute budgets comparable, or disclose differences.
- Use multiple seeds for an improvement claim. Distinguish pixel-weighted from
  frame-weighted metrics and dependent video frames from independent observations.
- Report MAE/RMSE in metres and metric regions. For depth repair, separately inspect
  hidden, observed, boundary, and hole-interior behavior as applicable. A better
  aggregate score must not hide damage to retained measurements.
- Reject nonfinite loss/output, reload saved checkpoints for inference checks,
  and record environment/version/configuration evidence. State low-resolution,
  pseudo-label, sensor-supervision, and single-trajectory limitations.
- Depth-restoration gains do not establish semantic accuracy, SLAM accuracy, or
  sim-to-real generalization. Each requires a defined task and independent evidence.

## Testing, validation, and efficient execution

- Run meaningful tests for changed contracts: split isolation, path containment,
  masks, units, timestamp ordering, duplicate policy, span/gap handling, projection,
  label preservation, checkpoint loading, and rejected quality modes.
- Compile changed Python modules and run applicable tests before committing.
  Use the full ingestion audit when data or loader behavior changes; do not repeat
  expensive decoding when existing evidence still matches unchanged inputs/code.
- Validate manifests and provenance hashes, source format checks, and report
  completeness. Export failures must not produce a success/readiness claim.
- Reuse verified caches and checkpoints. Bound CPU workers and GPU batches; avoid
  loading entire native-resolution datasets into GPU memory.
- Benchmark at the intended quality and precision. Report resolution, hardware,
  peak memory when measured, and whether timing includes preprocessing/disk I/O.
  Do not reduce fidelity silently to claim faster execution.
- Summarize findings, tests, pending issues, branch/commit state, and whether
  anything was pushed. Keep task completion separate from production readiness.

## Current project evidence and remaining gates

- Trajectory ingestion passed for all 3,088 frames / 21,616 sensor payloads.
- Training characterization covered 1,136 frames / 7,952 payloads. Native sensor
  overlap was reviewed on 16 training frames for both cameras and distortion variants.
- Eight timing/projection/split contract tests passed. IMU ordering and bracketing
  helpers are implemented; measurement interpolation and fusion are not certified.
- Exported-image distortion convention and production registration remain
  unresolved. Augmentation realism requires calibration beyond existing stress tests.
- Prior depth-training pilots and their checkpoints are experiment evidence; do
  not assume workspace experiment scripts are already integrated into this repository.
- Renderer selection/pilot, controlled real-versus-synthetic comparisons, and final
  held-out evaluation remain separate deliverables. Update this evidence section
  only when new verified results justify it.

## Local quality-review evidence

- The quality-only review passed 19 contract tests; the current local batch
  passes 47 after IMU, renderer and task-quality tests. The user's current request
  authorizes this completed batch for data-synthesis/dev; later batches require
  new publication instructions. Main integration remains separately prohibited.
- Extra annotated indexed masks use per-file instance indices. Normalize through
  their labelMapping, including values below 16; do not reuse that normalizer on
  trajectory semantic masks. All 237 extra pairs remain quarantined until source
  collection/split provenance and held-out overlap are verified.
- A bounded training-only native-resolution pilot accepted 25 variants and rejected
  11. This demonstrates export/label integrity, not physical sensor realism.
- Camera-LiDAR fusion remains disabled in configs/image_geometry.json pending
  Azure export metadata. Keep unverified geometry out of production consumers.


## IMU integration evidence

- SequenceImu is integrated into optional load_sample IMU brackets. Explicit
  sorting and positive gap tolerance are required; default loading omits IMU.
- All 6,176 frame-side queries and nine representative samples passed contract
  checks. At the explicit 20 ms inspection tolerance, both sides cover 1,136/1,136
  train, 1,035/1,037 validation and 767/915 test frames. Coverage remains partial;
  test reads are integrity inspection, never model or tolerance selection.
- Preserve invalid masks and reasons. Do not claim sorting repairs missing data,
  or interpret opaque measurement columns without verified units/frames.
- The full 21,616-payload audit passed again and manifest bytes are unchanged.
  See docs/imu_ingestion_review.json for source hashes and remaining gates.


## IMU measurement policy and metadata evidence

- configs/imu_measurements.json keeps real-data interpolation disabled. The
  alignment API is tested on known fixtures; no TorWIC unit/frame mapping has
  been certified. Never infer that mapping from the separate Mapping exporter.
- Enabling a stream requires reviewed column, SI unit, coordinate-frame, clock
  and gap-policy evidence tied to its exact source hash. Linear resampling is
  an estimate in that sensor frame, not IMU preintegration or fusion readiness.
- Metadata-key inspection covered 258 annotation JSON files and found no source
  date/trajectory fields. All 237 pairs stay quarantined. Do not treat absence of
  metadata or exact file matches as clearance of transformed held-out overlap.
- The metadata-only batch passed 37 tests. Historical 19/29-test evidence remains
  a snapshot of the earlier batches. Publication still requires user confirmation.


## Goods task and renderer evidence

- The recovered plan focuses on warehouse goods image/label generation. The first
  task uses left-camera goods segmentation (semantic 7 -> binary 1); preserve
  existing depth experiments as separate evidence. See docs/mvp_scope.txt.
- Six real training examples and metadata-only evaluation membership are prepared.
  The labels are predictions, and visual triage found likely foreground misses
  and false positives. Do not call them independent ground truth.
- The current local suite passes 47 tests; full native payload/IMU audits passed
  again. Read docs/dataset_readiness.json for exact scope and source hashes.
- The three-image Blender pilot passed analytic depth, 768 interior ray checks,
  lossless export and ZIP verification. Synthetic training eligibility stays false;
  geometric correctness does not certify natural appearance or sim-to-real gains.
- Renderer Z-pass convention must be checked, not assumed from its name. This
  fixture verifies axial depth directly. Keep depth/labels free of RGB denoising,
  fractional class IDs, geometry-blending resampling and invented sensor noise.
- Calibration declares IMU frame names/extrinsics. Their binding to exported
  measurement columns remains unverified; preserve the disabled interpolation gate.
