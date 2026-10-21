# MVP development instructions

These rules apply to the repository and its subdirectories. Follow the user's
current instructions and the verified MVP scope. Distinguish completed evidence
from proposed work; do not invent requirements from an unavailable planning document.

## Product architecture and boundaries

- DubaiDATA builds a customer-adaptable synthetic-data pipeline. Its three
  principles are: dataset-independent core, simulator-independent generation
  contracts, and infrastructure-independent execution. See
  docs/pipeline_architecture.txt and docs/mvp_scope.txt for the current design.
- TorWIC, its numeric formats/splits/frames below, and goods segmentation are
  the first adapter/application. Do not hardcode them into new shared contracts.
  Dataset independence requires explicit new adapters, not guessed source formats.
- Keep customer controls, canonical metadata, coverage, provenance, quality and
  evaluation outside dataset-specific and simulator-specific implementations.
  Required modality/fidelity support must be checked before job execution.
- Simulator workers use separate pinned environments and provider-neutral jobs.
  Separate generation backend, execution runner and storage adapter decisions.
  Personal paths, cloud credentials and a mandatory provider do not belong in
  shared customer specifications. Reject unsupported requests rather than invent outputs.
- BlenderProc is the current intended backend; retain the existing Blender pilot.
  Isaac Sim is a future candidate selected when requirements justify engineering
  and validation. GCP is an optional runner choice, not a production commitment.
- Preserve portable scene/asset/sensor intent. New backends still need conversion,
  known-fixture geometry/sensor checks and appearance review; do not promise
  automatic migration, equivalent materials or pixel-identical results.
- Goods segmentation is the primary benchmark; depth is a conditional bounded
  benchmark. Simulated LiDAR remains deferred, and the same-scene RGB/depth/LiDAR
  generation gate remains open. A camera pilot does not establish full completion.
- Stakeholder-facing scope, architecture, plans, status documents and README
  must use capabilities, milestones and acceptance criteria. Keep calendar dates,
  schedule labels, deadlines, private scheduling, chat history and rollback/publishing
  housekeeping out of these documents. Internal planning remains private. Preserve
  source-capture timestamps and original audit/provenance records where required;
  do not alter datasets, historical hashes or technical evidence to polish prose.
- The architecture document is a target design. Do not claim core, backend or
  executor capabilities are implemented until verified by the relevant tests.
  Reuse working code incrementally and preserve evidence snapshots.

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
  or numbered phase names for new code, reports, and commits.
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

## Human-review communication

- Keep stakeholder-facing documents professional: describe capabilities, results,
  qualification status and acceptance criteria. Do not include human-review headings,
  reviewer checklists, manual operator instructions or feedback-file procedures.
  Keep operational follow-up in ignored local reports and private communication.
- Collect actionable human-evaluation items and artifact links at the end of the
  task batch when needed in the conversation, not in stakeholder documentation. Continue independent engineering while those reviews remain pending.
  Do not treat silence, automated checks or a prepared gallery as human approval.

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

- Data quality is the first priority. Throughput, volume and delivery pressure
  must not bypass source integrity, label correctness, geometry, fidelity or
  applicable human-review gates. Reject/quarantine failures and retain reasons.
- Scale only within quality-validated settings. Measure accepted samples per
  unit compute, rejection rate, coverage, memory/storage usage and task utility;
  raw output counts or plausible images alone do not establish success.
- Use bounded-memory processing, partitioned jobs and retry-safe artifacts as
  workloads grow. Preserve sample IDs, parameter/seed binding, provenance and
  acceptance semantics under parallel execution. Never silently lower fidelity.
- Treat scalability and customer value as hypotheses until demonstrated by
  representative workloads and independent evaluation. Architecture alone is
  not proof of production readiness or guaranteed model improvement.
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
- Historical depth experiments are now integrated as import-safe project CLIs.
  Keep their checkpoints/results separate from goods-segmentation evidence; see
  docs/pipeline_integration.txt for environment, migration and verification.
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
- The renderer batch passed 47 tests; full native payload/IMU audits passed
  again. Read docs/dataset_readiness.json for exact scope and source hashes.
- The three-image Blender pilot passed analytic depth, 768 interior ray checks,
  lossless export and ZIP verification. Synthetic training eligibility stays false;
  geometric correctness does not certify natural appearance or sim-to-real gains.
- Renderer Z-pass convention must be checked, not assumed from its name. This
  fixture verifies axial depth directly. Keep depth/labels free of RGB denoising,
  fractional class IDs, geometry-blending resampling and invented sensor noise.
- Calibration declares IMU frame names/extrinsics. Their binding to exported
  measurement columns remains unverified; preserve the disabled interpolation gate.


## Organized training pipeline evidence

- Research pipeline integration is verified separately from the
  earlier data-understanding milestone. Ten legacy depth scripts are integrated,
  import-safe and configurable; use module CLIs from the project root.
- Use the CUDA interpreter from ignored .tools/training_runtime.json for training
  and the contract suite. The organization snapshot passed 55 tests; cache
  verification extends the suite to 62. Do not assume the ingestion .venv has
  training dependencies.
- Models/checkpoints and caches are ignored local artifacts. Thirteen original
  checkpoints retain their hashes; eighteen model/preprocessing components retain
  executable AST. Do not rewrite historical result hashes to imply a new experiment.
- New runs must preserve raw/input/previous results. Quality warm-start paths are
  explicit. Clean inference preserves observations, and denoising stays gated.
- Ten outside-cwd CLI checks, all checkpoint forwards, three discarded optimizer
  steps, two prediction CLIs and the paired audit passed. No full retraining or
  October model evaluation occurred. Publication requires the user's explicit
  authorization for the current batch; future batches need new instructions.

## Verified adapter preparation foundation

- Canonical sample/cache-job contracts, TorWIC and declared paired-PNG adapters,
  shared cache preparation and a local subprocess executor are implemented.
  82 tests passed; the complete June 320x180 cache has identical NPZ hashes.
- Keep the legacy research cache format explicit; canonical caches preserve raw
  values and source units plus float32 metres/valid masks. Do not silently quantize
  another customer's depth scale into millimetres. Resizing updates camera metadata.
- The source loader's package import was corrected with identical decoder function/
  class AST. Preserve raw inputs, prior caches/checkpoints and historical hash records.
- The host package is installable and includes schema resources. Local worker jobs
  do not accept shell commands or machine paths. Deployment configuration owns them.
- Generation entries are capability declarations with unavailable common workers;
  do not claim BlenderProc/Isaac integration, simulated LiDAR, remote execution,
  reviewed goods training, calibrated realism or production scalability from these
  preparation checks. No new rendering or training is part of this batch.

## Controlled real-sample synthesis evidence

- Native RGB gain and training-only depth-mask jobs run through the fixed local
  executor with portable controls and evidence hashes. Shared quality checks live
  outside the generation adapter. 98 tests passed, including a different customer
  format and stable partitioned execution. Cross-split payload paths are rejected.
- The six-source pilot passed 11 of 18 variants through automated checks and
  rejected 7. Repeated data/review bytes and decisions match; sources are unchanged.
  LiDAR/labels are preserved byte-for-byte. See docs/sensor_condition_evidence.json.
- All variants remain training-ineligible; human appearance and independent goods
  references are pending. Missing coverage remains explicit. Do not relax bounds
  to pass rejected cases or call gain/dropout calibrated sensor simulation.

## Verified benchmark preparation

- The goods task adapter, native reference workspace, corrected-mask importer and
  snapshot-bound admission workflow are verified by 115 tests. Runtime fixture
  update/reload passed at 640x360, batch size 4, float32. No trained candidate was saved.
- Reference datasets require separate semantic qualification. Preserve their
  source and submission hashes; label corrections create new versions and require
  every compared model to use the same targets. Technical import is not proof of
  annotation accuracy or customer-model fitness.
- Use the snapshot-bound feedback template from the current preparation workflow;
  the older five-column gallery CSV cannot bind decisions to a complete batch version.
- Reference masks are model-assisted corrections, with potential anchoring bias.
  Configured cohort counts are pilot requirements, not statistical adequacy guarantees.
  No benchmark score, model improvement or simulator integration follows from runtime checks.
- Preserve frozen source/results and publish only on the user's new explicit request.

## Verified reference and real-only execution tools

- This software release contains native annotation editing, exact submission
  preservation, provenance-aware admission and a real-only training/evaluation
  engine. Keep experimental training extensions and dataset-specific outcomes
  separate from software contract verification.
- Fixture training, strict checkpoint reload and native evaluation validate the
  implementation. They do not qualify real masks, certify accuracy or establish
  synthetic-data utility. Never edit prediction outputs to improve scores.
- Preserve all local experiment and annotation histories. Label revisions need
  a new source-bound version; compare every candidate against the same targets.
- Keep stakeholder wording professional and technical authorship truthful.
  Publication of this selected subset does not authorize a future push or any
  integration into main. Preserve uncommitted working changes.
