# MVP scope

Prove a customer-conditioned synthetic-data workflow using TorWIC-SLAM as the
reference customer: ingest, understand coverage, define controlled variations,
generate supported data, validate it, export lineage and measure downstream use.

## Included

- Fixed TorWIC RGB, measured depth, real LiDAR, calibration, pose/timing and available semantics.
- Dataset adapters, canonical records, profiling and source/version bindings.
- Bounded sensor-condition variations and a limited warehouse scene family.
- Goods segmentation as the primary utility benchmark; depth restoration is conditional.
- Exact-batch appearance/label qualification, admission and matched evaluation.
- Reproducible configuration, accepted data, quality/coverage reports and demo evidence.

## Acceptance

Quality, geometry, labels, appearance and task utility are separate gates. The
current experiment requires its complete frozen criteria, including IoU gain,
precision, recall preservation and condition-specific safeguards. A passing
generation job or a similar aggregate score does not establish usefulness.

Training, development and test are separated by whole trajectory. The locked
test is reserved for final evaluation after selection is frozen. The original
same-scene RGB/depth/LiDAR goal remains open until simulated LiDAR is implemented
and qualified; current camera-only rendering does not close that requirement.

## Outside this implementation

No full robot navigation/control, safety certification, automatic support for
arbitrary customer formats, full digital twin or production multi-tenant portal.
Isaac Sim and cloud execution remain candidates selected by requirements.

Use [current status](status.md) for completion and unresolved gates. Historical
scope and experiment specifications remain in `archive/docs/` unchanged.
