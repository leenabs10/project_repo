# Documentation

## Read these first

| Question | Document |
| --- | --- |
| What is the project? | [Project overview](../README.md) |
| How do I run it? | [Run guide](../.tools/RUN_GUIDE.md) |
| What is the current decision? | [Current status](goods_benchmark_status.txt) |
| What happened in the latest comparison? | [Dense-layout results](dense_layout_coverage_results.txt) |
| What remains for delivery? | [Dataset release plan](goods_dataset_release_plan.txt) |
| What are the product boundaries? | [Scope](mvp_scope.txt) |
| How is the pipeline organized? | [Architecture](pipeline_architecture.txt) |
| Where should I change code? | [Source map](../src/README.md) |
| Which configuration should I use? | [Configuration map](../configs/README.md) |
| What are the engineering rules? | [Development guide](development.md), [repository policy](../AGENTS.md) |

The current status identifies the selected cohort and latest decision. Previous
results retain their own reference, configuration and execution bindings. A
preparation report is not proof that a later experiment passed acceptance.

## Previous evidence and detailed references

<details>
<summary>Open the full reference library</summary>

# Documentation index

Start with scope and architecture, then use the development guide for setup.
Evidence documents describe specific snapshots; older test counts and readiness
flags do not supersede later source-bound execution results.

## Product and development

| Document | Purpose |
| --- | --- |
| [MVP scope](mvp_scope.txt) | Product boundaries and acceptance targets |
| [Architecture](pipeline_architecture.txt) | Contracts, adapter boundaries, and target architecture |
| [Development guide](development.md) | Runtime selection, checks, code map, and configuration rules |
| [Delivery milestones](mvp_delivery_plan.txt) | Capability milestones; consult current evidence for completion |
| [Repository policy](../AGENTS.md) | Standing development, data, evaluation, and publication rules |

## Goods benchmark and synthesis

| Document | Purpose |
| --- | --- |
| [Goods status](goods_benchmark_status.txt) / [JSON](goods_benchmark_status.json) | Reference cohort, retained benchmark state, and limitations |
| [Dataset release plan](goods_dataset_release_plan.txt) | Release gates, frozen utility requirements and campaign stopping rule |
| [Release preflight](goods_release_preflight.txt) / [JSON](goods_release_preflight.json) | Repeatability diagnostics, opt-in numerical reduction and current training coverage |
| [Corrected-reference control](goods_corrected_reference_control.txt) / [JSON](goods_corrected_reference_control.json) | Approved training-reference revision, fresh control and completed grouped-cargo pass/fail comparison |
| [Dense-layout results](dense_layout_coverage_results.txt) | Final bounded arrangement revision, fresh three-seed comparison and complete utility decision |
| [Representation assessment](warehouse_representation_assessment.txt) | Training-only content gaps, renderer limits and prerequisites for a new approach |
| [Package asset qualification](package_asset_qualification.txt) | Additional carton/container sources, corrected visible semantics and six-view quality scout |
| [Scene collections](scene_collection_qualification.txt) | Source-bound multi-job input, acceptance checks and remaining diversity limits |
| [Wrapping feasibility](wrapping_feasibility.txt) | Proposed visibility, ownership and depth fixtures; wrapping remains unsupported |
| [Grouped-cargo results](grouped_cargo_coverage_results.txt) | First bounded grouped-cargo comparison; candidate rejected |
| [Scene generation](scene_generation.txt) | Backend interface and qualification evidence |
| [Equipment protocol](equipment_pair_protocol.txt) / [results](equipment_pair_results.txt) | Completed empty/loaded comparison; candidate rejected |
| [Equipment precision experiment](equipment_precision_experiment.txt) | Frozen equipment precision protocol, source bindings and utility/recall gates |
| [Equipment precision results](equipment_precision_results.txt) | Completed fresh matched comparison; IoU/precision gains with recall regressions, candidate rejected |
| [Equipment precision failure analysis](equipment_precision_failure_analysis.txt) | Earlier native regression evidence and the hypothesis that led to the completed grouped-cargo experiment |
| [Reference reassessment](goods_reference_reassessment.txt) | Validated person correction and rescored unchanged predictions |
| [Occupied-rack preparation](occupied_rack_context_preparation.txt) | Historical preparation snapshot: cargo placement, geometry checks and exposure tradeoff |
| [Occupied-rack results](occupied_rack_context_results.txt) | Completed matched comparison, every seed/frame change and complete utility decision |
| [Reviewed training coverage](reviewed_training_coverage_results.txt) | Expanded real cohort comparison and its reference-version limitations |
| [ResNet backbone results](resnet_backbone_results.txt) | Matched three-seed backbone experiment; retained model unchanged |
| [Additional model trial](additional_model_trial_results.txt) | Exploratory VGG-U-Net and YOLO semantic comparison, setup and limitations |
| [Threshold diagnostic](goods_threshold_diagnostic_results.txt) | Later exploratory inference tradeoffs; no model promotion |
| [Failure analysis](goods_failure_analysis.txt) | Saved errors and actual training exposure |
| [Scene benchmark results](scene_benchmark_results.txt) | Earlier matched scene comparisons |
| [Sampling results](sampling_benchmark_results.txt) | Balanced donor-schedule ablation |
| [Supported-scene protocol](supported_scene_protocol.txt) | Training-supported generation constraints |
| [Contact review](scene_contact_review.txt) / [tyre appearance](tyre_appearance_qualification.txt) | Bounded appearance and geometry investigations |

Historical experiment rules remain fixed. Reconcile the scope improvement target
with each future protocol explicitly, without rewriting past decisions. Repeated
development use does not establish independent generalization. The locked test
is not a tuning resource.

## Data foundation and sensor evidence

- [Dataset notes](dataset_notes.txt), [characterization](dataset_characterization.txt),
  and [ingestion validation](ingestion_validation.json).
- [Data quality](data_quality_review.txt) and [dataset readiness](dataset_readiness.txt).
- [Depth-cache verification](depth_cache_verification.txt): source derivatives,
  masks, units, and historical depth benchmark evidence.
- [Sensor metadata](sensor_metadata_review.txt), [IMU review](imu_ingestion_review.txt),
  and [sensor conditions](sensor_condition_generation.txt): verified conventions and limits.
- [Preparation readiness](preparation_readiness.txt) and
  [adapter foundation](adapter_foundation.json): source-bound preparation snapshots.

## Supporting implementation records

[Technology stack](tech_stack.txt), [pipeline integration](pipeline_integration.txt),
and [development workflow](development_workflow.txt) retain detailed implementation
and historical verification records. Use the development guide as the setup entry
point. Paired JSON files retain machine-readable bindings and should not be removed
merely because a prose summary exists.

Generated galleries, local review links, and operator instructions stay in ignored
`reports/`. Its local index provides navigation without moving historical runs.

</details>
