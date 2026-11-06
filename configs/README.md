# Configuration guide

| Location | Purpose |
| --- | --- |
| `local.example.json` | Machine setup example; copy to ignored `local.json` and set absolute paths |
| `jobs/` | Supported cache and sensor-condition specifications |
| `runtime/` | Deployment examples for supported workers |
| `tasks/` | Goods taxonomy and reference requirements |
| `../archive/configs/` | Original experiment and renderer recipes, preserved byte-for-byte |

The external workspace holds dataset payloads, assets, models and completed run
protocols. Current evidence is linked from `docs/status.md` and the report launcher.

Before a new job, select the intended source, task, reference, backend and runtime
versions. Allocate a fresh output path. Do not edit a past recipe or substitute
a similarly named configuration to silence a checksum failure.

## Local report selection

`report_workspace` in ignored `local.json` selects the report server port, scene
page, prediction page, qualification summary and dashboard links. All paths are
relative POSIX paths under the workspace's `reports/` directory. Selected files
must exist; absolute paths, escaping paths and unsupported fields are rejected.
The maintained launcher and HTML template contain no experiment-specific paths.

```json
{
  "port": 8810,
  "scenes": "accepted-scenes/index.html",
  "predictions": "comparison/report/index.html",
  "summary": "Describe the qualification state of these exact artifacts.",
  "links": [
    {
      "label": "Comparison results",
      "path": "comparison/report/index.html",
      "description": "Matched results, gains and regressions."
    }
  ]
}
```

This fragment belongs inside `report_workspace`; it does not select training
inputs or approve annotations. `mvp.cmd reports` refreshes generated navigation
even when reusing its verified local server. Other frozen reports remain intact.

## Explicit real and synthetic auxiliary inputs

Goods training supports `explicit_mixed_source_sequence` within an auxiliary-only
scene configuration. Each scheduled event declares its source as
`real_training_reference` or `admitted_scene`, along with the sample ID, update
and verified target kind. Both source types must appear in this policy.

```json
{
  "strategy": "explicit_mixed_source_sequence",
  "events": [
    {
      "update": 2,
      "sample_id": "training-reference-A",
      "target_kind": "background_only",
      "input_source": "real_training_reference"
    },
    {
      "update": 4,
      "sample_id": "accepted-scene-B",
      "target_kind": "mixed",
      "input_source": "admitted_scene"
    }
  ]
}
```

This is the `auxiliary_sampling` fragment of a complete configuration. Its events
must cover every declared auxiliary update. Real inputs must belong to the
qualified training cohort; synthetic inputs require completed admission. RGB and
target stay paired within their own source root. Native categorical masks are
validated before resizing, and ignored ID 255 remains excluded from supervision.

Saved mixed-source runs include `auxiliary_events.json` with executed source and
RGB/target hashes. Exposure reports count real replay as real and disclose changes
in total real versus synthetic presentations. Existing sampling policies retain
their original behaviour. Input support alone does not establish synthetic utility.
