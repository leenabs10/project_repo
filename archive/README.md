# Research archive

Historical evidence is retained without occupying the active documentation and
configuration folders.

| Folder | Contents |
| --- | --- |
| `docs/` | Original summaries, original index and machine-readable bindings |
| `configs/` | Original experiment, synthesis and renderer configurations |
| `relocation.json` | Mapping from original repository paths to their new locations |

Files in this archive retain their original bytes and names. Values and readiness
claims apply to their recorded cohort, configuration, code and execution versions.
They do not supersede `docs/status.md`.

Large local artifacts are in the external workspace selected by `configs/local.json`.
Its `archive/tools` preserves completed local drivers and old runtime records;
its `migration/source_snapshot` preserves source bytes before the layout update.
The full move ledger and checksums are in the workspace `migration` directory.

Historical absolute path strings are intentionally preserved as evidence. To
reproduce an old run, resolve them through the relocation map in a fresh output
context and use its recorded code/runtime. Do not execute old fixed-output drivers
as if they were the current workflow.
