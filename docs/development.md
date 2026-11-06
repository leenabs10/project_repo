# Development guide

## Environment

Use the configured Python 3.12 environment and existing pinned requirements.
`configs/local.json` contains `workspace_root` and `python_executable`; it does
not belong in Git. Copy the example and select valid absolute paths on a new machine.
Do not move or reinstall a working environment just to reorganize the repository.

```powershell
.\mvp.cmd check
.\mvp.cmd test
```

The check reports package and GPU availability. Tests include small fixtures,
not a new customer experiment. Check the relevant changed contracts first.

## Paths and artifacts

Keep executable paths, storage roots and local review selection in local
configuration. Portable specifications stay under `configs/jobs`, `configs/runtime`
and `configs/tasks`. Historical recipes are in `archive/configs/`; several remain
fixture inputs and must not be treated as interchangeable current defaults.

All generated outputs belong in the external workspace. Protect both legacy and
relocated raw-data paths. Preserve original source records and source hashes.
Relative payload paths within frozen report directories remain unchanged.

## Source and verification

Follow `src/README.md` to locate the maintained components. Keep public imports
compatible, place execution behind entry points and document units, shapes,
frames, masks and failure behavior. Do not import training merely to read metadata.

For path or contract changes, run relevant fixtures and the full current suite.
Compile changed modules and review whitespace. Documentation-only changes do
not require retraining. Historical source hashes remain bound to their original
snapshot; a migration is a new code version, not a retroactive experiment change.

## Experiments and publication

Freeze references, task, configuration, initialization, source versions, runtime
and ordered sampling before fitting. Use fresh outputs. Qualification applies to
the exact exported batch. The closed campaign remains rejected until a separately
declared experiment supplies new evidence. Preserve uncertainty and review provenance.

Develop on `dev`. Preserve unrelated local changes. Review explicit diffs before
committing. Push only when requested; integration into `main` requires separate
explicit approval. Read `AGENTS.md` for the standing policy.
