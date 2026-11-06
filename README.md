# DubaiDATA Synthetic Data MVP

A customer-adaptable pipeline for profiling real sensor data, generating
controlled synthetic variations, validating their quality and measuring task
usefulness. Warehouse goods segmentation on TorWIC is the reference application.

## Run

```powershell
.\mvp.cmd
```

The menu opens results, scenes, predictions and saved annotations, or runs checks.
Read [the run guide](docs/run.md) for the short command list.

## Read

| Need | Document |
| --- | --- |
| Current result and remaining gates | [Status](docs/status.md) |
| Product boundaries | [Scope](docs/scope.md) |
| Components and data flow | [Architecture](docs/architecture.md) |
| Environment and verification | [Development](docs/development.md) |
| Main code entry points | [Source map](src/README.md) |
| Configuration selection | [Configuration guide](configs/README.md) |
| Previous evidence | [Research archive](archive/README.md) |

## Layout

```text
src/           maintained pipeline and benchmark code
tests/         current software checks and fixtures
configs/       portable specifications and ignored local setup
scripts/       daily launcher and diagnostics
requirements/  pinned environment specifications
docs/          concise active documentation
archive/       historical summaries and experiment configurations
```

Datasets, assets, reports, models and old local operators are physically outside
the repository, in the workspace selected by `configs/local.json`. A new checkout
needs that local configuration and the required data artifacts; they are not in Git.
See `configs/local.example.json`.

The current two-candidate campaign is closed after both candidates failed the full
utility criteria. No model was promoted and the locked test remains unevaluated.
Qualified simulated LiDAR and broader domain fidelity remain open.

Develop on `dev`. Follow [AGENTS.md](AGENTS.md); publishing and main integration
require explicit authorization.
