# Run the MVP

Use the VS Code PowerShell terminal in the project folder.

```powershell
.\mvp.cmd
```

Choose a number from the menu. Python activation and server setup are handled
by the launcher using `configs/local.json`.

| Task | Direct command | Expected result |
| --- | --- | --- |
| Current decision | `.\mvp.cmd status` | Prints the concise status |
| Results workspace | `.\mvp.cmd reports` | Opens the current evidence page |
| Synthetic scenes | `.\mvp.cmd scenes` | Opens the reviewed scene gallery |
| Prediction comparison | `.\mvp.cmd predictions` | Opens metrics and prediction images |
| Saved annotations | `.\mvp.cmd annotations` | Opens the selected saved review service |
| Environment check | `.\mvp.cmd check` | Reports interpreter, packages and GPU availability |
| Software tests | `.\mvp.cmd test` | Runs the current fixture/software suite |
| Stop report service | `.\mvp.cmd stop` | Stops only the launcher's recorded report process |

Report pages run in the background. The annotation service stays in its terminal;
open the printed address and use Ctrl+C in that terminal when finished. Saved
reviews remain on disk. Reopening does not require repeating completed annotation.

## Show progress

1. Open `mvp.cmd scenes` for generated RGB and labels.
2. Open `mvp.cmd predictions` for matched real-image results.
3. Open `mvp.cmd reports` for complete evidence and the selected reference manifest.

The latest candidates failed utility gates. Appearance acceptance does not imply
task improvement; show both gains and regressions.

Report and scene selections come from `report_workspace` in ignored
`configs/local.json`; see `configs/README.md`. A new checkout has no implicit
experiment selection. Reopening the report workspace refreshes its navigation
without restarting a server belonging to this checkout.

## New jobs

A new generation/training run needs an explicit supported specification, qualified
inputs, frozen settings and a fresh workspace output directory. The launcher does
not automatically restart a completed research campaign.

Inspect a maintained interface with the configured interpreter:

```powershell
$pythonExe = (Get-Content 'configs\local.json' -Raw | ConvertFrom-Json).python_executable
& $pythonExe -B -m src.pipeline.scene --help
```

Other interfaces: `src.pipeline.run`, `src.validation.scene_quality`,
`src.validation.scene_admission`, `src.training.goods_benchmark` and
`src.pipeline.scene_benchmark`. Read `configs/README.md` before selecting inputs.

## Troubleshoot

| Problem | Action |
| --- | --- |
| Runtime unavailable | Check `configs/local.json`, then run `mvp.cmd check` |
| Browser cannot connect | Rerun `mvp.cmd reports`; inspect the workspace `services` logs |
| Port belongs to another service | Leave it intact and inspect that service first |
| Saved reference or report missing | Check the configured workspace; restore the exact artifact |
| A historical absolute path no longer exists | Consult `archive/relocation.json` and the workspace migration ledger |
| Tests fail | Investigate the failing current check; old saved counts do not supersede it |

Closing a browser tab does not stop a service. Do not stop every Python process.
