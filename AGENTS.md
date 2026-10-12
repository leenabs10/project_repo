# Development rules

- Perform implementation on `dev` or a feature branch based on `dev`.
- Never merge, rebase, cherry-pick, reset, or push development changes into `main`
  without the user's explicit permission for that integration. Successful tests
  and permission to work on `dev` do not authorize a merge into `main`.
- Use descriptive professional names for new modules, reports, and commits.
  Avoid calendar-week or phase-number names. Preserve historical evidence paths
  unless the user explicitly requests a migration.
- Keep raw datasets immutable. Do not commit datasets, generated reports,
  environments, secrets, or model weights; portable aggregate evidence may live
  in `docs` after review.
- Derive augmentation parameters only from confirmed training data. Keep
  validation for development comparisons and October test data out of tuning.
- Preserve units and source labels. Model predictions must never silently become
  ground truth. Identify uncalibrated noise and mask transfer as stress tests.
- Test meaningful data contracts and report limitations before claiming readiness.
