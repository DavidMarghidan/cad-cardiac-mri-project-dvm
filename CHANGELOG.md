# Changelog

All notable public-release changes are documented here.

## 1.0.0 - 2026-10-05

- Final reference execution based on 5,097 accepted targets.
- Retrained all five patient-level 2.5D Attention U-Net folds.
- Recomputed all five out-of-fold prediction parts.
- Rebuilt all eleven CPU/float32 EfficientNet-B0 representations after semantic feature-lineage invalidation.
- Regenerated nested patient-level evaluation and paired bootstrap comparisons.
- Added restart-safe initial-CPU, GPU, and final-CPU entry points.
- Added atomic idempotent writes, semantic fingerprints, safe legacy-cache adoption, per-fold recovery, status, backup, and dry-run cleanup utilities.
- Confirmed MIT licensing for original code and CC BY 4.0 for original documentation and explanatory figures.
- Added application-facing research overview, abstract poster, contribution statement, current technical summary, and citation metadata.

Earlier numerical values are superseded and should not be pooled with the 5 October 2026 result set.
