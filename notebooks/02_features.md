# Notebook 02 — Featurization

>>> `from retention_bench.features import build_pipeline`

## 1. Standardise numerics
- Confirm scaler stats stable across sectors (each Ridge is fit on its own sector subset, but the feature shapes match).

## 2. Optional log-transform
- `headcount` is heavy-tailed; consider log-transform if the worst-sector RMSE doesn't hit target.
