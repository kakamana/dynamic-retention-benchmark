# Notebook 01 — EDA: Cross-sector retention panel

>>> `from retention_bench.data import generate; df = generate()`

## 1. Per-sector distributions
- Median + IQR retention by sector.
- Boxplot of `retention_rate` ~ sector.

## 2. Feature correlations
- Pearson and Spearman of `comp_percentile`, `training_hours`, `manager_quality` vs retention.
- Per sector — confirm the slope direction is consistent.

## 3. Action coverage
- Counts of `(sector, action_taken)`. Flag arms with n < 30.

## 4. Reward sanity
- Mean reward per (sector, action) — does the leaderboard match `ACTION_EFFECT` priors?
