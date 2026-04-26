# Notebook 03 — Modelling

## 1. Per-sector Ridge
>>> `from retention_bench.models import train_per_sector_ridge`
>>> `models = train_per_sector_ridge(df_train)` — print each sector's R², MAE.

## 2. ε-greedy bandit
>>> `from retention_bench.models import EpsilonGreedyBandit`
>>> `bandit = EpsilonGreedyBandit().fit(df_train)`
>>> Inspect `bandit.rank("ICT")` — top arms in ICT.

## 3. Replay simulation
>>> Iterate over the held-out year; pull `bandit.choose(sector)`; compare cumulative reward vs random policy.

## 4. Persist
>>> `models.save(...)` for both objects.
