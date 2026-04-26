# Evaluation Plan — Dynamic Retention Benchmarking

## 1. Held-out test set
20% per-sector holdout (so each sector contributes proportionally to the test set).

## 2. Primary scorecard
| Sector | RMSE (pooled) | RMSE (per-sector) | MAE (per-sector) |
|--------|---------------|-------------------|------------------|
| Health | – | – | – |
| ICT | – | – | – |
| Finance | – | – | – |
| Public | – | – | – |
| Construction | – | – | – |
| Hospitality | – | – | – |
| Education | – | – | – |
| Retail | – | – | – |

## 3. Bandit replay
- 10,000 simulated pulls on the held-out year.
- Cumulative reward vs **uniform-random** baseline (target ≥ 1.30×).
- Per-arm coverage chart — flag arms with < 30 observations.

## 4. Slice analysis
- Small (< 200 headcount) vs large (≥ 5,000 headcount) orgs.
- 2018-2020 vs 2021-2024 (cohort drift).

## 5. Latency
- `/benchmark` — p95 < 80 ms on a single instance.

## 6. Robustness
- Drop one feature at a time, re-fit, measure RMSE delta.
- Inject 5% reward noise → measure bandit-rank instability.

## 7. Deployment readiness checklist
- [ ] RMSE ≤ 0.06 across all sectors
- [ ] Worst-sector MAE ≤ 0.08
- [ ] Bandit ≥ 1.30× random on replay
- [ ] Min-n suppression (no arm shown with n < 30 in production UI)
- [ ] Model card published at `mlops/model_card.md`
