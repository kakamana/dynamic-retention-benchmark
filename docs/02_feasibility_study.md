# Feasibility Study — Dynamic Retention Benchmarking

## 1. Data feasibility
- **Primary:** synthetic org-year panel produced by `src/retention_bench/data.py` (≥ 3,200 rows × 10 columns). 8 NAICS-style sectors with believable per-sector base retention rates and per-sector action effects.
- **Real-world drop-in:** swap the generator for an exported HRIS table with the same schema; the rest of the pipeline is invariant.

## 2. Technical feasibility
- **Algorithmic shortlist**
  - Regression: Ridge per sector (main), global Ridge (baseline), LightGBM (stretch).
  - Bandit: ε-greedy (main), LinUCB (stretch), Thompson sampling (stretch).
- **Compute:** training under 5 s on 3k rows; inference < 20 ms.
- **Serving:** FastAPI + joblib (~1 MB).

## 3. Economic feasibility
| Line item | Monthly cost |
|-----------|--------------|
| 1× small container | ~$8 |
| Storage | ~$1 |
| MLflow (self-hosted) | $0 |
| **Total** | **~$9 / mo** |

**Value:** routing the next $X of HR budget to the historically highest-lift action per sector. Even a 1-point improvement on a 1,000-person org saves an order of magnitude more than the model costs.

## 4. Operational feasibility
- **Retraining:** quarterly per-sector Ridge fit; bandit updates daily as new outcomes flow in.
- **Monitoring:** RMSE drift per sector; bandit-arm coverage (alert if any (sector, action) has fewer than 30 observations).
- **Human-in-the-loop:** every recommended action requires HR-leader sign-off before being logged as "taken."

## 5. Ethical / legal feasibility
- **Confidentiality:** only sector aggregates surface in the API.
- **No individual prediction:** this project is org-level by design; no employee-level inference.
- **Disclaimer:** the API response includes the discussion-aid disclaimer.

## 6. Recommendation
**Go.** Pure synthetic data, simple stack, and the bandit gives the project a story that pure regression doesn't.
