# Business Requirements — Dynamic Retention Benchmarking

## 1. Problem Statement
Today's HR retention benchmarks are static PDFs: "the median IT firm retains 87% YoY." HR leaders see the gap, but not the lever. We need a dashboard that shows where the org sits on its **sector-specific retention curve** AND ranks **which HR action has historically returned the most retention** in the same sector — comp bumps, training programmes, manager coaching, wellbeing. The bandit recommender keeps learning as more outcomes flow in.

## 2. Stakeholders
| Role | Interest | Success criterion |
|------|----------|-------------------|
| Chief People Officer | Capital allocation across HR programmes | Top-3 actions ranked by historical lift |
| HR Business Partner | Defensible argument for budget | Per-sector predicted retention with intervals |
| Finance | Cost vs return on retention spend | Clear $ delta per retained employee |
| Data Protection Officer | No org-level disclosure of competitors | Aggregated medians only |

## 3. Business Objectives
1. **RMSE ≤ 0.06** for retention-rate prediction on the held-out per-sector test split.
2. **Bandit cumulative reward ≥ 1.30×** the random-action baseline on the synthetic panel.
3. `/benchmark` p95 latency **< 80 ms** for one-org calls.
4. Top-3 action ranking returned with the n it was averaged over.

## 4. KPIs
| KPI | Definition | Target | Baseline |
|-----|-----------|--------|----------|
| RMSE retention | Held-out RMSE | ≤ 0.06 | 0.085 (global Ridge) |
| Worst-sector MAE | max sector MAE | ≤ 0.08 | 0.11 |
| Bandit cumulative reward | sum of rewards on simulated 10k pulls vs random | ≥ 1.30× | 1.00× |
| `/benchmark` p95 latency | 50 calls, single instance | < 80 ms | n/a |

## 5. Scope
**In scope:** annual org-year panel rows (`org_id, sector, year, headcount, retention_rate, comp_percentile, training_hours, manager_quality, action_taken, reward`); 8 NAICS-style sectors.
**Out of scope:** individual-employee modelling (covered by H1); cross-country wage-norm normalisation; simulation of layoffs.

## 6. Constraints & Assumptions
- **Privacy:** the panel is synthetic; in production, only sector aggregates would leave the data warehouse.
- **Compute:** CPU-only single container.
- **Action set fixed:** 5 discrete HR actions in this build; a continuous-action bandit (e.g. how-much-comp-bump) is a future extension.

## 7. Risks
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ε-greedy under-explores | Medium | Medium | LinUCB / Thompson sampling as drop-in upgrade |
| Reward proxy noisy | High | Medium | Smooth via per-(sector, action) running mean with min-n threshold |
| Cross-sector leakage | Low | Medium | Per-sector Ridge fit is structural; no global pool |

## 8. Timeline
- **Week 1** — Synthetic panel + EDA
- **Week 2** — Hierarchical Ridge fits + per-sector eval
- **Week 3** — Bandit + replay simulation
- **Week 4** — FastAPI + UI + model card
