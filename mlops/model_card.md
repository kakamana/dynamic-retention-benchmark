# Model Card — Hierarchical Ridge + ε-greedy Bandit

## Intended use
Decision-aid for HR leaders deciding which retention lever to fund next, by sector. Never used as automated employment-decision input.

## Training data
Synthetic cross-sector org-year panel (≥ 3,200 rows × 10 columns). See `data/data_card.md`.

## Model family
- **Predict** — one Ridge regression per sector (`alpha = 1.0`).
- **Recommend** — ε-greedy contextual bandit over 5 HR actions, keyed by sector, ε = 0.10.

## Metrics (held-out test, to be filled)
| Metric | Target |
|--------|--------|
| RMSE | ≤ 0.06 |
| Worst-sector MAE | ≤ 0.08 |
| Bandit cumulative reward (vs random) | ≥ 1.30× |
| `/benchmark` p95 latency | < 80 ms |

## Limitations
- Synthetic action effects; real organisational outcomes are noisier and confounded.
- ε-greedy under-explores once arm means stabilise; LinUCB / Thompson sampling is the upgrade.
- No interaction terms in the Ridge fit (e.g. comp × sector × tenure).

## Ethical considerations
- Aggregate-only outputs.
- Disclaimer in every API response.
- Min-n suppression: arms with n < 30 should not be surfaced as "best."

## Retraining
- Per-sector Ridge — quarterly.
- Bandit — online, daily as new outcomes flow in.

## Ownership
- On-call DS: Asad
- Runbook: `mlops/runbook.md` (TBD)
