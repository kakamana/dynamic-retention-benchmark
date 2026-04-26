# Data Card — H3 Dynamic Retention Benchmarking

## Dataset composition

| Layer | Source | Rows × cols | Purpose |
|-------|--------|-------------|---------|
| Synthetic org-year panel | `src/retention_bench/data.py` | ≥ 3,200 × 10 | Cross-sector retention training + bandit bootstrap |

## Fields

| Column | Type | Description |
|--------|------|-------------|
| `org_id` | str | Pseudonymous organisation id |
| `sector` | enum | One of 8 NAICS-style sectors |
| `year` | int | 2018–2024 |
| `headcount` | int | log-normal sampled (50 to ~50k) |
| `retention_rate` | float | Annual retention rate (0..1) |
| `comp_percentile` | float | 0..100 vs sector |
| `training_hours` | float | per employee per year |
| `manager_quality` | float | 1..5 composite score |
| `action_taken` | enum | one of 5 HR actions |
| `reward` | float | observed retention lift attributable to action_taken |

## Known limitations
- Each sector's base retention rate is hard-coded; in real life this drifts year-over-year.
- The action-effect map is fixed (no interaction with comp/training); a richer simulator would model interactions.
- Reward is generated as `action_effect + noise`, not from an actual organisational experiment.

## PII
None. `org_id` is a synthetic counter.

## Splits
- 80% train · 20% test, stratified per-sector.

## Reproducing
```bash
python -m retention_bench.data
# wrote 3,200 rows -> data/processed/retention_panel.parquet
```
Deterministic seed = 42.

## Licensing
Synthetic panel + code: MIT.
