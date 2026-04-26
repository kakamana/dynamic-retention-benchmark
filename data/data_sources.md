# Data Sources — H3 Dynamic Retention Benchmarking

## Primary
| # | Source | Use | License |
|---|--------|-----|---------|
| 1 | Synthetic generator (`src/retention_bench/data.py`) | Cross-sector mock panel | MIT |

## Secondary / reference
| Source | URL | Use |
|--------|-----|-----|
| BLS JOLTS | https://www.bls.gov/jlt/ | Sector-level quits / hires reference |
| US Census NAICS | https://www.census.gov/naics/ | Sector taxonomy |
| SHRM Annual Benchmarking | https://www.shrm.org/ | Anchor for plausible retention rates |
| Mercer / Aon comp surveys | n/a (paid) | Comp percentile shape |

## Real-world drop-in
The pipeline accepts any panel matching the schema in `data/data_card.md`. Replace the generator with your HRIS export.

## Attribution
If you reuse the schema or sector list, please cite this repo and the upstream NAICS / BLS taxonomies.
