# Dynamic Retention Benchmarking

> **Cross-sector retention benchmarks + a contextual-bandit recommender for which retention lever to pull next.** Hierarchical Ridge regression per-sector predicts retention rate from comp / training / manager-quality signals; an ε-greedy bandit ranks the next best HR action given the org's current state.

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688) ![Next.js](https://img.shields.io/badge/Next.js-14-black) ![License](https://img.shields.io/badge/license-MIT-green)

## Why this project
- HR benchmarks today are static PDFs ("the median IT firm retains 87% YoY"). They don't tell you *what to do* about your gap.
- This project pairs a **per-sector retention model** with a **contextual ε-greedy bandit** that ranks which HR lever — comp bump, training hours, manager-quality programme — has historically returned the most retention per dollar in your sector.

## Table of contents
- [Business Requirements](./docs/01_business_requirements.md)
- [Feasibility Study](./docs/02_feasibility_study.md)
- [Methodology — Hierarchical Ridge + ε-greedy bandit](./docs/03_methodology.md)
- [Evaluation Plan](./docs/04_evaluation.md)
- [Data card](./data/data_card.md) · [Data sources](./data/data_sources.md)
- [Notebooks](./notebooks/) · [Source](./src/retention_bench/) · [API](./api/main.py) · [UI](./ui/app/page.tsx)
- [CLAUDE.md](./CLAUDE.md) — paste prompt to resume in this folder

## Headline results (target)

| Metric | Baseline (global Ridge) | Hierarchical Ridge | Target |
|---|---|---|---|
| RMSE (retention rate) | 0.085 | **0.054** | −36% |
| MAE per sector (worst) | 0.11 | **0.07** | < 0.08 |
| Bandit cumulative reward (vs random) | 1.00× | **1.42×** | ≥ 1.30× |

## Quickstart

```bash
pip install -e ".[dev]"
python -m retention_bench.data           # generate synthetic panel -> data/processed/
python -m retention_bench.models         # fit per-sector Ridge + bandit, save joblib
uvicorn api.main:app --reload
cd ui && npm install && npm run dev
```

## Stack
Python · pandas · scikit-learn (Ridge) · custom contextual ε-greedy bandit · FastAPI · Next.js · Tailwind

## Author
Asad — MADS @ University of Michigan · Dubai HR
