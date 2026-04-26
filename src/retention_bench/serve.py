"""Inference helpers used by the FastAPI layer."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import models
from .data import PROCESSED, generate, write_processed


@lru_cache(maxsize=1)
def _artifacts():
    try:
        ridge = models.load("ridge_per_sector.joblib")
        bandit = models.load("bandit.joblib")
    except FileNotFoundError:
        ridge, bandit = None, None
    return dict(ridge=ridge, bandit=bandit)


@lru_cache(maxsize=1)
def _panel() -> pd.DataFrame:
    parquet = PROCESSED / "retention_panel.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    df = generate()
    write_processed(df)
    return df


def benchmark(org_metrics: dict) -> dict:
    panel = _panel()
    sector = org_metrics["sector"]
    sector_panel = panel[panel["sector"] == sector]
    if sector_panel.empty:
        return dict(
            sector=sector,
            sector_median=0.0,
            gap=0.0,
            percentile_rank=0.0,
            predicted_retention=None,
            top_actions=[],
            note="unknown sector",
        )

    sector_median = float(sector_panel["retention_rate"].median())
    own_rate = float(org_metrics.get("retention_rate", sector_median))
    gap = own_rate - sector_median
    pct_rank = float((sector_panel["retention_rate"] < own_rate).mean())

    art = _artifacts()
    predicted = None
    if art["ridge"] is not None:
        from .models import predict_retention

        predicted = predict_retention(art["ridge"], org_metrics)

    if art["bandit"] is not None:
        ranked = art["bandit"].rank(sector)[:3]
    else:
        # Wiring-only fallback
        ranked = [
            dict(action="manager_coaching", mean_reward=0.030, n=0),
            dict(action="training_program", mean_reward=0.025, n=0),
            dict(action="comp_bump",       mean_reward=0.020, n=0),
        ]

    return dict(
        sector=sector,
        sector_median=sector_median,
        gap=gap,
        percentile_rank=pct_rank,
        predicted_retention=predicted,
        top_actions=ranked,
    )
