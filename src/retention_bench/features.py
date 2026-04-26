"""Feature pipeline for the retention panel."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = ["headcount", "comp_percentile", "training_hours", "manager_quality"]


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add a log-headcount column. The Ridge fits use the raw NUMERIC_FEATURES list,
    but downstream EDA + per-sector exploratory plots benefit from the log scale.
    """
    out = df.copy()
    out["log_headcount"] = np.log1p(out["headcount"].clip(lower=1))
    return out


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), NUMERIC_FEATURES)],
        remainder="drop",
    )
    return Pipeline(steps=[("pre", pre)])
