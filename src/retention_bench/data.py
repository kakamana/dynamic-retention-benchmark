"""Synthetic cross-sector retention panel.

Generates ≥ 3,000 organisation-year rows across 8 NAICS-style sectors with believable
correlations between compensation percentile, training hours, manager-quality score,
and observed retention rate. Each row also records a randomly-taken HR action and the
associated reward, used to bootstrap the contextual bandit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED = DATA_DIR / "processed"

SECTORS = [
    "Health", "ICT", "Finance", "Public", "Construction",
    "Hospitality", "Education", "Retail",
]
# Per-sector intercept on retention rate (rough: regulated sectors retain better)
SECTOR_BASE = {
    "Health": 0.86, "ICT": 0.78, "Finance": 0.84, "Public": 0.92,
    "Construction": 0.74, "Hospitality": 0.68, "Education": 0.88, "Retail": 0.66,
}
ACTIONS = ["comp_bump", "training_program", "manager_coaching", "wellbeing_program", "no_action"]
# Sector-specific effectiveness multipliers (chosen so that no single action dominates everywhere)
ACTION_EFFECT = {
    "Health":      {"comp_bump": 0.020, "training_program": 0.030, "manager_coaching": 0.040, "wellbeing_program": 0.025, "no_action": 0.000},
    "ICT":         {"comp_bump": 0.045, "training_program": 0.030, "manager_coaching": 0.020, "wellbeing_program": 0.015, "no_action": 0.000},
    "Finance":     {"comp_bump": 0.040, "training_program": 0.020, "manager_coaching": 0.030, "wellbeing_program": 0.015, "no_action": 0.000},
    "Public":      {"comp_bump": 0.005, "training_program": 0.025, "manager_coaching": 0.030, "wellbeing_program": 0.020, "no_action": 0.000},
    "Construction":{"comp_bump": 0.030, "training_program": 0.015, "manager_coaching": 0.025, "wellbeing_program": 0.030, "no_action": 0.000},
    "Hospitality": {"comp_bump": 0.035, "training_program": 0.020, "manager_coaching": 0.025, "wellbeing_program": 0.030, "no_action": 0.000},
    "Education":   {"comp_bump": 0.020, "training_program": 0.025, "manager_coaching": 0.035, "wellbeing_program": 0.020, "no_action": 0.000},
    "Retail":      {"comp_bump": 0.030, "training_program": 0.018, "manager_coaching": 0.022, "wellbeing_program": 0.025, "no_action": 0.000},
}


def generate(n_rows: int = 3200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        sector = rng.choice(SECTORS)
        year = int(rng.integers(2018, 2025))
        headcount = int(np.clip(rng.lognormal(6.5, 0.8), 50, 50_000))
        comp_percentile = float(np.clip(rng.normal(50, 18), 5, 95))
        training_hours = float(np.clip(rng.normal(28, 12), 0, 120))
        manager_quality = float(np.clip(rng.normal(3.4, 0.7), 1.0, 5.0))

        action = str(rng.choice(ACTIONS, p=[0.20, 0.20, 0.20, 0.15, 0.25]))

        # Underlying retention model (linear with mild noise + action effect)
        base = SECTOR_BASE[sector]
        contribution = (
            0.0008 * (comp_percentile - 50)
            + 0.0012 * (training_hours - 28)
            + 0.020 * (manager_quality - 3.4)
        )
        action_lift = ACTION_EFFECT[sector][action]
        noise = float(rng.normal(0, 0.025))
        retention_rate = float(np.clip(base + contribution + action_lift + noise, 0.30, 0.99))

        # Reward = lift attributable to the action vs no_action baseline (with noise)
        reward = float(action_lift + rng.normal(0, 0.005))

        rows.append(
            dict(
                org_id=f"O{i:05d}",
                sector=sector,
                year=year,
                headcount=headcount,
                retention_rate=retention_rate,
                comp_percentile=comp_percentile,
                training_hours=training_hours,
                manager_quality=manager_quality,
                action_taken=action,
                reward=reward,
            )
        )
    return pd.DataFrame(rows)


def write_processed(df: pd.DataFrame) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "retention_panel.parquet"
    df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    df = generate()
    out = write_processed(df)
    print(f"wrote {len(df):,} rows -> {out}")
