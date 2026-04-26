"""Per-sector Ridge regression + ε-greedy contextual bandit."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from .data import ACTIONS, PROCESSED, generate, write_processed
from .features import NUMERIC_FEATURES, build_pipeline

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- Ridge per sector -----------------------------


def train_per_sector_ridge(df: pd.DataFrame, alpha: float = 1.0) -> dict[str, Pipeline]:
    """Fit one Ridge per sector. The hierarchy is structural: each sector has its own
    intercept + slopes; a global Ridge would shrink across-sector signal.
    """
    out: dict[str, Pipeline] = {}
    for sector, sub in df.groupby("sector"):
        pipe = Pipeline(steps=[("pre", build_pipeline()), ("ridge", Ridge(alpha=alpha))])
        pipe.fit(sub[NUMERIC_FEATURES], sub["retention_rate"])
        out[str(sector)] = pipe
    return out


def predict_retention(models_per_sector: dict[str, Pipeline], row: dict) -> float:
    sector = row["sector"]
    if sector not in models_per_sector:
        # Average of all sector predictions if unseen sector
        preds = [m.predict(pd.DataFrame([row])[NUMERIC_FEATURES])[0] for m in models_per_sector.values()]
        return float(np.mean(preds))
    pipe = models_per_sector[sector]
    return float(pipe.predict(pd.DataFrame([row])[NUMERIC_FEATURES])[0])


# ----------------------------- ε-greedy bandit -----------------------------


@dataclass
class EpsilonGreedyBandit:
    """Contextual ε-greedy over a discrete action set, keyed by sector.

    For each (sector, action) we maintain a running mean reward and a count.
    With probability ε we explore uniformly; otherwise we exploit the arm with the
    highest mean reward observed for that sector.
    """

    actions: list[str] = field(default_factory=lambda: list(ACTIONS))
    epsilon: float = 0.10
    seed: int = 42
    means: dict[tuple[str, str], float] = field(default_factory=dict)
    counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def update(self, sector: str, action: str, reward: float) -> None:
        key = (sector, action)
        n = self.counts.get(key, 0)
        m = self.means.get(key, 0.0)
        self.counts[key] = n + 1
        self.means[key] = m + (reward - m) / (n + 1)

    def fit(self, df: pd.DataFrame) -> "EpsilonGreedyBandit":
        for _, row in df.iterrows():
            self.update(row["sector"], row["action_taken"], row["reward"])
        return self

    def rank(self, sector: str) -> list[dict]:
        scored = []
        for a in self.actions:
            scored.append(
                dict(
                    action=a,
                    mean_reward=self.means.get((sector, a), 0.0),
                    n=self.counts.get((sector, a), 0),
                )
            )
        scored.sort(key=lambda x: -x["mean_reward"])
        return scored

    def choose(self, sector: str, rng: np.random.Generator | None = None) -> str:
        rng = rng or np.random.default_rng(self.seed)
        if rng.random() < self.epsilon:
            return str(rng.choice(self.actions))
        ranked = self.rank(sector)
        return ranked[0]["action"]


# ----------------------------- Persistence -----------------------------


def _load_panel() -> pd.DataFrame:
    parquet = PROCESSED / "retention_panel.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    df = generate()
    write_processed(df)
    return df


def save(obj, name: str) -> Path:
    path = MODEL_DIR / name
    joblib.dump(obj, path)
    return path


def load(name: str):
    return joblib.load(MODEL_DIR / name)


def train_all() -> dict:
    df = _load_panel()
    ridge_per_sector = train_per_sector_ridge(df)
    bandit = EpsilonGreedyBandit().fit(df)
    save(ridge_per_sector, "ridge_per_sector.joblib")
    save(bandit, "bandit.joblib")
    return dict(
        n_rows=len(df),
        n_sectors=len(ridge_per_sector),
        bandit_arms=len(bandit.actions),
    )


if __name__ == "__main__":
    summary = train_all()
    print(summary)
