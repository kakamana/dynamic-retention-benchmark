"""FastAPI for Dynamic Retention Benchmarking.

Endpoints:
    GET  /health
    POST /benchmark            - org metrics → sector_median, gap, percentile_rank, top_actions
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Dynamic Retention Benchmarking - HR", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = (
    "Bandit recommendations reflect historical lift in the synthetic panel. "
    "Treat as a discussion aid, not a guaranteed retention outcome."
)


class OrgMetrics(BaseModel):
    sector: Literal[
        "Health", "ICT", "Finance", "Public", "Construction",
        "Hospitality", "Education", "Retail",
    ]
    headcount: int = Field(ge=10)
    comp_percentile: float = Field(ge=0, le=100)
    training_hours: float = Field(ge=0, le=400)
    manager_quality: float = Field(ge=1.0, le=5.0)
    retention_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class ActionRanking(BaseModel):
    action: str
    mean_reward: float
    n: int


class BenchmarkResponse(BaseModel):
    sector: str
    sector_median: float
    gap: float
    percentile_rank: float
    predicted_retention: float | None = None
    top_actions: list[ActionRanking]
    disclaimer: str = DISCLAIMER


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(metrics: OrgMetrics) -> BenchmarkResponse:
    try:
        from retention_bench.serve import benchmark as do_benchmark

        result = do_benchmark(metrics.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return BenchmarkResponse(**result)
