"""Smoke tests for the retention benchmarking project."""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SAMPLE = dict(
    sector="ICT",
    headcount=800,
    comp_percentile=55.0,
    training_hours=32.0,
    manager_quality=3.6,
    retention_rate=0.74,
)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_benchmark_stub():
    r = client.post("/benchmark", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["sector"] == "ICT"
    assert isinstance(body["top_actions"], list)
    assert "disclaimer" in body


def test_data_generator_shape():
    from retention_bench.data import generate

    df = generate(n_rows=200, seed=1)
    assert len(df) == 200
    assert set(df.columns) >= {
        "org_id", "sector", "year", "headcount", "retention_rate",
        "comp_percentile", "training_hours", "manager_quality",
        "action_taken", "reward",
    }


def test_bandit_rank_deterministic():
    from retention_bench.data import generate
    from retention_bench.models import EpsilonGreedyBandit

    df = generate(n_rows=500, seed=1)
    a = EpsilonGreedyBandit().fit(df)
    b = EpsilonGreedyBandit().fit(df)
    assert a.rank("ICT")[0]["action"] == b.rank("ICT")[0]["action"]
