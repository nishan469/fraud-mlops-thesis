import os

import numpy as np
import pytest

from fraud_mlops.pipeline import ReplayRunner
from fraud_mlops.registry import ModelRegistry


def test_replay_end_to_end(store, cfg):
    summary, weekly, decisions = ReplayRunner(cfg, store=store, log=lambda *_: None).run()

    assert summary["retrains"] >= 2                      # 60-day replay, 14-day schedule
    assert summary["promotions"] >= 1
    assert 0 < summary["served_mean_weekly_pr_auc"] <= 1
    assert len(weekly) == len(decisions)
    for name in ("served_weekly.csv", "decisions.csv", "summary.json"):
        assert os.path.exists(os.path.join(cfg.replay.out_dir, name))

    # the registry's champion is the final served model and reproduces its scores
    reg = ModelRegistry(cfg.mlflow)
    assert reg.champion_version() == summary["final_champion"]
    loaded = reg.load()
    rows = store.rows(100, 105)
    p = loaded.predict(store.X(rows))
    assert p.shape == (rows.stop - rows.start,) and np.all((p >= 0) & (p <= 1))

    # the dashboard embeds this replay's data and the registry versions it used
    from fraud_mlops.dashboard import build_dashboard, collect
    data = collect(cfg)
    assert data["kpis"]["retrains"] == summary["retrains"]
    assert len(data["steps"]) == len(decisions) and len(data["weekly"]) == len(weekly)
    assert {v["version"] for v in data["versions"]} >= {summary["final_champion"]}
    html = open(build_dashboard(cfg), encoding="utf-8").read()
    assert "__DASHBOARD_DATA__" not in html and "<title>Fraud Replay Monitor</title>" in html


def test_scoring_api(store, cfg):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from fraud_mlops.serving import create_app

    cfg.replay.start_frac = 0.85                         # short replay, just to get a champion
    ReplayRunner(cfg, store=store, log=lambda *_: None).run()

    with TestClient(create_app(cfg)) as client:
        assert client.get("/health").json()["champion_version"] is not None
        # raw JSON: strings for categoricals, a missing feature (card1) and an unknown extra
        body = {"transactions": [
            {"TransactionAmt": 120.5, "C1": 0.3, "ProductCD": "C", "unknown_field": 1},
            {"TransactionAmt": 9.99, "C1": None, "ProductCD": "W", "card1": 1042}]}
        out = client.post("/predict", json=body).json()
        assert len(out["predictions"]) == 2
        assert all(0 <= p["fraud_probability"] <= 1 for p in out["predictions"])
        assert client.post("/predict", json={"transactions": []}).status_code == 422
