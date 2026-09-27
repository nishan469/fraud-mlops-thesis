"""Scoring API serving the registry's `champion` model.

  GET  /health   -> status and the champion version being served
  POST /predict  -> {"transactions": [{"TransactionAmt": 50.0, "ProductCD": "W", ...}, ...]}
                    returns fraud probability and flag per transaction
  POST /reload   -> pick up a newly promoted champion without restarting
Missing features are treated as missing values (LightGBM handles NaN natively).
"""

import threading
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .registry import ModelRegistry


class PredictRequest(BaseModel):
    transactions: list[dict]


def create_app(cfg):
    registry = ModelRegistry(cfg.mlflow)
    state = {"model": None}
    lock = threading.Lock()

    def load():
        if registry.champion_version() is None:
            raise HTTPException(503, "No champion model registered yet; run a replay first.")
        with lock:
            state["model"] = registry.load()
        return state["model"]

    @asynccontextmanager
    async def lifespan(_app):
        try:
            load()
        except HTTPException:
            pass   # health reports "no model loaded"; /reload once a model exists
        yield

    app = FastAPI(title="IEEE-CIS fraud scoring", version="0.1", lifespan=lifespan)

    @app.get("/health")
    def health():
        m = state["model"]
        return {"status": "ok" if m else "no model loaded",
                "model_name": registry.name, "champion_version": m.version if m else None}

    @app.post("/reload")
    def reload():
        m = load()
        return {"champion_version": m.version}

    @app.post("/predict")
    def predict(req: PredictRequest):
        m = state["model"] or load()
        if not req.transactions:
            raise HTTPException(422, "transactions must be a non-empty list")
        scores, flags = m.decide(pd.DataFrame(req.transactions))
        return {"model_version": m.version, "threshold": m.threshold,
                "predictions": [{"fraud_probability": float(s), "is_fraud": int(f)}
                                for s, f in zip(scores, flags)]}

    return app
