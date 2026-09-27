import numpy as np
import pandas as pd
import pytest

from fraud_mlops.config import load_config
from fraud_mlops.data import TransactionStore


def make_synthetic(n=12000, days=120, seed=0):
    """Small IEEE-CIS-shaped table with concept drift: the fraud signal moves from
    TransactionAmt to C1 over time, so older models decay."""
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.uniform(0, days * 86400, n))
    t = dt / (days * 86400)
    amt = rng.lognormal(3.5, 1.0, n)
    c1 = rng.normal(0, 1, n)
    product = rng.choice(["W", "C", "H", "R"], n, p=[0.6, 0.2, 0.1, 0.1])
    logit = (-3.6 + (1 - t) * 1.2 * (np.log(amt) - 3.5) + t * 1.5 * c1
             + 0.8 * (product == "C"))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"TransactionID": np.arange(n), "isFraud": y, "TransactionDT": dt.astype(int),
                       "TransactionAmt": amt.astype("float32"), "C1": c1.astype("float32"),
                       "card1": rng.integers(1000, 1100, n),
                       "ProductCD": pd.Categorical(product)})
    df.loc[rng.uniform(size=n) < 0.2, "C1"] = np.nan
    return df


@pytest.fixture(scope="session")
def store():
    return TransactionStore(make_synthetic())


@pytest.fixture
def cfg(tmp_path):
    return load_config(None, {
        "mlflow": {"tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                   "artifact_dir": str(tmp_path / "mlruns")},
        "model": {"num_leaves": 15, "min_child_samples": 20, "num_boost_round": 60,
                  "early_stopping_rounds": 10},
        "policy": {"label_delay_days": 10, "step_days": 7, "retrain_every_days": 14,
                   "train_window_days": 40, "monitor_days": 14, "min_positives": 5},
        "gate": {"min_positives": 5},
        "monitoring": {"psi_features": 3, "psi_sample": 1000},
        "replay": {"start_frac": 0.5, "out_dir": str(tmp_path / "replay")},
    })
