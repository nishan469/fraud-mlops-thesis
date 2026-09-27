import pandas as pd
import pytest

from fraud_mlops.faults import Fault, FaultyStore, fault_features
from fraud_mlops.metrics import pr_auc
from fraud_mlops.pipeline import ReplayRunner
from fraud_mlops.policy import PromotionGate
from fraud_mlops.training import train_on_window


@pytest.fixture(scope="module")
def champion(store):
    from fraud_mlops.config import load_config
    cfg = load_config(None, {"model": {"num_leaves": 15, "min_child_samples": 20,
                                       "num_boost_round": 60, "early_stopping_rounds": 10}})
    return train_on_window(store, 0, 60, cfg.model, n_ref_sample=1000)


def faulty_challenger(store, cfg, fault, champion, lo=30, hi=90):
    feats = fault_features(fault, champion)
    return train_on_window(FaultyStore(store, fault, lo, hi, feats), lo, hi, cfg.model,
                           n_ref_sample=1000)


def test_faulty_store_corrupts_only_the_window(store):
    fs = FaultyStore(store, Fault("x", "label_loss", rate=1.0), 30, 60)
    w = store.rows(30, 60)
    assert fs.y[w].sum() == 0 and store.y[w].sum() > 0          # all frauds lost inside
    assert (fs.y[: w.start] == store.y[: w.start]).all()         # untouched outside
    unit = FaultyStore(store, Fault("u", "feature_unit", factor=100, k=1), 30, 60, ("C1",))
    both = store.rows(25, 35)                                    # straddles the window start
    clean, bad = store.X(both)["C1"], unit.X(both)["C1"]
    inside = store.day[both] >= 30
    pd.testing.assert_series_equal(clean[~inside], bad[~inside], check_dtype=False)
    assert (bad[inside].abs().sum()) == pytest.approx(100 * clean[inside].abs().sum(), rel=1e-4)


def test_gate_promotes_clean_challenger(store, cfg, champion):
    challenger = train_on_window(store, 30, 90, cfg.model, n_ref_sample=1000)
    assert PromotionGate(cfg.gate).evaluate(store, challenger, champion).promote


def test_gate_rejects_label_shuffle(store, cfg, champion):
    bad = faulty_challenger(store, cfg, Fault("s", "label_shuffle"), champion)
    result = PromotionGate(cfg.gate).evaluate(store, bad, champion)
    assert not result.promote
    assert result.challenger_pr_auc < result.champion_pr_auc - 0.05


def test_gate_catches_training_serving_skew(store, cfg, champion):
    """A unit bug in training features looks fine on the training job's own validation
    scores; only re-scoring through the production path exposes it."""
    # all three numeric features: the synthetic signal drifts onto C1, so a fault that
    # misses it leaves a challenger that is legitimately better than the stale champion
    fault = Fault("u", "feature_unit", factor=100, k=3)
    bad = faulty_challenger(store, cfg, fault, champion)
    result = PromotionGate(cfg.gate).evaluate(store, bad, champion)
    assert not result.promote
    va = bad.val_rows
    self_reported = pr_auc(store.y[va], bad.val_scores)          # what the old gate compared
    assert self_reported > result.challenger_pr_auc + 0.05


@pytest.mark.parametrize("gate_enabled", [True, False])
def test_replay_with_fault(store, cfg, gate_enabled):
    runner = ReplayRunner(cfg, store=store, log=lambda *_: None, gate_enabled=gate_enabled,
                          faults={1: Fault("label_shuffle", "label_shuffle")})
    summary, _, decisions = runner.run()
    faulty = decisions[decisions["fault"].fillna("") == "label_shuffle"].iloc[0]
    if gate_enabled:
        assert faulty["promoted"] is False or faulty["promoted"] == False   # noqa: E712
        assert summary["rejections"] >= 1
        assert (decisions["champion"] != faulty["challenger"]).all()
    else:
        assert bool(faulty["promoted"]) and "would reject" in faulty["gate"]
        # the safety net replaces the no-skill model at the next step (cooldown = step)
        after = decisions[decisions["day"] > faulty["day"]].iloc[0]
        assert after["retrain"] and after["reason"].startswith("safety net")
        assert "no_skill" in after["alerts"]
