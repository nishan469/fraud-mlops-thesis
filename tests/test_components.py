import numpy as np
import pytest

from fraud_mlops.config import load_config
from fraud_mlops.metrics import pr_auc, psi
from fraud_mlops.monitoring import Monitor, MonitorReport
from fraud_mlops.policy import PromotionGate, RetrainPolicy
from fraud_mlops.training import train_on_window


def test_default_config_loads():
    cfg = load_config("configs/default.toml")
    assert cfg.policy.retrain_every_days == 14
    assert cfg.policy.train_window_days == 60


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(None, {"policy": {"retrain_evry_days": 7}})


def test_psi_detects_shift_and_ignores_resampling():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20000)
    assert psi(ref, rng.normal(0, 1, 20000)) < 0.01
    assert psi(ref, rng.normal(1.5, 1, 20000)) > 0.25


def test_pr_auc_single_class_is_nan():
    assert np.isnan(pr_auc(np.zeros(10), np.linspace(0, 1, 10)))


def test_policy_schedule_and_safety_net(cfg):
    policy = RetrainPolicy(cfg.policy)
    quiet = MonitorReport(day=0, matured_from=0, matured_until=0)
    drop = MonitorReport(day=0, matured_from=0, matured_until=0, live_pr_auc=0.1,
                         ref_pr_auc=0.6, alerts=["performance_drop"])
    assert not policy.decide(100, 95, quiet).retrain                       # too early
    assert policy.decide(100, 86, quiet).reason.startswith("scheduled")   # 14 days
    assert policy.decide(100, 92, drop).reason.startswith("safety net")   # past cooldown
    assert not policy.decide(100, 96, drop).retrain                       # within cooldown
    assert policy.training_window(100) == (50, 90)                        # delay 10, window 40


def test_no_skill_triggers_safety_net(cfg):
    rep = MonitorReport(day=0, matured_from=0, matured_until=0, live_pr_auc=0.05,
                        ref_pr_auc=0.06, matured_fraud_rate=0.04, alerts=["no_skill"])
    decision = RetrainPolicy(cfg.policy).decide(100, 92, rep)
    assert decision.retrain and "no skill" in decision.reason


def test_safety_net_reference_ignores_corrupted_self_report(store, cfg):
    """A model trained on shuffled labels reports a near-random validation PR-AUC; judged
    against that alone it looks healthy. Recent live history and the no-skill check catch it."""
    from fraud_mlops.faults import Fault, FaultyStore
    bad = train_on_window(FaultyStore(store, Fault("s", "label_shuffle"), 0, 50), 0, 50,
                          cfg.model, n_ref_sample=1000)
    monitor = Monitor(cfg.policy, cfg.monitoring)
    fresh = monitor.check(store, bad, now=80)
    assert fresh.ref_pr_auc == bad.spec["val_pr_auc"]            # no history yet
    monitor.history = [0.5, 0.45, 0.55]                          # healthy production so far
    rep = monitor.check(store, bad, now=80)
    assert rep.history_pr_auc == 0.5 and rep.ref_pr_auc == 0.5
    assert "performance_drop" in rep.alerts and "no_skill" in rep.alerts


def test_training_and_gate(store, cfg):
    champion = train_on_window(store, 0, 50, cfg.model, n_ref_sample=1000)
    assert 0 < champion.threshold < 1
    assert champion.spec["train_hi"] == 50
    gate = PromotionGate(cfg.gate)
    assert gate.evaluate(store, champion, None).promote

    challenger = train_on_window(store, 40, 100, cfg.model, n_ref_sample=1000)
    result = gate.evaluate(store, challenger, champion)
    assert result.n_eval > 0
    if not np.isnan(result.champion_pr_auc):
        # the synthetic signal drifts, so the newer model should hold up on recent data
        assert result.challenger_pr_auc >= result.champion_pr_auc - 0.05


def test_monitor_uses_only_unseen_matured_labels(store, cfg):
    model = train_on_window(store, 0, 50, cfg.model, n_ref_sample=1000)
    rep = Monitor(cfg.policy, cfg.monitoring).check(store, model, now=80,
                                                     last_served=store.rows(73, 80),
                                                     last_scores=model.predict(store.X(store.rows(73, 80))))
    assert rep.matured_from >= 50 and rep.matured_until == 70
    assert rep.matured_rows > 0 and not np.isnan(rep.live_pr_auc)
    assert not np.isnan(rep.score_psi) and len(rep.feature_psi) > 0


def test_expanding_window_starts_at_beginning():
    from fraud_mlops.config import PolicyConfig
    from fraud_mlops.policy import RetrainPolicy
    pol = RetrainPolicy(PolicyConfig(label_delay_days=30, train_window_days=0))
    assert pol.training_window(100) == (0.0, 70)
    pol = RetrainPolicy(PolicyConfig(label_delay_days=30, train_window_days=60))
    assert pol.training_window(100) == (10, 70)
