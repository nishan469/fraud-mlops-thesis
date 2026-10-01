"""Automatic training-window selection: one challenger per candidate window, the gate keeps
the best one, scored on rows that no candidate trained on."""

import dataclasses

from fraud_mlops.faults import Fault, FaultyStore
from fraud_mlops.pipeline import ReplayRunner
from fraud_mlops.policy import PromotionGate, RetrainPolicy
from fraud_mlops.registry import ModelRegistry
from fraud_mlops.training import train_on_window


def test_policy_lists_candidate_windows(cfg):
    assert RetrainPolicy(cfg.policy).windows() == [40]
    pol = RetrainPolicy(dataclasses.replace(cfg.policy, candidate_windows=[40, 0]))
    assert pol.windows() == [40, 0]
    assert pol.training_window(100, 0) == (0.0, 90) and pol.training_window(100, 40) == (50, 90)


def test_select_keeps_the_better_candidate(store, cfg):
    champion = train_on_window(store, 0, 50, cfg.model, n_ref_sample=1000)
    good = train_on_window(store, 30, 90, cfg.model, n_ref_sample=1000)
    shuffled = FaultyStore(store, Fault("shuffle", "label_shuffle"), 0, 90, ())
    bad = train_on_window(shuffled, 0, 90, cfg.model, n_ref_sample=1000)
    best, gate, scores = PromotionGate(cfg.gate).select(store, [bad, good], champion)
    assert best == 1 and len(scores) == 2 and scores[1] > scores[0]
    assert gate.challenger_pr_auc == scores[1] and gate.n_eval > 0
    # both candidates are judged on the same rows: the most recent ones neither trained on
    assert gate.n_eval <= good.val_rows.stop - good.val_rows.start


def test_replay_with_candidate_windows(store, cfg):
    cfg = dataclasses.replace(cfg, policy=dataclasses.replace(cfg.policy, candidate_windows=[40, 0]))
    summary, _, decisions = ReplayRunner(cfg, store=store, log=lambda *_: None).run()
    chosen = summary["windows_selected"]
    assert set(chosen) <= {"40d", "0d"} and sum(chosen.values()) == summary["retrains"]
    assert decisions.loc[decisions["retrain"], "window_days"].notna().all()
    # every candidate is registered; the ones not selected are tagged as rejected
    reg = ModelRegistry(cfg.mlflow)
    versions = reg.client.search_model_versions(f"name='{reg.name}'")
    assert len(versions) == 2 * (summary["retrains"] + 1)
    assert sum("not selected" in (v.tags.get("rejected") or "") for v in versions) \
        == summary["retrains"] + 1
