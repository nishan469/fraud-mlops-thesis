"""Replay: runs the full continuous-learning loop over the historical stream as if live.

Every `step_days`:
  1. monitor   - champion's live PR-AUC on matured labels, score/feature PSI
  2. decide    - scheduled retrain, or safety-net retrain on a sharp performance drop
  3. retrain   - train a challenger on the sliding window of labelled data, register it
  4. gate      - promote the challenger to `champion` only if it is not worse on unseen data
  5. serve     - score the next step's transactions with the champion
Monitoring is logged to an MLflow run (one metric point per step); each training is its own
MLflow run and registered model version.
"""

import copy
import json
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from mlflow.entities import Metric

from .data import TransactionStore, load_transactions
from .faults import FaultyStore, fault_features
from .monitoring import Monitor, mean_or_nan, served_weekly_metrics
from .policy import PromotionGate, RetrainPolicy
from .registry import ModelRegistry
from .training import train_on_window


class ReplayRunner:
    """`faults` ({scheduled retrain number: Fault}, 1 = first scheduled retrain; safety-net
    retrains are never faulted and don't shift the numbering) and
    `gate_enabled=False` exist for the gate experiments (gate_fault_test.py); `model_cache`
    (a dict shared across runners) avoids retraining identical windows between scenarios."""

    def __init__(self, cfg, store=None, log=print, faults=None, gate_enabled=True,
                 model_cache=None):
        self.cfg = cfg
        self.log = log
        if store is None:
            log("Loading data ...")
            store = TransactionStore(load_transactions(cfg.data.data_dir, cfg.data.cache_path))
        self.store = store
        self.registry = ModelRegistry(cfg.mlflow)
        self.policy = RetrainPolicy(cfg.policy)
        self.gate = PromotionGate(cfg.gate)
        self.monitor = Monitor(cfg.policy, cfg.monitoring)
        self.faults = faults or {}
        self.gate_enabled = gate_enabled
        self.model_cache = model_cache if model_cache is not None else {}

    def _train(self, now, reason, fault=None, champion=None):
        """Train on the policy's window; with a fault, through a corrupted view of the data.
        Returns (model, store the gate should read labels from)."""
        lo, hi = self.policy.training_window(now)
        train_store, gate_store, feats = self.store, self.store, ()
        if fault is not None:
            feats = fault_features(fault, champion)
            train_store = FaultyStore(self.store, fault, lo, hi, feats)
            if fault.upstream:
                gate_store = train_store
        key = (round(lo, 4), round(hi, 4), fault.name if fault else None, feats)
        tag = f" [FAULT: {fault.name}]" if fault else ""
        if key in self.model_cache:
            model = copy.copy(self.model_cache[key])
            self.log(f"  reusing model for days {max(lo, 0):.1f}-{hi:.1f}{tag}")
        else:
            self.log(f"  training on days {max(lo, 0):.1f}-{hi:.1f} ({reason}){tag}")
            model = train_on_window(train_store, lo, hi, self.cfg.model,
                                    n_ref_sample=self.cfg.monitoring.psi_sample)
            self.model_cache[key] = copy.copy(model)
        model.version = None
        tags = {"trigger": reason, "replay_day": f"{now:.2f}"}
        if fault:
            tags["fault"] = fault.name
        self.registry.register(model, tags=tags)
        return model, gate_store

    def run(self, end_day=None):
        cfg, store = self.cfg, self.store
        t0 = store.day_at_fraction(cfg.replay.start_frac)
        end = min(end_day or np.inf, float(store.day[-1]) + 1e-6)
        steps = np.arange(t0, end, cfg.policy.step_days)
        os.makedirs(cfg.replay.out_dir, exist_ok=True)

        mon_run = self.registry.client.create_run(
            self.registry.monitoring_exp, run_name=f"replay_{time.strftime('%Y%m%d_%H%M%S')}",
            tags={"label_delay_days": str(cfg.policy.label_delay_days)})
        run_id = mon_run.info.run_id
        self.registry.client.log_dict(run_id, asdict(cfg), "config.json")
        self.log(f"Replay days {t0:.1f}-{end:.1f} in {len(steps)} steps; MLflow run {run_id}")

        champion, _ = self._train(t0, "initial")
        self.registry.promote(champion.version, "initial model")
        last_retrain, n_scheduled = t0, 0
        served, decisions, last_served, last_scores = [], [], None, None

        for now in steps:
            report = self.monitor.check(store, champion, now, last_served, last_scores)
            decision = self.policy.decide(now, last_retrain, report)
            row = {"day": now, "champion": champion.version, "alerts": ";".join(report.alerts),
                   **report.metrics(), "retrain": decision.retrain, "reason": decision.reason}

            if decision.retrain:
                self.log(f"day {now:.1f}: {decision.reason}")
                fault = None
                if decision.reason.startswith("scheduled"):
                    n_scheduled += 1
                    fault = self.faults.get(n_scheduled)
                challenger, gate_store = self._train(now, decision.reason, fault, champion)
                gate = self.gate.evaluate(gate_store, challenger, champion)
                if not self.gate_enabled:
                    verdict = "pass" if gate.promote else "reject"
                    gate.promote, gate.reason = True, f"gate disabled (would {verdict}: {gate.reason})"
                row.update({"challenger": challenger.version, "promoted": gate.promote,
                            "gate": gate.reason, "gate_challenger_pr_auc": gate.challenger_pr_auc,
                            "gate_champion_pr_auc": gate.champion_pr_auc,
                            "fault": fault.name if fault else ""})
                self.log(f"  gate: {'PROMOTE' if gate.promote else 'REJECT'} v{challenger.version}"
                         f" - {gate.reason}")
                if gate.promote:
                    self.registry.promote(challenger.version, f"day {now:.1f}: {gate.reason}")
                    champion = challenger
                else:
                    self.registry.reject(challenger.version, gate.reason)
                last_retrain = now

            sl = store.rows(now, min(now + cfg.policy.step_days, end))
            if sl.stop > sl.start:
                last_served, last_scores = sl, champion.predict(store.X(sl))
                served.append((now, sl, last_scores, champion.version))
            decisions.append(row)
            step = int(round(now))
            ts = int(time.time() * 1000)
            self.registry.client.log_batch(run_id, metrics=[
                Metric(k, v, ts, step) for k, v in report.metrics().items()]
                + [Metric("retrain", float(decision.retrain), ts, step)])

        weekly = pd.DataFrame(served_weekly_metrics(store, served))
        decisions = pd.DataFrame(decisions)
        summary = {"label_delay_days": cfg.policy.label_delay_days,
                   "served_mean_weekly_pr_auc": mean_or_nan(weekly["pr_auc"]),
                   "served_min_weekly_pr_auc": float(weekly["pr_auc"].min()),
                   "retrains": int(decisions["retrain"].sum()),
                   "promotions": int((decisions.get("promoted", pd.Series(dtype=object)) == True).sum()),
                   "rejections": int((decisions.get("promoted", pd.Series(dtype=object)) == False).sum()),
                   "final_champion": champion.version, "mlflow_run_id": run_id}
        ts = int(time.time() * 1000)
        self.registry.client.log_batch(run_id, metrics=[
            Metric("served_pr_auc", float(r.pr_auc), ts, int(round(r.day_start)))
            for r in weekly.itertuples() if r.pr_auc == r.pr_auc]
            + [Metric(k, float(v), ts, 0) for k, v in summary.items()
               if k in ("served_mean_weekly_pr_auc", "served_min_weekly_pr_auc", "retrains",
                        "promotions", "rejections")])
        weekly.to_csv(os.path.join(cfg.replay.out_dir, "served_weekly.csv"), index=False)
        decisions.to_csv(os.path.join(cfg.replay.out_dir, "decisions.csv"), index=False)
        with open(os.path.join(cfg.replay.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        for name in ("served_weekly.csv", "decisions.csv", "summary.json"):
            self.registry.client.log_artifact(run_id, os.path.join(cfg.replay.out_dir, name))
        self.registry.client.set_terminated(run_id)
        return summary, weekly, decisions
