"""Monitoring the champion. Performance on matured (delayed) labels drives the safety net;
score and feature PSI are recorded as information only, because in the thesis experiments
they stayed low while performance fell (they never fired)."""

from dataclasses import dataclass, field

import numpy as np

from .metrics import pr_auc, psi


@dataclass
class MonitorReport:
    day: float
    matured_from: float
    matured_until: float
    matured_rows: int = 0
    matured_positives: int = 0
    live_pr_auc: float = float("nan")
    ref_pr_auc: float = float("nan")
    score_psi: float = float("nan")
    feature_psi: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)

    @property
    def performance_drop(self):
        return "performance_drop" in self.alerts

    def metrics(self):
        """Flat numeric dict for MLflow."""
        out = {"live_pr_auc": self.live_pr_auc, "ref_pr_auc": self.ref_pr_auc,
               "score_psi": self.score_psi, "matured_rows": self.matured_rows,
               "matured_positives": self.matured_positives,
               "max_feature_psi": max(self.feature_psi.values(), default=float("nan"))}
        out.update({f"psi_{k}": v for k, v in self.feature_psi.items()})
        return {k: float(v) for k, v in out.items() if v == v}   # drop NaN


class Monitor:
    def __init__(self, policy_cfg, monitoring_cfg):
        self.p, self.m = policy_cfg, monitoring_cfg

    def check(self, store, champion, now, last_served=None, last_scores=None):
        """`last_served` is the row slice most recently scored, `last_scores` its scores."""
        until = now - self.p.label_delay_days
        start = max(champion.spec["train_hi"], until - self.p.monitor_days)
        rep = MonitorReport(day=now, matured_from=start, matured_until=until,
                            ref_pr_auc=champion.spec["val_pr_auc"])

        # 1) performance on labels that have matured and that the champion did not train on
        if until > start:
            rows = store.rows(start, until)
            y = store.y[rows]
            rep.matured_rows, rep.matured_positives = int(len(y)), int(y.sum())
            if rep.matured_positives >= self.p.min_positives:
                rep.live_pr_auc = pr_auc(y, champion.predict(store.X(rows)))
                if rep.live_pr_auc < rep.ref_pr_auc * (1 - self.p.safety_net_tol):
                    rep.alerts.append("performance_drop")

        # 2) drift of scores and top features on the latest traffic (informational)
        if last_scores is not None and champion.val_scores is not None:
            rep.score_psi = psi(champion.val_scores, last_scores)
            if rep.score_psi > self.m.psi_alert:
                rep.alerts.append("score_drift")
        if last_served is not None and champion.feature_ref:
            recent = store.df.iloc[last_served]
            for f in list(champion.feature_ref)[: self.m.psi_features]:
                rep.feature_psi[f] = psi(champion.feature_ref[f], recent[f].to_numpy())
            drifted = [f for f, v in rep.feature_psi.items() if v > self.m.psi_alert]
            if drifted:
                rep.alerts.append("feature_drift:" + ",".join(drifted))
        return rep


def served_weekly_metrics(store, served):
    """After the replay, all labels are known: realised PR-AUC of what was actually served,
    per serving step. `served` is a list of (step_start_day, row_slice, scores, version)."""
    rows = []
    for day, sl, scores, version in served:
        y = store.y[sl]
        rows.append({"day_start": day, "n": int(len(y)), "fraud_rate": float(y.mean()),
                     "pr_auc": pr_auc(y, scores), "model_version": version})
    return rows


def mean_or_nan(values):
    values = [v for v in values if v == v]
    return float(np.mean(values)) if values else float("nan")
