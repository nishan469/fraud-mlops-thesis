"""When to retrain, and whether a newly trained challenger replaces the champion."""

from dataclasses import dataclass

import numpy as np

from .metrics import pr_auc


@dataclass
class Decision:
    retrain: bool
    reason: str = ""


class RetrainPolicy:
    """Primary: retrain every `retrain_every_days` (schedule on a sliding window).
    Safety net: retrain early on a sharp live performance drop, respecting a cooldown."""

    def __init__(self, cfg):
        self.cfg = cfg

    def decide(self, now, last_retrain_day, report):
        since = now - last_retrain_day
        if since >= self.cfg.retrain_every_days - 1e-9:
            return Decision(True, f"scheduled ({since:.0f}d since last retrain)")
        if report is not None and report.performance_drop and since >= self.cfg.cooldown_days - 1e-9:
            return Decision(True, f"safety net: live PR-AUC {report.live_pr_auc:.3f} < "
                                  f"{1 - self.cfg.safety_net_tol:.2f} x ref {report.ref_pr_auc:.3f}")
        return Decision(False)

    def training_window(self, now):
        hi = now - self.cfg.label_delay_days
        return hi - self.cfg.train_window_days, hi


@dataclass
class GateResult:
    promote: bool
    reason: str
    challenger_pr_auc: float = float("nan")
    champion_pr_auc: float = float("nan")
    n_eval: int = 0


class PromotionGate:
    """Compare challenger and champion on the challenger's hold-out (the most recent labelled
    data), restricted to rows the champion never trained on. Both models are re-scored through
    the production feature path (`store`), not with scores cached at training time, so a
    challenger trained on corrupted features (training/serving skew) is judged on the inputs
    it would actually receive in production."""

    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate(self, store, challenger, champion):
        if champion is None:
            return GateResult(True, "no champion yet")
        va = challenger.val_rows
        start = max(va.start, store.row(champion.spec["train_hi"]))
        rows = slice(start, va.stop)
        y = store.y[rows]
        if y.sum() < self.cfg.min_positives:
            return GateResult(True, f"only {int(y.sum())} unseen positives; promoted by default",
                              n_eval=int(len(y)))
        X = store.X(rows)
        ch = pr_auc(y, challenger.predict(X))
        cp = pr_auc(y, champion.predict(X))
        ok = bool(ch >= cp - self.cfg.max_pr_auc_drop) or np.isnan(cp)
        why = (f"challenger {ch:.3f} vs champion {cp:.3f} on {len(y)} unseen rows "
               f"(allowed drop {self.cfg.max_pr_auc_drop})")
        return GateResult(ok, why, ch, cp, int(len(y)))
