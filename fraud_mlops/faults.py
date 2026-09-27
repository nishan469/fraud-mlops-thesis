"""Fault injection for testing the promotion gate.

A fault corrupts what one training job sees, the way real pipeline failures do. By default
only the training extract is corrupted: the production feature path and label store stay
clean, so the gate (which re-scores on production data) can catch it. `upstream=True` models
corruption in the label store itself, which the gate also reads: a known blind spot.

Kinds
  label_shuffle  labels permuted within the window (misaligned join in the training job)
  label_loss     a share (`rate`) of frauds recorded as legitimate (chargeback feed outage)
  feature_unit   the champion's top-k numeric features multiplied by `factor` in training
                 only (unit bug, e.g. cents vs dollars: training/serving skew)
"""

from dataclasses import dataclass

import numpy as np

KINDS = ("label_shuffle", "label_loss", "feature_unit")


@dataclass(frozen=True)
class Fault:
    name: str
    kind: str
    upstream: bool = False
    rate: float = 0.8
    factor: float = 100.0
    k: int = 5
    seed: int = 0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown fault kind {self.kind!r}; choose from {KINDS}")


class FaultyStore:
    """A TransactionStore view with the fault applied to rows in [lo_day, hi_day)."""

    def __init__(self, base, fault, lo_day, hi_day, features=()):
        self.base, self.fault = base, fault
        self.df, self.day = base.df, base.day
        self.features, self.categorical = base.features, base.categorical
        self.window = base.rows(lo_day, hi_day)
        self.corrupt_features = list(features) if fault.kind == "feature_unit" else []
        rng = np.random.default_rng(fault.seed)
        y = base.y.copy()
        w = self.window
        if fault.kind == "label_shuffle":
            y[w] = rng.permutation(y[w])
        elif fault.kind == "label_loss":
            pos = np.flatnonzero(y[w]) + w.start
            y[rng.choice(pos, size=int(len(pos) * fault.rate), replace=False)] = 0
        self.y = y

    def row(self, day):
        return self.base.row(day)

    def rows(self, start_day, end_day):
        return self.base.rows(start_day, end_day)

    def X(self, rows):
        X = self.base.X(rows)
        if self.corrupt_features:
            pos = np.arange(rows.start, rows.stop)
            hit = (pos >= self.window.start) & (pos < self.window.stop)
            if hit.any():
                X = X.copy()
                for f in self.corrupt_features:
                    v = X[f].to_numpy(dtype="float64", na_value=np.nan)   # int columns would overflow
                    v[hit] *= self.fault.factor
                    X[f] = v
        return X


def fault_features(fault, champion):
    """Features a feature fault hits: the champion's most important numeric ones."""
    return tuple(champion.top_numeric_features(fault.k)) if fault.kind == "feature_unit" else ()


SCENARIOS = {
    "label_shuffle": Fault("label_shuffle", "label_shuffle"),
    "label_loss": Fault("label_loss", "label_loss", rate=0.8),
    "feature_unit": Fault("feature_unit", "feature_unit", factor=100.0, k=5),
    "upstream_label_shuffle": Fault("upstream_label_shuffle", "label_shuffle", upstream=True),
}
