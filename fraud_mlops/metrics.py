"""Evaluation metrics and the Population Stability Index."""

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def pr_auc(y, p):
    """PR-AUC (average precision), NaN when only one class is present."""
    y = np.asarray(y)
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, p))


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[int(np.nanargmax(f1[:-1]))])


def classification_metrics(y, p, threshold):
    y = np.asarray(y)
    yhat = (np.asarray(p) >= threshold).astype(int)
    two_classes = len(y) > 0 and y.min() != y.max()
    return {"n": int(len(y)), "fraud_rate": float(y.mean()) if len(y) else float("nan"),
            "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat, zero_division=0)),
            "f1": float(f1_score(y, yhat, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
            "pr_auc": pr_auc(y, p)}


def psi(ref, cur, n_bins=10):
    """Population Stability Index of `cur` against `ref` (quantile bins of ref; NaN is its
    own bin). Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major shift."""
    ref, cur = pd.Series(np.asarray(ref, dtype=float)), pd.Series(np.asarray(cur, dtype=float))
    if ref.dropna().empty or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.nanquantile(ref.dropna(), np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    def dist(s):
        counts = pd.cut(s.dropna(), edges).value_counts(sort=False).to_numpy(dtype=float)
        counts = np.append(counts, s.isna().sum())
        return np.clip(counts / max(len(s), 1), 1e-6, None)

    r, c = dist(ref), dist(cur)
    return float(np.sum((c - r) * np.log(c / r)))
