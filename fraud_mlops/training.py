"""Training a model on a window of labelled data."""

import time

import lightgbm as lgb
import numpy as np

from .metrics import best_f1_threshold, pr_auc
from .model import FraudModel


def lgb_params(cfg, pos_weight):
    return dict(objective="binary", metric="auc", learning_rate=cfg.learning_rate,
                num_leaves=cfg.num_leaves, max_depth=-1, min_child_samples=cfg.min_child_samples,
                subsample=cfg.subsample, subsample_freq=1, colsample_bytree=cfg.colsample_bytree,
                reg_lambda=cfg.reg_lambda, scale_pos_weight=pos_weight,
                random_state=cfg.seed, n_jobs=-1, verbose=-1)


def train_on_window(store, lo_day, hi_day, cfg, n_ref_sample=50000):
    """Train on rows with lo_day <= day < hi_day. The most recent cfg.val_frac of the window
    is held out (time order) for early stopping, threshold choice and the promotion gate."""
    window = store.rows(lo_day, hi_day)
    n = window.stop - window.start
    if n < 100:
        raise ValueError(f"Too few rows to train on days {lo_day:.1f}-{hi_day:.1f}: {n}")
    cut = window.start + int(n * (1 - cfg.val_frac))
    tr, va = slice(window.start, cut), slice(cut, window.stop)
    y_tr, y_va = store.y[tr], store.y[va]
    if y_tr.sum() == 0:
        raise ValueError(f"No positives in training window {lo_day:.1f}-{hi_day:.1f}")

    params = lgb_params(cfg, (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
    X_tr, X_va = store.X(tr), store.X(va)
    t = time.time()
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=store.categorical)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=store.categorical, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=cfg.num_boost_round, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])
    seconds = time.time() - t

    val_scores = booster.predict(X_va, num_iteration=booster.best_iteration)
    spec = {"features": list(store.features), "categorical": list(store.categorical),
            "threshold": best_f1_threshold(y_va, val_scores) if y_va.sum() else 0.5,
            "best_iteration": int(booster.best_iteration or booster.current_iteration()),
            "train_lo": float(max(lo_day, float(store.day[0]))), "train_hi": float(hi_day),
            "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
            "val_pr_auc": pr_auc(y_va, val_scores), "val_fraud_rate": float(y_va.mean()),
            "train_seconds": seconds, "seed": cfg.seed}
    model = FraudModel(booster, spec)
    model.val_rows, model.val_scores = va, val_scores

    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(np.arange(tr.start, tr.stop), size=min(n_ref_sample, len(y_tr)), replace=False)
    ref = store.df.iloc[np.sort(idx)]
    model.feature_ref = {f: ref[f].to_numpy() for f in model.top_numeric_features(50)}
    return model
