"""
IEEE-CIS Fraud Detection: retraining-strategy simulation under label delay
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Replays the last part of the data as a stream and compares how a deployed LightGBM
model is kept up to date:

  static              train once, never update (the baseline from ieee_cis_eda_baseline.py)
  periodic_expanding  retrain every --period_days on all labelled data so far
  periodic_sliding    retrain every --period_days on the last --window_days of labelled data
  drift_score_psi     retrain when PSI of the model's scores (no labels needed) > --psi_threshold
  drift_performance   retrain when PR-AUC on newly matured labels drops > --perf_tol below
                      a reference (--perf_ref: the model's validation PR-AUC, or its first
                      live measurement); trains on all data or --drift_window_days

Label delay: a transaction's label only becomes usable --label_delay days after it happens
(chargebacks arrive late). Every model, including the initial one, is trained only on
data whose labels were available at the time it was trained.

Usage (reuses loading/feature/metric code from ieee_cis_eda_baseline.py, same folder):
  python retraining_simulation.py --data_dir ./data --label_delay 30 --period_days 14

Outputs (in --out_dir, default ./outputs/retraining):
  sim_summary.csv, sim_by_window.csv, sim_events.csv, sim_monitor.csv, sim_performance.png
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import ieee_cis_eda_baseline as baseline
from ieee_cis_eda_baseline import (best_f1_threshold, load_data, metrics,
                                   prepare_features, psi, train_baseline)

STRATEGIES = ["static", "periodic_expanding", "periodic_sliding",
              "drift_score_psi", "drift_performance"]


# ---------------------------------------------------------------- models
class Deployed:
    """A trained model plus the reference information captured at training time."""

    def __init__(self, key, model, thr, ref_pr_auc, ref_scores, lo, hi, trained_at, train_sec):
        self.key = key
        self.model, self.thr = model, thr
        self.ref_pr_auc, self.ref_scores = ref_pr_auc, ref_scores
        self.lo, self.hi = lo, hi              # training data window, in days
        self.trained_at = trained_at           # stream day the model went live
        self.train_sec = train_sec


class Trainer:
    """Trains on a day window with a time-ordered val split. Models and predictions are
    cached, so re-running strategies that retrain on the same windows is cheap."""

    def __init__(self, df, day, feats, cat_cols, val_frac, seed=42):
        self.df, self.day = df, day
        self.feats, self.cat_cols, self.val_frac = feats, cat_cols, val_frac
        self.seed = seed
        self.cache = {}
        self.pred_cache = {}

    def set_seed(self, seed):
        """Change LightGBM's random_state for future fits; drops cached models/predictions."""
        self.seed = seed
        self.cache.clear()
        self.pred_cache.clear()

    def idx(self, d):
        return int(np.searchsorted(self.day, d, side="left"))

    def predict(self, dep, a, b):
        """Scores of model `dep` on rows a:b."""
        key = (dep.key, a, b)
        if key not in self.pred_cache:
            self.pred_cache[key] = dep.model.predict(self.df.iloc[a:b][self.feats],
                                                     num_iteration=dep.model.best_iteration)
        return self.pred_cache[key]

    def fit(self, lo, hi, trained_at):
        key = (self.seed, round(lo, 4), round(hi, 4))
        if key not in self.cache:
            baseline.SEED = self.seed   # train_baseline reads random_state from this global
            data = self.df.iloc[self.idx(lo):self.idx(hi)]
            cut = int(len(data) * (1 - self.val_frac))
            train, val = data.iloc[:cut], data.iloc[cut:]
            print(f"  training on days {max(lo, 0):.1f}-{hi:.1f} "
                  f"({len(train):,} train / {len(val):,} val)")
            t = time.time()
            model, _ = train_baseline(train, val, self.feats, self.cat_cols)
            sec = time.time() - t
            p_val = model.predict(val[self.feats], num_iteration=model.best_iteration)
            y_val = val["isFraud"].values
            self.cache[key] = (model, best_f1_threshold(y_val, p_val),
                               float(average_precision_score(y_val, p_val)), p_val, sec)
        model, thr, ref_pr, ref_scores, sec = self.cache[key]
        return Deployed(key, model, thr, ref_pr, ref_scores, lo, hi, trained_at, sec)


# ---------------------------------------------------------------- simulation
def simulate(strategy, initial, trainer, steps, args, t_end=np.inf):
    y_all = trainer.df["isFraud"].values
    dep = initial
    ref = dep.ref_pr_auc if args.perf_ref == "val" else None
    events, monitor, windows = [], [], []
    scores, thrs = [], []
    last_p = None
    verbose = getattr(args, "verbose", True)

    for now in steps:
        labelled_until = now - args.label_delay
        since = now - dep.trained_at
        reason, stat = None, np.nan

        if strategy.startswith("periodic") and since >= args.period_days:
            reason = f"schedule ({args.period_days}d)"

        elif strategy == "drift_score_psi" and last_p is not None:
            stat = psi(dep.ref_scores, last_p)
            if stat > args.psi_threshold and since >= args.cooldown_days:
                reason = f"score PSI {stat:.3f} > {args.psi_threshold}"

        elif strategy == "drift_performance":
            # labels that matured recently, restricted to data the model has not trained on
            a = trainer.idx(max(dep.hi, labelled_until - args.monitor_days))
            b = trainer.idx(labelled_until)
            if b > a and y_all[a:b].sum() >= args.min_positives:
                stat = float(average_precision_score(y_all[a:b], trainer.predict(dep, a, b)))
                if ref is None:
                    # perf_ref == "first": the first live measurement becomes the reference
                    ref = stat
                elif stat < ref * (1 - args.perf_tol) and since >= args.cooldown_days:
                    reason = f"PR-AUC {stat:.3f} < {1 - args.perf_tol:.2f} x ref {ref:.3f}"

        monitor.append({"strategy": strategy, "day": now, "stat": stat, "ref_pr_auc": ref})

        if reason:
            if strategy == "periodic_sliding":
                lo = labelled_until - args.window_days
            elif strategy == "drift_performance" and args.drift_window_days > 0:
                lo = labelled_until - args.drift_window_days
            else:
                lo = -np.inf
            if verbose:
                print(f"[{strategy}] day {now:.1f}: retrain because {reason}")
            dep = trainer.fit(lo, labelled_until, trained_at=now)
            ref = dep.ref_pr_auc if args.perf_ref == "val" else None
            events.append({"strategy": strategy, "day": now, "reason": reason,
                           "train_lo": max(lo, 0.0), "train_hi": labelled_until,
                           "train_sec": dep.train_sec,
                           "best_iteration": dep.model.best_iteration})
            last_p = None

        # serve this step with the currently deployed model
        a, b = trainer.idx(now), trainer.idx(min(now + args.step_days, t_end))
        if b <= a:
            continue
        p = trainer.predict(dep, a, b)
        scores.append(p)
        thrs.append(np.full(len(p), dep.thr))
        last_p = p
        m = metrics(y_all[a:b], p, dep.thr)
        m.update({"strategy": strategy, "day_start": now})
        windows.append(m)

    return np.concatenate(scores), np.concatenate(thrs), windows, events, monitor


def pooled_summary(strategy, y, p, thr, windows, events):
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    prec = tp / max(yhat.sum(), 1)
    rec = tp / max(y.sum(), 1)
    w = pd.DataFrame(windows)
    return {"strategy": strategy,
            "n_retrains": len(events),
            "train_sec_total": float(sum(e["train_sec"] for e in events)),
            "roc_auc_pooled": float(roc_auc_score(y, p)),
            "pr_auc_pooled": float(average_precision_score(y, p)),
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "pr_auc_mean_window": float(w["pr_auc"].mean()),
            "pr_auc_min_window": float(w["pr_auc"].min()),
            "roc_auc_mean_window": float(w["roc_auc"].mean())}


def plot(by_window, events, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = dict(zip(by_window["strategy"].unique(), plt.cm.tab10.colors))
    for ax, col, label in [(axes[0], "pr_auc", "PR-AUC"), (axes[1], "roc_auc", "ROC-AUC")]:
        for s, g in by_window.groupby("strategy", sort=False):
            ax.plot(g["day_start"], g[col], marker="o", ms=3, label=s, color=colors[s])
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for _, e in events.iterrows():
        axes[0].axvline(e["day"], color=colors[e["strategy"]], ls=":", lw=1, alpha=0.7)
    axes[0].set_title("Retraining strategies over the stream (dotted lines = retrain events)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Day (start of evaluation window)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_strategies(trainer, strategies, args):
    """Run every strategy from the same initial model. Also returns per-transaction
    scores/thresholds so callers (e.g. sweep_label_delay.py) can bootstrap."""
    day = trainer.day
    t0 = float(day[int(len(day) * args.stream_start_frac)])
    steps = np.arange(t0, day.max(), args.step_days)
    print(f"\nStream: day {t0:.1f} to {day.max():.1f}, {len(steps)} steps of {args.step_days}d, "
          f"label delay {args.label_delay}d")

    print("Initial model (shared by all strategies):")
    initial = trainer.fit(-np.inf, t0 - args.label_delay, trained_at=t0)
    s0 = trainer.idx(t0)
    y_stream = trainer.df["isFraud"].values[s0:]

    summary, all_windows, all_events, all_monitor, preds = [], [], [], [], {}
    for s in strategies:
        print(f"\n=== {s} ===")
        p, thr, windows, events, monitor = simulate(s, initial, trainer, steps, args)
        summary.append(pooled_summary(s, y_stream[:len(p)], p, thr, windows, events))
        preds[s] = (p, thr)
        all_windows += windows
        all_events += events
        all_monitor += monitor

    events = pd.DataFrame(all_events, columns=["strategy", "day", "reason", "train_lo",
                                               "train_hi", "train_sec", "best_iteration"])
    return {"summary": pd.DataFrame(summary), "by_window": pd.DataFrame(all_windows),
            "events": events, "monitor": pd.DataFrame(all_monitor),
            "preds": preds, "stream_slice": slice(s0, len(day))}


# ---------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./outputs/retraining")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--stream_start_frac", type=float, default=0.6,
                    help="stream starts at this row fraction (0.6 = same test period as baseline)")
    ap.add_argument("--label_delay", type=float, default=30, help="days until a label is usable")
    ap.add_argument("--step_days", type=float, default=7, help="decision + evaluation window")
    ap.add_argument("--period_days", type=float, default=14)
    ap.add_argument("--window_days", type=float, default=60, help="sliding window length")
    ap.add_argument("--val_frac", type=float, default=0.15, help="tail of each training window")
    ap.add_argument("--psi_threshold", type=float, default=0.1)
    ap.add_argument("--perf_tol", type=float, default=0.15, help="relative PR-AUC drop that triggers")
    ap.add_argument("--monitor_days", type=float, default=14, help="matured-label window for drift_performance")
    ap.add_argument("--min_positives", type=int, default=30)
    ap.add_argument("--perf_ref", choices=["val", "first"], default="val",
                    help="drift_performance reference: model's validation PR-AUC, or its first "
                         "PR-AUC measured on matured live labels")
    ap.add_argument("--drift_window_days", type=float, default=0,
                    help="drift_performance training window (0 = all labelled data)")
    ap.add_argument("--cooldown_days", type=float, default=14, help="min days between drift retrains")
    return ap


def parse_strategies(text):
    strategies = [s.strip() for s in text.split(",") if s.strip()]
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise SystemExit(f"Unknown strategies: {unknown}. Choose from {STRATEGIES}")
    return strategies


def load_trainer(args):
    df = load_data(args.data_dir)
    feats, cat_cols = prepare_features(df)
    day = df["TransactionDT"].values / 86400.0
    return Trainer(df, day, feats, cat_cols, args.val_frac)


def main():
    args, _ = build_parser().parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)
    strategies = parse_strategies(args.strategies)
    res = run_strategies(load_trainer(args), strategies, args)

    summary, by_window, events = res["summary"], res["by_window"], res["events"]
    summary.to_csv(os.path.join(args.out_dir, "sim_summary.csv"), index=False)
    by_window.to_csv(os.path.join(args.out_dir, "sim_by_window.csv"), index=False)
    events.to_csv(os.path.join(args.out_dir, "sim_events.csv"), index=False)
    res["monitor"].to_csv(os.path.join(args.out_dir, "sim_monitor.csv"), index=False)
    plot(by_window, events, os.path.join(args.out_dir, "sim_performance.png"))

    print("\n=== Summary ===\n" + summary.round(4).to_string(index=False))
    if len(events):
        print("\n=== Retrain events ===\n" + events.round(2).to_string(index=False))

    # optional MLflow logging, one run per strategy
    try:
        import mlflow
        for _, r in summary.iterrows():
            with mlflow.start_run(run_name=f"sim_{r['strategy']}"):
                mlflow.log_params({k: v for k, v in vars(args).items() if k != "strategies"})
                mlflow.log_param("strategy", r["strategy"])
                mlflow.log_metrics({k: float(v) for k, v in r.items() if k != "strategy"})
                for _, w in by_window[by_window["strategy"] == r["strategy"]].iterrows():
                    mlflow.log_metric("window_pr_auc", w["pr_auc"], step=int(w["day_start"]))
        print("\nLogged runs to MLflow (view with: mlflow ui)")
    except ImportError:
        print("\nMLflow not installed, skipping experiment logging.")

    print(f"\nDone. Outputs are in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
