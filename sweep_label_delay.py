"""
IEEE-CIS Fraud Detection: label-delay sweep with bootstrap confidence intervals
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Runs retraining_simulation.py's strategies once per label delay and puts confidence
intervals on the results with a paired day-block bootstrap: within each 7-day evaluation
window, whole days are resampled with replacement (transactions within a day are
correlated), and the same resample is used for every strategy, so differences vs. static
are paired.

Headline metric is the mean of per-window PR-AUC, not PR-AUC pooled over the stream:
pooling mixes scores from differently calibrated models and penalises strategies that
retrain often (at a 60-day delay it flipped the sign of the retraining benefit).

Note: the CIs cover evaluation-sample noise only, not training randomness (seed is fixed).

Usage (same folder as retraining_simulation.py and ieee_cis_eda_baseline.py):
  python sweep_label_delay.py --data_dir ./data --delays 0,15,30,60 --n_boot 500

Accepts all retraining_simulation.py flags (--period_days, --window_days, ...).
Outputs (in --out_dir, default ./outputs/sweep):
  sweep_results.csv, sweep_by_window.csv, sweep_events.csv, sweep_label_delay.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from retraining_simulation import build_parser, load_trainer, parse_strategies, run_strategies

SEED = 42


def f1_at(y, p, thr):
    yhat = p >= thr
    tp = np.sum(yhat & (y == 1))
    prec, rec = tp / max(yhat.sum(), 1), tp / max(y.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def window_ids(day, step_days):
    """Evaluation window of each stream transaction (same 7-day steps as the simulation)."""
    return np.floor((day - day[0]) / step_days).astype(int)


def mean_window_pr_auc(y, p, win_idx):
    """Mean of per-window PR-AUC. Each window is scored by a single model, so this avoids
    pooling scores from differently calibrated models (which penalises frequent retraining)."""
    return float(np.mean([average_precision_score(y[i], p[i]) for i in win_idx
                          if 0 < y[i].sum() < len(i)]))


def point_estimates(y, preds, win_idx):
    return {s: (mean_window_pr_auc(y, p, win_idx), f1_at(y, p, thr))
            for s, (p, thr) in preds.items()}


def bootstrap(y, day, preds, step_days, n_boot, rng):
    """Paired, window-stratified day-block bootstrap: within every evaluation window, whole
    days are resampled with replacement; the same resample is used for every strategy."""
    win = window_ids(day, step_days)
    d = np.floor(day).astype(int)
    blocks = [[np.where((win == w) & (d == u))[0] for u in np.unique(d[win == w])]
              for w in np.unique(win)]
    out = {s: np.empty((n_boot, 2)) for s in preds}
    for b in range(n_boot):
        win_idx = [np.concatenate([bl[k] for k in rng.integers(0, len(bl), len(bl))])
                   for bl in blocks]
        idx = np.concatenate(win_idx)
        for s, (p, thr) in preds.items():
            out[s][b] = mean_window_pr_auc(y, p, win_idx), f1_at(y[idx], p[idx], thr[idx])
    return out


def summarise(delay, res, y, day, step_days, n_boot, rng):
    preds = res["preds"]
    win = window_ids(day, step_days)
    point = point_estimates(y, preds, [np.where(win == w)[0] for w in np.unique(win)])
    boots = bootstrap(y, day, preds, step_days, n_boot, rng)
    base = boots.get("static")
    rows = []
    for _, r in res["summary"].iterrows():
        s = r["strategy"]
        pr, f1 = point[s]
        bs = boots[s]
        row = {"label_delay": delay, "strategy": s, "n_retrains": int(r["n_retrains"]),
               "train_sec_total": r["train_sec_total"],
               "pr_auc": pr, "pr_auc_lo": np.percentile(bs[:, 0], 2.5),
               "pr_auc_hi": np.percentile(bs[:, 0], 97.5),
               "f1": f1, "f1_lo": np.percentile(bs[:, 1], 2.5),
               "f1_hi": np.percentile(bs[:, 1], 97.5),
               "pr_auc_pooled": r["pr_auc_pooled"],
               "pr_auc_min_window": r["pr_auc_min_window"]}
        if base is not None:
            diff = bs[:, 0] - base[:, 0]
            row.update({"delta_pr_auc_vs_static": pr - point["static"][0],
                        "delta_lo": np.percentile(diff, 2.5),
                        "delta_hi": np.percentile(diff, 97.5),
                        "p_delta_le_0": float(np.mean(diff <= 0))})
        rows.append(row)
    return rows


def plot(results, out_path):
    has_delta = "delta_pr_auc_vs_static" in results
    fig, axes = plt.subplots(1, 2 if has_delta else 1, figsize=(13 if has_delta else 7, 5))
    axes = np.atleast_1d(axes)
    strategies = list(results["strategy"].unique())
    offsets = np.linspace(-0.25, 0.25, len(strategies))
    delays = sorted(results["label_delay"].unique())
    pos = {d: i for i, d in enumerate(delays)}

    for off, s in zip(offsets, strategies):
        g = results[results["strategy"] == s].sort_values("label_delay")
        x = g["label_delay"].map(pos) + off
        axes[0].errorbar(x, g["pr_auc"], yerr=[g["pr_auc"] - g["pr_auc_lo"], g["pr_auc_hi"] - g["pr_auc"]],
                         fmt="o-", capsize=3, label=s)
        if has_delta and s != "static":
            axes[1].errorbar(x, g["delta_pr_auc_vs_static"],
                             yerr=[g["delta_pr_auc_vs_static"] - g["delta_lo"],
                                   g["delta_hi"] - g["delta_pr_auc_vs_static"]],
                             fmt="o-", capsize=3, label=s)
    axes[0].set_ylabel("Mean weekly PR-AUC on stream (95% CI)")
    axes[0].set_title("Performance vs. label delay")
    if has_delta:
        axes[1].axhline(0, color="grey", lw=1)
        axes[1].set_ylabel("PR-AUC gain over static (paired 95% CI)")
        axes[1].set_title("Benefit of updating vs. label delay")
    for ax in axes:
        ax.set_xticks(range(len(delays)))
        ax.set_xticklabels([f"{d:g}" for d in delays])
        ax.set_xlabel("Label delay (days)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = build_parser()
    ap.set_defaults(out_dir="./outputs/sweep")
    ap.add_argument("--delays", default="0,15,30,60", help="comma-separated label delays in days")
    ap.add_argument("--n_boot", type=int, default=500)
    args, _ = ap.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)
    strategies = parse_strategies(args.strategies)
    delays = [float(d) for d in args.delays.split(",")]
    rng = np.random.default_rng(SEED)

    trainer = load_trainer(args)   # load once; its model cache is shared across delays
    y_all = trainer.df["isFraud"].values
    results, windows, events = [], [], []
    for delay in delays:
        args.label_delay = delay
        res = run_strategies(trainer, strategies, args)
        sl = res["stream_slice"]
        print(f"\nBootstrapping ({args.n_boot} reps) for label delay {delay:g}d ...")
        results += summarise(delay, res, y_all[sl], trainer.day[sl], args.step_days,
                             args.n_boot, rng)
        # keep raw predictions so the bootstrap can be redone without retraining
        np.savez_compressed(os.path.join(args.out_dir, f"preds_delay{delay:g}.npz"),
                            y=y_all[sl], day=trainer.day[sl],
                            **{f"{s}_p": p for s, (p, _) in res["preds"].items()},
                            **{f"{s}_thr": t for s, (_, t) in res["preds"].items()})
        windows.append(res["by_window"].assign(label_delay=delay))
        events.append(res["events"].assign(label_delay=delay))

        # save after every delay so a crash late in the sweep keeps earlier results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(args.out_dir, "sweep_results.csv"), index=False)
        pd.concat(windows).to_csv(os.path.join(args.out_dir, "sweep_by_window.csv"), index=False)
        pd.concat(events).to_csv(os.path.join(args.out_dir, "sweep_events.csv"), index=False)

    plot(results_df, os.path.join(args.out_dir, "sweep_label_delay.png"))
    cols = ["label_delay", "strategy", "n_retrains", "pr_auc", "pr_auc_lo", "pr_auc_hi",
            "delta_pr_auc_vs_static", "delta_lo", "delta_hi", "p_delta_le_0"]
    print("\n=== Sweep results ===\n"
          + results_df[[c for c in cols if c in results_df]].round(4).to_string(index=False))
    print(f"\nDone. Outputs are in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
