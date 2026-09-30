"""
IEEE-CIS Fraud Detection: seed robustness of the key retraining comparisons
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

The bootstrap CIs elsewhere only cover evaluation-sample noise. This re-runs the main
strategies with several LightGBM random seeds (row/feature subsampling changes) to see
whether the differences between strategies survive training randomness.

Usage:
  python seed_robustness.py --data_dir ./data --seeds 42,1,2,3,4 --delays 0,30

Outputs (in --out_dir, default ./outputs/seeds):
  seed_runs.csv, seed_summary.csv, seed_diffs.csv, seed_robustness.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resume_utils import check_settings, load_saved
from retraining_simulation import build_parser, load_trainer, simulate

CONFIGS = {
    "static": {"strategy": "static"},
    "expanding_14": {"strategy": "periodic_expanding", "period_days": 14},
    "sliding_14_60": {"strategy": "periodic_sliding", "period_days": 14, "window_days": 60},
    "sliding_28_30": {"strategy": "periodic_sliding", "period_days": 28, "window_days": 30},
    "drift_default": {"strategy": "drift_performance", "perf_tol": 0.15, "monitor_days": 14,
                      "cooldown_days": 14, "perf_ref": "val", "drift_window_days": 0},
    "drift_w60": {"strategy": "drift_performance", "perf_tol": 0.05, "monitor_days": 14,
                  "cooldown_days": 14, "perf_ref": "val", "drift_window_days": 60},
}
# (a, b): difference a - b is reported per seed
PAIRS = [("expanding_14", "static"), ("sliding_14_60", "static"), ("sliding_28_30", "static"),
         ("drift_default", "static"), ("drift_w60", "static"),
         ("drift_default", "sliding_14_60"), ("drift_w60", "sliding_14_60"),
         ("sliding_14_60", "expanding_14")]


def main():
    ap = build_parser()
    ap.set_defaults(out_dir="./outputs/seeds")
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--delays", default="0,30")
    ap.add_argument("--resume", action="store_true",
                    help="keep seed/delay runs already saved in out_dir and run only the rest")
    args, _ = ap.parse_known_args()
    args.verbose = False
    os.makedirs(args.out_dir, exist_ok=True)
    check_settings(args.out_dir, args)

    trainer = load_trainer(args)
    day = trainer.day
    t0 = float(day[int(len(day) * args.stream_start_frac)])
    steps = np.arange(t0, day.max(), args.step_days)

    # a seed/delay pair counts as done only when every configuration was saved for it
    saved = load_saved(args.out_dir, "seed_runs.csv", args.resume)
    done = set()
    if len(saved):
        n = saved.groupby(["seed", "label_delay"])["config"].nunique()
        done = {(int(s), float(d)) for (s, d), k in n.items() if k == len(CONFIGS)}
        saved = saved[[(int(s), float(d)) in done
                       for s, d in zip(saved["seed"], saved["label_delay"])]]
        print(f"Resuming: {len(done)} seed/delay pairs already done")
    runs = saved.to_dict("records")

    for seed in [int(s) for s in args.seeds.split(",")]:
        pending = [float(d) for d in args.delays.split(",") if (seed, float(d)) not in done]
        if not pending:
            continue
        trainer.set_seed(seed)
        for delay in pending:
            args.label_delay = delay
            print(f"\n=== seed {seed}, label delay {delay:g}d ===")
            initial = trainer.fit(-np.inf, t0 - delay, trained_at=t0)
            for name, cfg in CONFIGS.items():
                a = argparse.Namespace(**vars(args))
                for k, v in cfg.items():
                    if k != "strategy":
                        setattr(a, k, v)
                _, _, windows, events, _ = simulate(cfg["strategy"], initial, trainer, steps, a)
                pr = pd.DataFrame(windows)["pr_auc"]
                runs.append({"seed": seed, "label_delay": delay, "config": name,
                             "mean_pr": float(pr.mean()), "min_pr": float(pr.min()),
                             "retrains": len(events)})
                print(f"  {name:14s} mean weekly PR-AUC {pr.mean():.4f}  retrains {len(events)}")
            pd.DataFrame(runs).to_csv(os.path.join(args.out_dir, "seed_runs.csv"), index=False)

    runs = pd.DataFrame(runs)
    summary = (runs.groupby(["label_delay", "config"], sort=False)
                   .agg(mean_pr=("mean_pr", "mean"), sd=("mean_pr", "std"),
                        min_seed=("mean_pr", "min"), max_seed=("mean_pr", "max"),
                        retrains=("retrains", "mean")).reset_index())
    summary.to_csv(os.path.join(args.out_dir, "seed_summary.csv"), index=False)

    wide = runs.pivot_table(index=["label_delay", "seed"], columns="config", values="mean_pr")
    diffs = []
    for delay, g in wide.groupby(level="label_delay"):
        for a, b in PAIRS:
            d = g[a] - g[b]
            diffs.append({"label_delay": delay, "comparison": f"{a} - {b}",
                          "mean_diff": d.mean(), "sd": d.std(), "min": d.min(), "max": d.max(),
                          "seeds_positive": f"{int((d > 0).sum())}/{len(d)}"})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(os.path.join(args.out_dir, "seed_diffs.csv"), index=False)

    delays = sorted(runs["label_delay"].unique())
    fig, axes = plt.subplots(1, len(delays), figsize=(6 * len(delays), 5), sharey=True)
    for ax, delay in zip(np.atleast_1d(axes), delays):
        g = runs[runs["label_delay"] == delay]
        names = list(CONFIGS)
        for i, n in enumerate(names):
            v = g[g["config"] == n]["mean_pr"]
            ax.scatter(np.full(len(v), i), v, alpha=0.7)
            ax.hlines(v.mean(), i - 0.3, i + 0.3, color="black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Label delay {delay:g} d (dots = seeds, bar = mean)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Mean weekly PR-AUC (test period)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "seed_robustness.png"), dpi=150)
    plt.close()

    pd.set_option("display.width", 200)
    print("\n=== Across seeds ===\n" + summary.round(4).to_string(index=False))
    print("\n=== Paired differences across seeds ===\n" + diffs.round(4).to_string(index=False))
    print(f"\nDone. Outputs are in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
