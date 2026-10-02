"""
Equal-budget comparison: does scheduled retraining win only because it retrains more often?
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Both families are run over a range of settings, so that their numbers of retrains overlap:
  schedule   retrain every P days (7 ... 224), on all labelled history or the last 60 days
  trigger    delayed-label PR-AUC trigger (tolerance x gap x reference x window, 60 settings)
             and the label-free score-PSI trigger (4 thresholds)
Two analyses:
  matched    each trigger setting with k >= 1 retrains is paired with the schedule that uses
             the SAME training window and the largest number of retrains that is <= k (ties: the
             shortest period). The schedule never gets more retrains than the trigger, so the
             comparison is conservative in the trigger's favour. Difference = schedule - trigger
             in mean weekly PR-AUC, with a paired bootstrap interval over weeks.
  frontier   best mean weekly PR-AUC each family reaches with at most k retrains (both families
             chosen with hindsight, so the comparison is symmetric).

Usage:
  python equal_budget.py --data_dir ./data --delays 0,30                     # IEEE-CIS
  python equal_budget.py --data_dir data/sparkov --stream_start_frac 0.5 --out_dir outputs/sparkov/equal_budget
Outputs (in --out_dir, default ./outputs/equal_budget):
  eb_runs.csv, eb_weekly.csv, eb_matched.csv, eb_summary.csv, eb_frontier.csv, eb_frontier.png
"""

import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resume_utils import check_settings, load_saved
from retraining_simulation import build_parser, load_trainer, simulate

PERIODS = (7, 14, 21, 28, 35, 42, 56, 70, 84, 112, 140, 168, 224)
WINDOWS = {"all": 0, "60d": 60}


def configs(stream_days):
    """(name, family, window, strategy, settings)."""
    out = []
    for w, p in itertools.product(WINDOWS, PERIODS):
        if p < stream_days:
            strategy = "periodic_expanding" if w == "all" else "periodic_sliding"
            out.append((f"schedule_{w}_{p}", "schedule", w, strategy,
                        {"period_days": p, "window_days": 60}))
    for w, tol, gap, ref in itertools.product(WINDOWS, (0.05, 0.1, 0.15, 0.2, 0.3), (7, 14, 28),
                                              ("val", "first")):
        out.append((f"perf_{w}_{tol}_{gap}_{ref}", "trigger", w, "drift_performance",
                    {"perf_tol": tol, "monitor_days": 14, "cooldown_days": gap, "perf_ref": ref,
                     "drift_window_days": WINDOWS[w]}))
    for thr in (0.02, 0.05, 0.1, 0.2):     # the score-PSI trigger always trains on all history
        out.append((f"psi_all_{thr}", "trigger", "all", "drift_score_psi",
                    {"psi_threshold": thr, "cooldown_days": 14}))
    return out


def boot_ci(diff, rng, n=2000):
    """95% interval of the mean of weekly differences, resampling weeks."""
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return np.nan, np.nan
    means = rng.choice(diff, (n, len(diff))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def matched(runs, weekly, rng):
    rows = []
    for (delay, window), g in runs.groupby(["label_delay", "window"]):
        sched = g[g["family"] == "schedule"].sort_values(["retrains", "period"])
        for t in g[(g["family"] == "trigger") & (g["retrains"] >= 1)].itertuples():
            cand = sched[sched["retrains"] <= t.retrains]
            if cand.empty:
                continue
            best_k = cand["retrains"].max()
            s = cand[cand["retrains"] == best_k].iloc[0]        # shortest period with that count
            ws = weekly[(weekly["label_delay"] == delay) & (weekly["config"] == s["config"])]
            wt = weekly[(weekly["label_delay"] == delay) & (weekly["config"] == t.config)]
            d = ws.set_index("week")["pr_auc"] - wt.set_index("week")["pr_auc"]
            lo, hi = boot_ci(d.to_numpy(), rng)
            rows.append({"label_delay": delay, "window": window, "trigger": t.config,
                         "trigger_retrains": t.retrains, "trigger_pr": t.mean_pr,
                         "schedule": s["config"], "schedule_retrains": int(best_k),
                         "schedule_pr": s["mean_pr"], "diff": s["mean_pr"] - t.mean_pr,
                         "diff_lo": lo, "diff_hi": hi})
    return pd.DataFrame(rows)


def frontier(runs):
    rows = []
    for (delay, window), g in runs.groupby(["label_delay", "window"]):
        static = g[g["family"] == "static"]["mean_pr"]
        kmax = int(g["retrains"].max())
        for k in range(0, kmax + 1):
            row = {"label_delay": delay, "window": window, "budget": k,
                   "static": float(static.iloc[0]) if len(static) else np.nan}
            for fam in ("schedule", "trigger"):
                f = g[(g["family"] == fam) & (g["retrains"] <= k)]
                row[fam] = float(f["mean_pr"].max()) if len(f) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot(runs, out_path):
    keys = sorted(runs.groupby(["label_delay", "window"]).groups)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.8), squeeze=False)
    for ax, (delay, window) in zip(axes[0], keys):
        g = runs[(runs["label_delay"] == delay) & (runs["window"] == window)]
        for fam, colour, marker in (("trigger", "#eb6834", "o"), ("schedule", "#2a78d6", "s")):
            f = g[g["family"] == fam]
            ax.scatter(f["retrains"], f["mean_pr"], s=22, color=colour, marker=marker,
                       alpha=0.55, label=fam, edgecolor="white", linewidth=0.5)
            best = f.groupby("retrains")["mean_pr"].max().sort_index().cummax()
            ax.step(best.index, best.values, where="post", color=colour, lw=2)
        st = g[g["family"] == "static"]["mean_pr"]
        if len(st):
            ax.axhline(st.iloc[0], color="#52514e", lw=1, ls="--", label="static")
        ax.set_title(f"delay {delay:g} d, window: {window}", fontsize=10)
        ax.set_xlabel("number of retrains")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("mean weekly PR-AUC")
    axes[0][0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = build_parser()
    ap.set_defaults(out_dir="./outputs/equal_budget")
    ap.add_argument("--delays", default="0,30")
    ap.add_argument("--resume", action="store_true",
                    help="keep label delays already saved in out_dir and run only the rest")
    args, _ = ap.parse_known_args()
    args.verbose = False
    os.makedirs(args.out_dir, exist_ok=True)
    check_settings(args.out_dir, args)

    trainer = load_trainer(args)
    day = trainer.day
    t0 = float(day[int(len(day) * args.stream_start_frac)])
    steps = np.arange(t0, day.max(), args.step_days)
    grid = configs(float(day.max() - t0))
    print(f"Stream: day {t0:.1f} to {day.max():.1f}; {len(grid)} configurations per delay")

    runs = load_saved(args.out_dir, "eb_runs.csv", args.resume)
    weekly = load_saved(args.out_dir, "eb_weekly.csv", args.resume)
    done = set()
    if len(runs):
        n = runs.groupby("label_delay")["config"].nunique()
        done = {float(d) for d, k in n.items() if k == len(grid) + 1}
        runs = runs[runs["label_delay"].isin(done)]
        weekly = weekly[weekly["label_delay"].isin(done)]
        print(f"Resuming: label delays already done: {sorted(done)}")
    runs, weekly = runs.to_dict("records"), weekly.to_dict("records")

    for delay in [float(d) for d in args.delays.split(",")]:
        if delay in done:
            continue
        args.label_delay = delay
        print(f"\n=== label delay {delay:g} d ===", flush=True)
        initial = trainer.fit(-np.inf, t0 - delay, trained_at=t0)
        todo = [("static", "static", "all", "static", {})] + grid
        for i, (name, family, window, strategy, settings) in enumerate(todo, 1):
            a = argparse.Namespace(**vars(args))
            for k, v in settings.items():
                setattr(a, k, v)
            _, _, windows, events, _ = simulate(strategy, initial, trainer, steps, a)
            w = pd.DataFrame(windows)
            runs.append({"label_delay": delay, "config": name, "family": family, "window": window,
                         "strategy": strategy, "period": settings.get("period_days", np.nan),
                         "mean_pr": float(w["pr_auc"].mean()), "min_pr": float(w["pr_auc"].min()),
                         "retrains": len(events)})
            weekly += [{"label_delay": delay, "config": name, "week": j, "pr_auc": p}
                       for j, p in enumerate(w["pr_auc"])]
            if family == "static":       # static serves both window groups
                runs.append({**runs[-1], "window": "60d"})
            print(f"  [{i}/{len(todo)}] {name:28s} PR-AUC {w['pr_auc'].mean():.4f}  "
                  f"retrains {len(events)}", flush=True)
        pd.DataFrame(runs).to_csv(os.path.join(args.out_dir, "eb_runs.csv"), index=False)
        pd.DataFrame(weekly).to_csv(os.path.join(args.out_dir, "eb_weekly.csv"), index=False)

    runs, weekly = pd.DataFrame(runs), pd.DataFrame(weekly)
    weekly = weekly.drop_duplicates(["label_delay", "config", "week"])
    m = matched(runs, weekly, np.random.default_rng(0))
    m.to_csv(os.path.join(args.out_dir, "eb_matched.csv"), index=False)
    summary = (m.groupby(["label_delay", "window"])
                .apply(lambda g: pd.Series({
                    "pairs": len(g),
                    "schedule_better": int((g["diff"] > 0).sum()),
                    "schedule_reliably_better": int((g["diff_lo"] > 0).sum()),
                    "trigger_reliably_better": int((g["diff_hi"] < 0).sum()),
                    "mean_diff": g["diff"].mean(), "median_diff": g["diff"].median(),
                    "mean_trigger_retrains": g["trigger_retrains"].mean(),
                    "mean_schedule_retrains": g["schedule_retrains"].mean()}),
                       include_groups=False)
                .reset_index())
    summary.to_csv(os.path.join(args.out_dir, "eb_summary.csv"), index=False)
    fr = frontier(runs)
    fr.to_csv(os.path.join(args.out_dir, "eb_frontier.csv"), index=False)
    plot(runs, os.path.join(args.out_dir, "eb_frontier.png"))

    pd.set_option("display.width", 200)
    print("\n=== Matched (schedule with <= the trigger's retrains, same window) ===\n"
          + summary.round(4).to_string(index=False))
    print(f"\nDone. Outputs are in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
