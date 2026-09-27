"""
IEEE-CIS Fraud Detection: walk-forward tuning of retraining strategies
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Why: in tune_drift_trigger.py the drift trigger fired at most once in the fixed tuning
period, so its settings could not be ranked. Here the whole grid runs as one long stream
from --tune_start_frac (day ~56) to the end, and the evaluation period (day ~101 onward)
is split into folds of --fold_weeks. At the start of each fold, every family re-selects its
configuration using only weekly windows whose labels have matured by then
(window end <= fold start - label_delay), objective:
    mean weekly PR-AUC - --retrain_cost * retrains per 4 weeks
The selected configuration's own trajectory is then scored on the fold (assumes the
system had been running that configuration; the switching transient is ignored).

Families: static | schedule (periodic_expanding + periodic_sliding) | drift_performance
Grid as tune_drift_trigger.py, minus 90-day windows (never competitive there).

Usage:
  python walkforward_tuning.py --data_dir ./data --delays 0,15,30

Outputs (in --out_dir, default ./outputs/walkforward):
  wf_folds.csv, wf_summary.csv, wf_weekly.csv, wf_walkforward.png
"""

import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retraining_simulation import build_parser, load_trainer, simulate
from sweep_label_delay import bootstrap, mean_window_pr_auc, window_ids

SEED = 42
FAMILIES = ["static", "schedule", "drift_performance"]


def grid():
    configs = [{"family": "static", "strategy": "static"}]
    configs += [{"family": "schedule", "strategy": "periodic_expanding", "period_days": p}
                for p in (7, 14, 28)]
    configs += [{"family": "schedule", "strategy": "periodic_sliding", "period_days": p,
                 "window_days": w} for p, w in itertools.product((7, 14, 28), (30, 60))]
    configs += [{"family": "drift_performance", "strategy": "drift_performance",
                 "perf_tol": t, "monitor_days": m, "cooldown_days": c, "perf_ref": r,
                 "drift_window_days": w}
                for t, m, c, r, w in itertools.product((0.05, 0.1, 0.15, 0.2, 0.3), (7, 14, 28),
                                                      (7, 14, 28), ("val", "first"), (0, 60))]
    for i, c in enumerate(configs):
        c["config_id"] = i
    return configs


def describe(cfg):
    skip = ("family", "config_id")
    return ", ".join(f"{k}={v}" for k, v in cfg.items() if k not in skip)


def run_grid(trainer, configs, origin, steps, eval_rows, args):
    """Simulate every config over the long stream; keep eval-period scores and weekly stats."""
    initial = trainer.fit(-np.inf, origin - args.label_delay, trained_at=origin)
    out = {}
    for i, cfg in enumerate(configs):
        a = argparse.Namespace(**vars(args))
        for k, v in cfg.items():
            if k not in ("family", "strategy", "config_id"):
                setattr(a, k, v)
        p, _, windows, events, _ = simulate(cfg["strategy"], initial, trainer, steps, a)
        w = pd.DataFrame(windows)[["day_start", "pr_auc"]]
        ev = np.array([e["day"] for e in events])
        w["retrains"] = [int(((ev >= d) & (ev < d + args.step_days)).sum()) for d in w["day_start"]]
        out[cfg["config_id"]] = (p[eval_rows].astype(np.float32), w)
        if (i + 1) % 50 == 0 or i + 1 == len(configs):
            print(f"  {i + 1}/{len(configs)} configs ({len(trainer.cache)} models trained)")
    return out


def objective(w, cost):
    if len(w) == 0:
        return np.nan
    return w["pr_auc"].mean() - cost * w["retrains"].sum() / len(w) * 4


def walk_forward(configs, runs, folds, args):
    """Per fold and family: choose on matured past windows, score on the fold."""
    rows = []
    for f_start, f_end in folds:
        known_until = f_start - args.label_delay
        past_obj, fold_obj = {}, {}
        for cfg in configs:
            w = runs[cfg["config_id"]][1]
            end = w["day_start"] + args.step_days
            past_obj[cfg["config_id"]] = objective(w[end <= known_until], args.retrain_cost)
            fold_obj[cfg["config_id"]] = objective(
                w[(w["day_start"] >= f_start - 1e-9) & (w["day_start"] < f_end - 1e-9)],
                args.retrain_cost)
        for fam in FAMILIES:
            ids = [c["config_id"] for c in configs if c["family"] == fam]
            po = pd.Series({i: past_obj[i] for i in ids})
            fo = pd.Series({i: fold_obj[i] for i in ids})
            # no matured history yet -> fall back to the family's first (default-ish) config
            chosen = ids[0] if po.isna().all() else int(po.idxmax())
            w = runs[chosen][1]
            in_fold = w[(w["day_start"] >= f_start - 1e-9) & (w["day_start"] < f_end - 1e-9)]
            rows.append({"label_delay": args.label_delay, "fold_start": f_start,
                         "fold_end": f_end, "family": fam, "config_id": chosen,
                         "params": describe(next(c for c in configs if c["config_id"] == chosen)),
                         "n_past_windows": int(sum(((runs[chosen][1]["day_start"] + args.step_days)
                                                    <= known_until))),
                         "fold_mean_pr": float(in_fold["pr_auc"].mean()),
                         "fold_retrains": int(in_fold["retrains"].sum()),
                         "distinct_past_scores": int(po.round(5).nunique()),
                         "rank_corr_past_vs_fold": float(po.rank().corr(fo.rank()))
                         if len(ids) > 2 else np.nan,
                         "hindsight_best_fold_mean_pr": float(max(
                             runs[i][1][(runs[i][1]["day_start"] >= f_start - 1e-9)
                                        & (runs[i][1]["day_start"] < f_end - 1e-9)]["pr_auc"].mean()
                             for i in ids))})
    return pd.DataFrame(rows)


def stitch(folds_df, runs, eval_day, folds):
    """Per-transaction eval-period scores of each family's walk-forward choices."""
    preds = {}
    for fam in FAMILIES:
        p = np.empty(len(eval_day), dtype=np.float32)
        for f_start, f_end in folds:
            cid = int(folds_df[(folds_df["family"] == fam)
                               & (folds_df["fold_start"] == f_start)]["config_id"].iloc[0])
            m = (eval_day >= f_start) & (eval_day < f_end)
            p[m] = runs[cid][0][m]
        preds[fam] = (p, np.full(len(p), 0.5))   # threshold unused (PR-AUC only)
    return preds


def plot(summary, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    delays = sorted(summary["label_delay"].unique())
    pos = {d: i for i, d in enumerate(delays)}
    for off, fam in zip((-0.15, 0, 0.15), FAMILIES):
        g = summary[summary["family"] == fam].sort_values("label_delay")
        x = g["label_delay"].map(pos) + off
        axes[0].errorbar(x, g["mean_pr"], yerr=[g["mean_pr"] - g["lo"], g["hi"] - g["mean_pr"]],
                         fmt="o-", capsize=3, label=fam)
    g = summary[summary["family"] == "drift_performance"].sort_values("label_delay")
    x = g["label_delay"].map(pos)
    axes[1].errorbar(x, g["delta_vs_schedule"],
                     yerr=[g["delta_vs_schedule"] - g["delta_vs_schedule_lo"],
                           g["delta_vs_schedule_hi"] - g["delta_vs_schedule"]],
                     fmt="o-", capsize=3, color="C2")
    axes[1].axhline(0, color="grey", lw=1)
    axes[0].set_ylabel("Mean weekly PR-AUC, eval period (95% CI)")
    axes[0].set_title("Walk-forward tuned strategies")
    axes[1].set_ylabel("Drift trigger minus schedule (paired 95% CI)")
    axes[1].set_title("Does the tuned drift trigger beat the tuned schedule?")
    for ax in axes:
        ax.set_xticks(range(len(delays)))
        ax.set_xticklabels([f"{d:g}" for d in delays])
        ax.set_xlabel("Label delay (days)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = build_parser()
    ap.set_defaults(out_dir="./outputs/walkforward")
    ap.add_argument("--delays", default="0,15,30")
    ap.add_argument("--tune_start_frac", type=float, default=0.35)
    ap.add_argument("--fold_weeks", type=int, default=4)
    ap.add_argument("--retrain_cost", type=float, default=0.002,
                    help="PR-AUC penalty per retrain (per 4 weeks, averaged)")
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="tiny grid for a quick pipeline check")
    args, _ = ap.parse_known_args()
    args.verbose = False
    os.makedirs(args.out_dir, exist_ok=True)

    trainer = load_trainer(args)
    day, y_all = trainer.day, trainer.df["isFraud"].values
    origin = float(day[int(len(day) * args.tune_start_frac)])
    steps = np.arange(origin, day.max(), args.step_days)
    eval_start = float(steps[np.searchsorted(steps, day[int(len(day) * args.stream_start_frac)])])
    eval_steps = steps[steps >= eval_start]
    fold_starts = eval_steps[::args.fold_weeks]
    folds = list(zip(fold_starts, list(fold_starts[1:]) + [np.inf]))
    eval_rows = slice(trainer.idx(eval_start) - trainer.idx(origin), None)
    ev = slice(trainer.idx(eval_start), len(day))
    print(f"Stream from day {origin:.1f}; evaluation from day {eval_start:.1f} in {len(folds)} "
          f"folds of {args.fold_weeks} weeks")

    configs = grid()
    if args.smoke:
        configs = [c for c in configs if c["config_id"] in (0, 3, 9, 10, 11, 12)]

    all_folds, summary, weekly = [], [], []
    for delay in [float(d) for d in args.delays.split(",")]:
        args.label_delay = delay
        print(f"\n=== label delay {delay:g}d ===")
        trainer.cache.clear()
        trainer.pred_cache.clear()
        runs = run_grid(trainer, configs, origin, steps, eval_rows, args)
        folds_df = walk_forward(configs, runs, folds, args)
        all_folds.append(folds_df)

        preds = stitch(folds_df, runs, day[ev], folds)
        win = window_ids(day[ev], args.step_days)
        win_idx = [np.where(win == w)[0] for w in np.unique(win)]
        point = {f: mean_window_pr_auc(y_all[ev], p, win_idx) for f, (p, _) in preds.items()}
        boots = bootstrap(y_all[ev], day[ev], preds, args.step_days, args.n_boot,
                          np.random.default_rng(SEED))
        for fam in FAMILIES:
            b = boots[fam][:, 0]
            row = {"label_delay": delay, "family": fam, "mean_pr": point[fam],
                   "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5),
                   "retrains": int(folds_df[folds_df["family"] == fam]["fold_retrains"].sum())}
            for ref in ("static", "schedule"):
                d = b - boots[ref][:, 0]
                row[f"delta_vs_{ref}"] = point[fam] - point[ref]
                row[f"delta_vs_{ref}_lo"] = np.percentile(d, 2.5)
                row[f"delta_vs_{ref}_hi"] = np.percentile(d, 97.5)
            summary.append(row)
        for fam, (p, _) in preds.items():
            for w, i in enumerate(win_idx):
                weekly.append({"label_delay": delay, "family": fam, "week": w,
                               "day_start": float(day[ev][i].min()),
                               "pr_auc": mean_window_pr_auc(y_all[ev], p, [i])})

        pd.concat(all_folds).to_csv(os.path.join(args.out_dir, "wf_folds.csv"), index=False)
        pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "wf_summary.csv"), index=False)
        pd.DataFrame(weekly).to_csv(os.path.join(args.out_dir, "wf_weekly.csv"), index=False)

    summary = pd.DataFrame(summary)
    plot(summary, os.path.join(args.out_dir, "wf_walkforward.png"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print("\n=== Folds ===\n" + pd.concat(all_folds)[
        ["label_delay", "fold_start", "family", "params", "n_past_windows", "fold_mean_pr",
         "fold_retrains", "distinct_past_scores", "rank_corr_past_vs_fold"]].round(3).to_string(index=False))
    print("\n=== Summary ===\n" + summary.round(4).to_string(index=False))
    print(f"\nDone. Outputs are in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
