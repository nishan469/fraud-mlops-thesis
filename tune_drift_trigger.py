"""
IEEE-CIS Fraud Detection: tuning the drift_performance retraining trigger
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Honest protocol (no tuning on the test stream):
  tune period : rows --tune_start_frac .. --stream_start_frac  (default days ~56-101)
  test period : rows --stream_start_frac .. end                (days ~101-183, same as the
                                                               sweep, so numbers compare)
Every configuration below is simulated on both periods. For each strategy family the
configuration with the best tune-period objective
    mean weekly PR-AUC - --retrain_cost * n_retrains
is selected, then reported on the test period with paired bootstrap CIs vs. static and
vs. the selected periodic_sliding schedule (the strongest fixed baseline so far).

Grid:
  periodic_expanding  period {7,14,28}
  periodic_sliding    period {7,14,28} x window {30,60,90}
  drift_performance   tol {.05,.1,.15,.2,.3} x monitor {7,14,28} x cooldown {7,14,28}
                      x reference {val, first} x training window {all, 60}

Models and predictions are cached in the Trainer, so the grid mostly re-uses the same
few dozen models. Usage:
  python tune_drift_trigger.py --data_dir ./data --label_delay 30

Outputs (in --out_dir, default ./outputs/tuning):
  tune_all_configs.csv, tune_selected.csv, tune_selected_by_window.csv, tune_tradeoff.png
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
from sweep_label_delay import bootstrap

SEED = 42
FAMILIES = ["static", "periodic_expanding", "periodic_sliding", "drift_performance"]


def grid():
    configs = [{"family": "static"}]
    configs += [{"family": "periodic_expanding", "period_days": p} for p in (7, 14, 28)]
    configs += [{"family": "periodic_sliding", "period_days": p, "window_days": w}
                for p, w in itertools.product((7, 14, 28), (30, 60, 90))]
    configs += [{"family": "drift_performance", "perf_tol": t, "monitor_days": m,
                 "cooldown_days": c, "perf_ref": r, "drift_window_days": w}
                for t, m, c, r, w in itertools.product((0.05, 0.1, 0.15, 0.2, 0.3), (7, 14, 28),
                                                      (7, 14, 28), ("val", "first"), (0, 60))]
    for i, c in enumerate(configs):
        c["config_id"] = i
    return configs


def make_period(trainer, name, start_frac, end_frac, args):
    day = trainer.day
    t0 = float(day[int(len(day) * start_frac)])
    t_end = float(day[int(len(day) * end_frac)]) if end_frac < 1 else np.inf
    steps = np.arange(t0, min(t_end, day.max()), args.step_days)
    print(f"\n{name} period: day {t0:.1f} to {min(t_end, day.max()):.1f} ({len(steps)} steps)")
    initial = trainer.fit(-np.inf, t0 - args.label_delay, trained_at=t0)
    sl = slice(trainer.idx(t0), trainer.idx(t_end) if np.isfinite(t_end) else len(day))
    return {"name": name, "steps": steps, "t_end": t_end, "initial": initial, "slice": sl}


def run_config(cfg, period, trainer, base_args):
    args = argparse.Namespace(**vars(base_args))
    for k, v in cfg.items():
        if k not in ("family", "config_id"):
            setattr(args, k, v)
    p, thr, windows, events, _ = simulate(cfg["family"], period["initial"], trainer,
                                          period["steps"], args, t_end=period["t_end"])
    pr = pd.DataFrame(windows)["pr_auc"]
    return {"mean_pr": float(pr.mean()), "min_pr": float(pr.min()),
            "n_retrains": len(events)}, (p, thr), windows, events


def describe(cfg):
    return ", ".join(f"{k}={v}" for k, v in cfg.items() if k not in ("family", "config_id")) or "-"


def plot_tradeoff(res, selected_ids, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = dict(zip(FAMILIES, plt.cm.tab10.colors))
    rng = np.random.default_rng(0)
    for fam, g in res.groupby("family"):
        jitter = rng.uniform(-0.15, 0.15, len(g))
        ax.scatter(g["test_n_retrains"] + jitter, g["test_mean_pr"], s=14, alpha=0.5,
                   color=colors[fam], label=fam)
    sel = res[res["config_id"].isin(selected_ids)]
    ax.scatter(sel["test_n_retrains"], sel["test_mean_pr"], s=160, facecolors="none",
               edgecolors="black", linewidths=1.5, label="selected on tune period")
    ax.set_xlabel("Number of retrains on test period")
    ax.set_ylabel("Mean weekly PR-AUC on test period")
    ax.set_title("Accuracy vs. retraining cost, all configurations (test period)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = build_parser()
    ap.set_defaults(out_dir="./outputs/tuning")
    ap.add_argument("--tune_start_frac", type=float, default=0.35)
    ap.add_argument("--retrain_cost", type=float, default=0.002,
                    help="objective penalty per retrain, in PR-AUC units")
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="tiny grid for a quick pipeline check")
    args, _ = ap.parse_known_args()
    args.verbose = False
    os.makedirs(args.out_dir, exist_ok=True)

    trainer = load_trainer(args)
    y_all = trainer.df["isFraud"].values
    tune = make_period(trainer, "tune", args.tune_start_frac, args.stream_start_frac, args)
    test = make_period(trainer, "test", args.stream_start_frac, 1.0, args)

    configs = grid()
    if args.smoke:  # one config per family, to check the pipeline end to end
        configs = [next(c for c in configs if c["family"] == f) for f in FAMILIES]
    rows, test_out = [], {}
    for i, cfg in enumerate(configs):
        r_tune, _, _, _ = run_config(cfg, tune, trainer, args)
        r_test, preds, windows, events = run_config(cfg, test, trainer, args)
        test_out[cfg["config_id"]] = (preds, windows, events)
        rows.append({"config_id": cfg["config_id"], "family": cfg["family"],
                     "params": describe(cfg),
                     **{f"tune_{k}": v for k, v in r_tune.items()},
                     **{f"test_{k}": v for k, v in r_test.items()}})
        if (i + 1) % 25 == 0 or i + 1 == len(configs):
            print(f"  {i + 1}/{len(configs)} configs done "
                  f"({len(trainer.cache)} models trained so far)")

    res = pd.DataFrame(rows)
    for p in ("tune", "test"):
        res[f"{p}_objective"] = res[f"{p}_mean_pr"] - args.retrain_cost * res[f"{p}_n_retrains"]
    res.to_csv(os.path.join(args.out_dir, "tune_all_configs.csv"), index=False)

    # selection on the tune period only
    selected = (res.sort_values(["tune_objective", "tune_n_retrains"], ascending=[False, True])
                   .groupby("family", sort=False).head(1).set_index("family").loc[FAMILIES])

    # paired bootstrap on the test period for the selected configs
    sl = test["slice"]
    preds = {fam: test_out[int(r["config_id"])][0] for fam, r in selected.iterrows()}
    boots = bootstrap(y_all[sl], trainer.day[sl], preds, args.step_days, args.n_boot,
                      np.random.default_rng(SEED))
    out = []
    for fam, r in selected.iterrows():
        b = boots[fam][:, 0]
        row = {"family": fam, "params": r["params"],
               "tune_mean_pr": r["tune_mean_pr"], "tune_n_retrains": r["tune_n_retrains"],
               "test_mean_pr": r["test_mean_pr"], "test_min_pr": r["test_min_pr"],
               "test_n_retrains": r["test_n_retrains"],
               "test_lo": np.percentile(b, 2.5), "test_hi": np.percentile(b, 97.5)}
        for ref in ("static", "periodic_sliding"):
            d = b - boots[ref][:, 0]
            row[f"delta_vs_{ref}"] = r["test_mean_pr"] - selected.loc[ref, "test_mean_pr"]
            row[f"delta_vs_{ref}_lo"] = np.percentile(d, 2.5)
            row[f"delta_vs_{ref}_hi"] = np.percentile(d, 97.5)
        out.append(row)
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(args.out_dir, "tune_selected.csv"), index=False)
    pd.concat([pd.DataFrame(test_out[int(r["config_id"])][1]).assign(family=fam)
               for fam, r in selected.iterrows()]).to_csv(
        os.path.join(args.out_dir, "tune_selected_by_window.csv"), index=False)
    plot_tradeoff(res, selected["config_id"].tolist(),
                  os.path.join(args.out_dir, "tune_tradeoff.png"))

    # how well does tune-period ranking transfer to the test period?
    drift = res[res["family"] == "drift_performance"]
    rho = drift["tune_objective"].rank().corr(drift["test_objective"].rank())
    best_test = drift.sort_values("test_objective", ascending=False).iloc[0]

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print("\n=== Selected on tune period, evaluated on test period ===")
    print(out[["family", "params", "test_n_retrains", "test_mean_pr", "test_lo", "test_hi",
               "delta_vs_static", "delta_vs_static_lo", "delta_vs_static_hi",
               "delta_vs_periodic_sliding", "delta_vs_periodic_sliding_lo",
               "delta_vs_periodic_sliding_hi"]].round(4).to_string(index=False))
    print(f"\nSpearman rank correlation tune vs test objective (drift configs): {rho:.2f}")
    print(f"Best drift config in hindsight (test, for reference only): {best_test['params']} "
          f"-> {best_test['test_mean_pr']:.4f} with {best_test['test_n_retrains']} retrains")
    print(f"\nDone. Outputs are in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
