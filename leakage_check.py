"""
IEEE-CIS client-identity leakage check.
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

IEEE-CIS labels a chargeback's card as fraud and then also labels later transactions linked
to the same account, email or address as fraud. The same client can therefore be fraud both
in a model's training data and in the weeks it scores, and a model may score well partly by
recognising known fraudulent clients. This script measures how much of the reported PR-AUC,
and of the retraining gain, depends on that.

Client ID (the usual Kaggle reconstruction): card1 + addr1 + (transaction day - D1), i.e. the
card, the billing region and the day the account was opened. Rows missing addr1 or D1 get no
ID. Every stream row is put in one group, using only labels available at its time
(transaction day - label delay), exactly like the models:

  known_fraud  the client already had a fraud whose label had arrived
  seen_clean   the client appeared in labelled data, never with a fraud label yet
  new          the client never appeared in labelled data
  no_id        client ID cannot be built

Each strategy's mean weekly PR-AUC (and its gain over the static model, with the paired
day-block bootstrap of sweep_label_delay.py) is then recomputed on:
  all rows | without known_fraud clients | new clients only

  python leakage_check.py --data_dir ./data --label_delay 30
Outputs (in --out_dir, default ./outputs/leakage): leakage_groups.csv, leakage_results.csv,
leakage_check.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retraining_simulation import build_parser, load_trainer, run_strategies
from sweep_label_delay import mean_window_pr_auc

STRATS = ["static", "periodic_sliding", "periodic_expanding"]
NAMES = {"static": "static", "periodic_sliding": "schedule, last 60 days",
         "periodic_expanding": "schedule, all history"}
SUBSETS = ["all rows", "without known-fraud clients", "new clients only"]


def client_groups(df, day, delay):
    """Group of every row, using only labels available delay days before it."""
    ok = df["card1"].notna() & df["addr1"].notna() & df["D1"].notna()
    uid = (df["card1"].astype("float64").astype("Int64").astype(str) + "_"
           + df["addr1"].astype("float64").astype("Int64").astype(str) + "_"
           + np.floor(day - df["D1"].astype("float64")).astype("Int64").astype(str))
    t = pd.DataFrame({"uid": uid.where(ok), "day": day, "fraud": df["isFraud"].values})
    first_seen = t.groupby("uid")["day"].transform("min")
    first_fraud = t["day"].where(t["fraud"] == 1).groupby(t["uid"]).transform("min")
    known = day - delay
    g = np.where(~ok, "no_id",
                 np.where(first_fraud.le(known), "known_fraud",
                          np.where(first_seen.le(known), "seen_clean", "new")))
    return g


def subset_scores(y, day, preds, mask, step_days, t0, n_boot, rng):
    """Point estimate and bootstrap of mean weekly PR-AUC on the rows in mask."""
    win = np.floor((day - t0) / step_days).astype(int)
    idx = np.where(mask)[0]
    ys, ds, ws = y[idx], day[idx], win[idx]
    point = {s: mean_window_pr_auc(ys, p[idx], [np.where(ws == w)[0] for w in np.unique(ws)])
             for s, (p, _) in preds.items()}
    sub = {s: (p[idx], thr[idx]) for s, (p, thr) in preds.items()}
    return point, _boot(ys, ds, ws, sub, n_boot, rng)


def _boot(y, day, win, preds, n_boot, rng):
    """Paired day-block bootstrap within each week (as in sweep_label_delay.bootstrap), with
    weeks counted from the stream start rather than from the subset's first row."""
    d = np.floor(day).astype(int)
    blocks = [[np.where((win == w) & (d == u))[0] for u in np.unique(d[win == w])]
              for w in np.unique(win)]
    out = {s: np.empty(n_boot) for s in preds}
    for b in range(n_boot):
        win_idx = [np.concatenate([bl[k] for k in rng.integers(0, len(bl), len(bl))])
                   for bl in blocks]
        for s, (p, _) in preds.items():
            out[s][b] = mean_window_pr_auc(y, p, win_idx)
    return out


def main():
    ap = build_parser()
    ap.set_defaults(out_dir="./outputs/leakage", strategies=",".join(STRATS))
    ap.add_argument("--n_boot", type=int, default=500)
    args, _ = ap.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    trainer = load_trainer(args)
    res = run_strategies(trainer, STRATS, args)
    sl = res["stream_slice"]
    df = trainer.df
    day_all = trainer.day
    groups = client_groups(df, day_all, args.label_delay)[sl]
    y = df["isFraud"].values[sl]
    day = day_all[sl]
    t0 = day[0]
    n = min(len(p) for p, _ in res["preds"].values())
    y, day, groups = y[:n], day[:n], groups[:n]
    preds = {s: (p[:n], thr[:n]) for s, (p, thr) in res["preds"].items()}

    # composition of the stream
    comp = (pd.DataFrame({"group": groups, "fraud": y})
            .groupby("group")["fraud"].agg(rows="size", frauds="sum", fraud_rate="mean")
            .reindex(["known_fraud", "seen_clean", "new", "no_id"]))
    comp["share_of_rows"] = comp["rows"] / comp["rows"].sum()
    comp["share_of_frauds"] = comp["frauds"] / comp["frauds"].sum()
    comp.to_csv(os.path.join(args.out_dir, "leakage_groups.csv"))
    print("\n=== Stream rows by client group (label delay "
          f"{args.label_delay:g} d) ===\n" + comp.round(4).to_string())

    masks = {"all rows": np.ones(n, bool),
             "without known-fraud clients": groups != "known_fraud",
             "new clients only": groups == "new"}
    rows = []
    for name, m in masks.items():
        point, boots = subset_scores(y, day, preds, m, args.step_days, t0, args.n_boot, rng)
        for s in STRATS:
            diff = boots[s] - boots["static"]
            rows.append({"subset": name, "strategy": s, "rows": int(m.sum()),
                         "frauds": int(y[m].sum()), "pr_auc": point[s],
                         "pr_auc_lo": np.percentile(boots[s], 2.5),
                         "pr_auc_hi": np.percentile(boots[s], 97.5),
                         "gain_vs_static": point[s] - point["static"],
                         "gain_lo": np.percentile(diff, 2.5),
                         "gain_hi": np.percentile(diff, 97.5)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out_dir, "leakage_results.csv"), index=False)
    print("\n=== Mean weekly PR-AUC by subset ===\n" + out.round(4).to_string(index=False))

    # figure: PR-AUC per subset (left), gain over static per subset (right)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"static": "#8A8A8A", "periodic_sliding": "#EB6834", "periodic_expanding": "#2A78D6"}
    x = np.arange(len(SUBSETS))
    for k, s in enumerate(STRATS):
        r = out[out["strategy"] == s].set_index("subset").loc[SUBSETS]
        axes[0].bar(x + (k - 1) * 0.26, r["pr_auc"], 0.24, color=colors[s], label=NAMES[s],
                    yerr=[r["pr_auc"] - r["pr_auc_lo"], r["pr_auc_hi"] - r["pr_auc"]], capsize=3)
        if s != "static":
            axes[1].bar(x + (k - 1.5) * 0.3, r["gain_vs_static"], 0.28, color=colors[s],
                        label=NAMES[s], capsize=3,
                        yerr=[r["gain_vs_static"] - r["gain_lo"], r["gain_hi"] - r["gain_vs_static"]])
    for ax, t in zip(axes, ["Mean weekly PR-AUC", "PR-AUC gain over static"]):
        ax.set_xticks(x)
        ax.set_xticklabels(["all rows", "without known-\nfraud clients", "new clients\nonly"])
        ax.set_title(t, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].axhline(0, color="#888", lw=1)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"IEEE-CIS client-identity check (label delay {args.label_delay:g} days)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "leakage_check.png"), dpi=200)
    print(f"\nDone. Outputs are in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
