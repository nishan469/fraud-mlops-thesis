"""
Does the framework pick a suitable training window by itself? (automatic window selection)
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

The best training window differed between datasets (last 60 days on IEEE-CIS, all labelled
history on Sparkov and BAF). Here the framework trains one challenger per candidate window at
every retrain, scores them on the same recent unseen data, and the promotion gate keeps the
best (configs: [policy] candidate_windows). It is compared, on the same machine and the same
stream, with the fixed choices and with a model that is never retrained:

  static     initial model on all history, never retrained
  sliding    retrain every 14 days on the last 60 days     (the IEEE-CIS default)
  expanding  retrain every 14 days on all labelled history
  adaptive   retrain every 14 days on 60 days AND all history; the gate keeps the best

All four replays share one data load and one model cache: a window the adaptive run trains
was usually trained by the sliding or expanding run already, so it adds little time.

  python run_window_selection.py --dataset ieee    --in_dir data
  python run_window_selection.py --dataset sparkov --in_dir data/sparkov
  python run_window_selection.py --dataset baf     --in_dir data/baf

Outputs (outputs/<dataset>/window_selection/): <variant>/ (served_weekly.csv, decisions.csv,
summary.json, dashboard.html), comparison.csv, comparison.md, and its own MLflow store
(mlflow.db, mlruns/) so the main registry is untouched.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

from fraud_mlops.config import load_config
from fraud_mlops.dashboard import build_dashboard
from fraud_mlops.data import TransactionStore, load_transactions
from fraud_mlops.pipeline import ReplayRunner

BASE = {"ieee": "configs/default.toml", "sparkov": "configs/sparkov.toml",
        "baf": "configs/baf.toml"}
NEVER = 1e9
VARIANTS = {
    "static": {"train_window_days": 0, "retrain_every_days": NEVER, "safety_net_tol": 1.0,
               "min_lift": 0.0, "candidate_windows": []},
    "sliding": {"train_window_days": 60, "candidate_windows": []},
    "expanding": {"train_window_days": 0, "candidate_windows": []},
    "adaptive": {"train_window_days": 60, "candidate_windows": [60, 0]},
}


def prepare(dataset, in_dir):
    """Returns the data_dir the framework should read (raw CSVs for IEEE-CIS, else a
    prepared transactions.parquet, built here if missing)."""
    if dataset == "ieee":
        return in_dir
    out = f"data/{dataset}"
    if os.path.exists(os.path.join(out, "transactions.parquet")):
        return out
    cmd = (["prepare_sparkov.py", "--data_dir", in_dir, "--out_dir", out] if dataset == "sparkov"
           else ["prepare_baf.py", "--in_dir", in_dir, "--out_dir", out])
    subprocess.run([sys.executable, *cmd], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(BASE))
    ap.add_argument("--in_dir", required=True, help="folder with the raw (or prepared) data")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--end_day", type=float, help="stop each replay at this day (quick checks)")
    args = ap.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

    out = f"outputs/{args.dataset}/window_selection"
    os.makedirs(out, exist_ok=True)
    data_dir = prepare(args.dataset, args.in_dir)
    mlflow = {"tracking_uri": f"sqlite:///{out}/mlflow.db", "artifact_dir": f"{out}/mlruns"}

    store, cache, rows = None, {}, []
    for name in args.variants.split(","):
        cfg = load_config(BASE[args.dataset], {
            "data": {"data_dir": data_dir},
            "mlflow": {**mlflow, "model_name": f"{args.dataset}-{name}"},
            "policy": VARIANTS[name],
            "replay": {"out_dir": f"{out}/{name}"}})
        if store is None:
            print("Loading data ...", flush=True)
            store = TransactionStore(load_transactions(cfg.data.data_dir, cfg.data.cache_path))
        print(f"\n=== {args.dataset}: {name} ===", flush=True)
        t = time.time()
        summary, _, _ = ReplayRunner(cfg, store=store, model_cache=cache,
                                     log=lambda m: print(m, flush=True)).run(end_day=args.end_day)
        build_dashboard(cfg)
        print(json.dumps(summary, indent=1), flush=True)
        rows.append({"variant": name, "mean_weekly_pr_auc": summary["served_mean_weekly_pr_auc"],
                     "min_weekly_pr_auc": summary["served_min_weekly_pr_auc"],
                     "retrains": summary["retrains"], "promotions": summary["promotions"],
                     "rejections": summary["rejections"],
                     "windows_selected": json.dumps(summary.get("windows_selected", {})),
                     "minutes": round((time.time() - t) / 60, 1)})

    table = pd.DataFrame(rows)
    if "static" in set(table["variant"]):
        static = float(table.loc[table["variant"] == "static", "mean_weekly_pr_auc"].iloc[0])
        table["vs_static"] = table["mean_weekly_pr_auc"] - static
    fixed = table[table["variant"].isin(["sliding", "expanding"])]
    if len(fixed) == 2:
        table["vs_best_fixed"] = table["mean_weekly_pr_auc"] - fixed["mean_weekly_pr_auc"].max()
    table.to_csv(f"{out}/comparison.csv", index=False)
    shown = table.round(4).astype(str)
    md = ["| " + " | ".join(shown.columns) + " |", "|" + "---|" * len(shown.columns)]
    md += ["| " + " | ".join(r) + " |" for r in shown.itertuples(index=False)]
    with open(f"{out}/comparison.md", "w", encoding="utf-8") as f:
        f.write(f"# Window selection: {args.dataset}\n\n" + "\n".join(md) + "\n")
    print("\n=== Comparison ===\n" + table.round(4).to_string(index=False))
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
