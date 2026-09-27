"""
IEEE-CIS Fraud Detection: does the promotion gate stop bad models? (fault injection)
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Replays the full framework (fraud_mlops) on the real stream with faults injected into
selected retraining jobs (default: the 2nd and 4th scheduled retrain), once with the promotion gate
on and once with it off, and compares both with a clean run.

Faults (fraud_mlops/faults.py):
  label_shuffle           training job reads misaligned labels
  label_loss              chargeback feed outage: 80% of frauds recorded as legitimate
  feature_unit            unit bug: champion's top-5 numeric features x100 in training only
  upstream_label_shuffle  label store itself corrupted, so the gate reads bad labels too

Usage:
  python gate_fault_test.py --retrains 2,4

Uses its own MLflow store (outputs/gate_test/mlflow.db) so the main registry is untouched.
Outputs (outputs/gate_test/): gate_results.csv, gate_fault_steps.csv, gate_weekly.csv,
gate_fault_test.png, and a dashboard per scenario in <scenario>/dashboard.html.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fraud_mlops.config import load_config
from fraud_mlops.dashboard import build_dashboard
from fraud_mlops.data import TransactionStore, load_transactions
from fraud_mlops.faults import SCENARIOS
from fraud_mlops.pipeline import ReplayRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.toml")
    ap.add_argument("--out_dir", default="outputs/gate_test")
    ap.add_argument("--retrains", default="2,4",
                    help="scheduled retrain numbers that receive the fault")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    args = ap.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    os.makedirs(args.out_dir, exist_ok=True)

    base = load_config(args.config)
    store = TransactionStore(load_transactions(base.data.data_dir, base.data.cache_path))
    hit = [int(r) for r in args.retrains.split(",")]
    cache = {}

    runs = [("clean", None, True)]
    for name in args.scenarios.split(","):
        runs += [(name, SCENARIOS[name], True), (name, SCENARIOS[name], False)]

    results, steps, weekly = [], [], []
    for name, fault, gate_on in runs:
        label = name if fault is None else f"{name}_gate_{'on' if gate_on else 'off'}"
        out = os.path.join(args.out_dir, label)
        cfg = load_config(args.config, {
            "mlflow": {"tracking_uri": f"sqlite:///{args.out_dir}/mlflow.db",
                       "artifact_dir": f"{args.out_dir}/mlruns",
                       "model_name": "ieee-cis-fraud-gate-test"},
            "replay": {"out_dir": out}})
        print(f"\n=== {label} ===")
        runner = ReplayRunner(cfg, store=store, gate_enabled=gate_on, model_cache=cache,
                              faults={r: fault for r in hit} if fault else None)
        summary, wk, dec = runner.run()
        build_dashboard(cfg)

        faulty = dec[dec.get("fault", pd.Series("", index=dec.index)).fillna("") != ""]
        results.append({"scenario": name, "gate": "on" if gate_on else ("off" if fault else "on"),
                        "mean_weekly_pr_auc": summary["served_mean_weekly_pr_auc"],
                        "worst_week_pr_auc": summary["served_min_weekly_pr_auc"],
                        "retrains": summary["retrains"], "promotions": summary["promotions"],
                        "rejections": summary["rejections"],
                        "faulty_models_promoted": int((faulty["promoted"] == True).sum()),  # noqa: E712
                        "faulty_models": len(faulty)})
        for r in faulty.to_dict("records"):
            steps.append({"scenario": name, "gate": "on" if gate_on else "off", "day": r["day"],
                          "challenger_pr_auc": r["gate_challenger_pr_auc"],
                          "champion_pr_auc": r["gate_champion_pr_auc"],
                          "promoted": r["promoted"], "gate_reason": r["gate"]})
        weekly.append(wk.assign(run=label))

    res = pd.DataFrame(results)
    clean = res.loc[res["scenario"] == "clean", "mean_weekly_pr_auc"].iloc[0]
    res["delta_vs_clean"] = res["mean_weekly_pr_auc"] - clean
    res.to_csv(os.path.join(args.out_dir, "gate_results.csv"), index=False)
    pd.DataFrame(steps).to_csv(os.path.join(args.out_dir, "gate_fault_steps.csv"), index=False)
    weekly = pd.concat(weekly)
    weekly.to_csv(os.path.join(args.out_dir, "gate_weekly.csv"), index=False)

    # one panel per fault: clean vs gate on vs gate off
    names = [n for n in args.scenarios.split(",")]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4), sharey=True)
    base_w = weekly[weekly["run"] == "clean"]
    fault_days = sorted({s["day"] for s in steps})
    for ax, n in zip(axes, names):
        ax.plot(base_w["day_start"], base_w["pr_auc"], color="#898781", lw=1.5, label="clean")
        for gate, color in (("on", "#2a78d6"), ("off", "#eb6834")):
            w = weekly[weekly["run"] == f"{n}_gate_{gate}"]
            ax.plot(w["day_start"], w["pr_auc"], marker="o", ms=3, lw=2, color=color,
                    label=f"fault, gate {gate}")
        for d in fault_days:
            ax.axvline(d, color="#c3c2b7", lw=1)
        ax.set_title(n.replace("_", " "), fontsize=10)
        ax.set_xlabel("Day of data")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Served PR-AUC (weekly)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Faulty retrains (grey rules): with vs without the promotion gate", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "gate_fault_test.png"), dpi=150)
    plt.close()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 90)
    print("\n=== Results ===\n" + res.round(4).to_string(index=False))
    print("\n=== Gate at the faulty retrains ===\n"
          + pd.DataFrame(steps).drop(columns="gate_reason").round(3).to_string(index=False))
    print(f"\nDone. Outputs in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
