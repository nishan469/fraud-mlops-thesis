"""
IEEE-CIS Fraud Detection: EDA + LightGBM baseline + temporal degradation analysis
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Runs locally or on Google Colab.

--- Colab quick start (run these in separate cells first) ---
!pip install -q lightgbm kaggle
from google.colab import files; files.upload()          # upload kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c ieee-fraud-detection -p ./data
!cd data && unzip -o -q ieee-fraud-detection.zip
# upload this script, then:
!python ieee_cis_eda_baseline.py --data_dir ./data --n_bins 10

--- Local ---
python ieee_cis_eda_baseline.py --data_dir ./data --n_bins 10

Outputs (in --out_dir, default ./outputs):
  eda_summary.txt, missingness_top30.png, baseline_metrics.json,
  metrics_by_time.csv, degradation_by_time.png, psi_by_time.csv, feature_drift_psi.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

SEED = 42


# ---------------------------------------------------------------- loading
def reduce_memory(df):
    """Downcast numerics so the ~590k x 434 table fits in Colab/laptop RAM."""
    for col in df.columns:
        dt = df[col].dtype
        if dt == "float64":
            df[col] = df[col].astype("float32")
        elif dt == "int64" and col not in ("TransactionID", "TransactionDT"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_data(data_dir):
    prepared = os.path.join(data_dir, "transactions.parquet")   # e.g. prepare_sparkov.py
    if os.path.exists(prepared):
        print(f"Loading {prepared} ...")
        return pd.read_parquet(prepared).sort_values("TransactionDT").reset_index(drop=True)
    tx_path = os.path.join(data_dir, "train_transaction.csv")
    id_path = os.path.join(data_dir, "train_identity.csv")
    print(f"Loading {tx_path} ...")
    tx = reduce_memory(pd.read_csv(tx_path))
    if os.path.exists(id_path):
        print(f"Loading {id_path} ...")
        idf = reduce_memory(pd.read_csv(id_path))
        # some Kaggle versions use id-01 instead of id_01
        idf.columns = [c.replace("-", "_") for c in idf.columns]
        tx = tx.merge(idf, on="TransactionID", how="left")
    else:
        print("train_identity.csv not found, continuing with transactions only.")
    tx = tx.sort_values("TransactionDT").reset_index(drop=True)
    return tx


# ---------------------------------------------------------------- EDA
def run_eda(df, out_dir):
    lines = []
    n, p = df.shape
    fraud_rate = df["isFraud"].mean()
    days = (df["TransactionDT"].max() - df["TransactionDT"].min()) / 86400
    lines.append(f"Rows: {n:,}   Columns: {p}")
    lines.append(f"Fraud rate: {fraud_rate:.4%}  (imbalance ~ {(1 - fraud_rate) / fraud_rate:.1f}:1)")
    lines.append(f"Time span: {days:.1f} days (TransactionDT is seconds from a reference point)")

    miss = df.isna().mean().sort_values(ascending=False)
    lines.append(f"Columns >50% missing: {(miss > 0.5).sum()}   >90% missing: {(miss > 0.9).sum()}")

    # fraud rate per week: first hint of concept drift
    week = (df["TransactionDT"] // (7 * 86400)).astype(int)
    weekly = df.groupby(week)["isFraud"].agg(["mean", "size"])
    lines.append("\nWeekly fraud rate (min / max): "
                 f"{weekly['mean'].min():.4%} / {weekly['mean'].max():.4%}")

    text = "\n".join(lines)
    print("\n=== EDA ===\n" + text)
    with open(os.path.join(out_dir, "eda_summary.txt"), "w") as f:
        f.write(text + "\n\nWeekly fraud rate:\n" + weekly.to_string())

    top = miss.head(30)[::-1]
    plt.figure(figsize=(8, 9))
    plt.barh(top.index, top.values)
    plt.xlabel("Fraction missing")
    plt.title("Top 30 columns by missingness")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "missingness_top30.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------- features
def prepare_features(df):
    drop = {"isFraud", "TransactionID", "TransactionDT"}
    feats = [c for c in df.columns if c not in drop]
    # any non-numeric column is categorical (works for pandas 1.x/2.x "object" and 3.x "str")
    cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return feats, cat_cols


def temporal_split(df, train_frac, val_frac, n_bins):
    """Time-ordered split. A random split would hide the drift we want to measure."""
    n = len(df)
    tr_end = int(n * train_frac)
    va_end = int(n * (train_frac + val_frac))
    train, val, test = df.iloc[:tr_end], df.iloc[tr_end:va_end], df.iloc[va_end:]
    bins = np.array_split(np.arange(len(test)), n_bins)
    test_bins = [test.iloc[b] for b in bins]
    return train, val, test_bins


# ---------------------------------------------------------------- model
def train_baseline(train, val, feats, cat_cols):
    pos = train["isFraud"].sum()
    spw = (len(train) - pos) / max(pos, 1)
    params = dict(
        objective="binary", learning_rate=0.05, num_leaves=256, max_depth=-1,
        min_child_samples=100, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, reg_lambda=1.0, scale_pos_weight=spw,
        metric="auc", random_state=SEED, n_jobs=-1, verbose=-1,
    )
    dtr = lgb.Dataset(train[feats], train["isFraud"], categorical_feature=cat_cols)
    dva = lgb.Dataset(val[feats], val["isFraud"], categorical_feature=cat_cols, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    return model, params


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    out = {"n": int(len(y)), "fraud_rate": float(np.mean(y)),
           "precision": float(precision_score(y, yhat, zero_division=0)),
           "recall": float(recall_score(y, yhat, zero_division=0)),
           "f1": float(f1_score(y, yhat, zero_division=0))}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return out


# ---------------------------------------------------------------- drift (PSI)
def psi(ref, cur, n_q=10):
    """Population Stability Index for a numeric feature (NaN treated as its own bucket)."""
    ref, cur = pd.Series(ref), pd.Series(cur)
    edges = np.unique(np.nanquantile(ref.dropna(), np.linspace(0, 1, n_q + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    def dist(s):
        counts = pd.cut(s.dropna(), edges).value_counts(sort=False).values.astype(float)
        counts = np.append(counts, s.isna().sum())
        return np.clip(counts / max(len(s), 1), 1e-6, None)

    r, c = dist(ref), dist(cur)
    return float(np.sum((c - r) * np.log(c / r)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--top_psi_features", type=int, default=10)
    args, _ = ap.parse_known_args()  # parse_known_args keeps Jupyter/Colab happy
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_data(args.data_dir)
    run_eda(df, args.out_dir)

    feats, cat_cols = prepare_features(df)
    train, val, test_bins = temporal_split(df, args.train_frac, args.val_frac, args.n_bins)
    print(f"\nTrain {len(train):,} | Val {len(val):,} | Test {sum(map(len, test_bins)):,} "
          f"in {args.n_bins} time bins")

    model, params = train_baseline(train, val, feats, cat_cols)
    p_val = model.predict(val[feats], num_iteration=model.best_iteration)
    thr = best_f1_threshold(val["isFraud"].values, p_val)

    # overall baseline metrics
    test_all = pd.concat(test_bins)
    p_test = model.predict(test_all[feats], num_iteration=model.best_iteration)
    baseline = {"threshold_from_val": thr,
                "validation": metrics(val["isFraud"].values, p_val, thr),
                "test_overall": metrics(test_all["isFraud"].values, p_test, thr),
                "best_iteration": model.best_iteration}
    print("\n=== Baseline ===\n" + json.dumps(baseline, indent=2))
    with open(os.path.join(args.out_dir, "baseline_metrics.json"), "w") as f:
        json.dump(baseline, f, indent=2)

    # performance per time bin: the degradation evidence
    rows = []
    for i, b in enumerate(test_bins):
        pb = model.predict(b[feats], num_iteration=model.best_iteration)
        m = metrics(b["isFraud"].values, pb, thr)
        m["bin"] = i + 1
        m["day_start"] = float(b["TransactionDT"].min() / 86400)
        rows.append(m)
    by_time = pd.DataFrame(rows)
    by_time.to_csv(os.path.join(args.out_dir, "metrics_by_time.csv"), index=False)
    print("\n=== Metrics by time bin ===\n" + by_time.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = by_time["day_start"]
    for col, label in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1 (fixed threshold)")]:
        ax.plot(x, by_time[col], marker="o", label=label)
    ax.set_xlabel("Days since start of data (test bin start)")
    ax.set_ylabel("Score")
    ax.set_title("Static LightGBM baseline: performance vs. time after training window")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "degradation_by_time.png"), dpi=150)
    plt.close()

    # feature drift (PSI) on the model's most important numeric features
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    num_top = [f for f in imp.index if f not in cat_cols][: args.top_psi_features]
    psi_rows = []
    for i, b in enumerate(test_bins):
        row = {"bin": i + 1, "day_start": float(b["TransactionDT"].min() / 86400)}
        for f in num_top:
            row[f] = psi(train[f], b[f])
        psi_rows.append(row)
    psi_df = pd.DataFrame(psi_rows)
    psi_df.to_csv(os.path.join(args.out_dir, "psi_by_time.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    for f in num_top:
        ax.plot(psi_df["day_start"], psi_df[f], marker=".", label=f)
    ax.axhline(0.1, ls="--", c="orange", lw=1, label="PSI 0.1 (moderate)")
    ax.axhline(0.25, ls="--", c="red", lw=1, label="PSI 0.25 (major)")
    ax.set_xlabel("Days since start of data (test bin start)")
    ax.set_ylabel("PSI vs. training window")
    ax.set_title("Feature drift of top-importance features")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "feature_drift_psi.png"), dpi=150)
    plt.close()

    # optional MLflow logging (pip install mlflow to enable)
    try:
        import mlflow
        with mlflow.start_run(run_name="lgbm_static_baseline"):
            mlflow.log_params({k: v for k, v in params.items() if k != "verbose"})
            mlflow.log_metrics({f"test_{k}": v for k, v in baseline["test_overall"].items()
                                if isinstance(v, float) and not np.isnan(v)})
            for _, r in by_time.iterrows():
                mlflow.log_metric("bin_pr_auc", r["pr_auc"], step=int(r["bin"]))
                mlflow.log_metric("bin_roc_auc", r["roc_auc"], step=int(r["bin"]))
            mlflow.log_artifacts(args.out_dir)
        print("\nLogged run to MLflow (view with: mlflow ui)")
    except ImportError:
        print("\nMLflow not installed, skipping experiment logging.")

    print(f"\nDone. All outputs are in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
