"""
Bank Account Fraud (BAF, NeurIPS 2022) -> the table layout the IEEE-CIS code expects.
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Third dataset for replicating the IEEE-CIS findings:
  https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
One million bank-account applications over 8 months, about 1.1% fraud, generated to
contain realistic drift between months. Put Base.csv in data/baf/ (or point --in_dir at the
Kaggle input folder; Base.csv is found anywhere below it), then:
  python prepare_baf.py
which writes data/baf/transactions.parquet with TransactionID, TransactionDT (seconds since
the start), isFraud and model features. Every script that takes --data_dir (and fraud_mlops
via configs/baf.toml) then runs on it unchanged.

Choices:
- BAF only records the month (0-7) of an application. Each row gets a time inside its month
  (30-day months, uniform, fixed seed), so rows are in month order and weekly windows can be
  formed; drift therefore happens between months, and weeks within a month are exchangeable.
  This is a limitation of the dataset, stated in the thesis.
- negative values in the columns BAF documents as "-1 = missing" become NaN
- device_fraud_count is dropped (constant in the Base variant); `month` is dropped as a
  feature (it would let the model memorise the period)
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
MISSING_IF_NEGATIVE = ["prev_address_months_count", "current_address_months_count",
                       "intended_balcon_amount", "bank_months_count",
                       "session_length_in_minutes", "device_distinct_emails_8w"]
DROP = ["fraud_bool", "month", "device_fraud_count"]
MONTH_SECONDS = 30 * 86400


def find_file(in_dir, name):
    hits = sorted(glob.glob(os.path.join(in_dir, "**", name), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {in_dir}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/baf", help="folder containing Base.csv (searched recursively)")
    ap.add_argument("--out_dir", default="data/baf")
    ap.add_argument("--variant", default="Base.csv", help="Base.csv, or e.g. 'Variant I.csv'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = find_file(args.in_dir, args.variant)
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    rng = np.random.default_rng(args.seed)
    t = df["month"].to_numpy("int64") * MONTH_SECONDS + rng.integers(0, MONTH_SECONDS, len(df))
    df = df.assign(_t=t).sort_values("_t", kind="stable").reset_index(drop=True)

    out = pd.DataFrame({
        "TransactionID": np.arange(len(df)),
        "TransactionDT": df["_t"].astype("int64"),
        "isFraud": df["fraud_bool"].astype("int8"),
    })
    for c in df.columns:
        if c in DROP or c == "_t":
            continue
        if c in CATEGORICAL:
            out[c] = df[c].astype("category")
        else:
            col = df[c].astype("float32")
            if c in MISSING_IF_NEGATIVE:
                col = col.where(col >= 0)
            out[c] = col

    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, "transactions.parquet")
    out.to_parquet(dest)
    by_month = df.groupby("month")["fraud_bool"].agg(["size", "mean"])
    print(f"{len(out):,} applications, {out['TransactionDT'].iloc[-1] / 86400:.0f} days, "
          f"{int(out['isFraud'].sum()):,} frauds ({out['isFraud'].mean():.3%})")
    print("Per month (rows, fraud rate):")
    for m, r in by_month.iterrows():
        print(f"  {m}: {int(r['size']):,}  {r['mean']:.3%}")
    print(f"Features ({out.shape[1] - 3}): {', '.join(out.columns[3:])}")
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
