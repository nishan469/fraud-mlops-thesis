"""
Sparkov credit-card transactions -> the table layout the IEEE-CIS code expects.
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Second dataset for replicating the IEEE-CIS findings:
  https://www.kaggle.com/datasets/kartik2112/fraud-detection  (simulated with Sparkov)
Put fraudTrain.csv and fraudTest.csv in data/sparkov/, then:
  python prepare_sparkov.py
which writes data/sparkov/transactions.parquet with TransactionID, TransactionDT (seconds
since the first transaction), isFraud and model features. Every script that takes
--data_dir (and fraud_mlops via configs/sparkov.toml) then runs on it unchanged.

Choices:
- rows sorted by time; data cut at --cutoff (default 2020-12-21) because the simulator
  produces almost no fraud after that (17 frauds, then 0, while volume doubles)
- per-card behaviour uses only the card's EARLIER transactions (no leakage):
  seconds since its previous transaction, count and spend in the previous 24 h,
  count in the previous 30 days, amount relative to its running average (a lifetime
  transaction count was dropped: it only grows, so it drifts by construction)
- identifiers and personal fields are dropped (names, street, card number, trans_num);
  the card number is used only to group a card's history
"""

import argparse
import os

import numpy as np
import pandas as pd

CATEGORICAL = ["category", "merchant", "gender", "state", "job"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def card_history_features(df):
    """Causal per-card features; df must be time-sorted with a 'ts' datetime column."""
    by_card = df.sort_values(["cc_num", "ts"], kind="stable")
    g = by_card.groupby("cc_num", sort=False)
    out = pd.DataFrame(index=by_card.index)
    out["card_secs_since_prev"] = g["ts"].diff().dt.total_seconds()
    prev_mean = g["amt"].transform(lambda s: s.shift().expanding().mean())
    out["amt_vs_card_mean"] = by_card["amt"] / prev_mean

    timed = by_card.set_index("ts").groupby("cc_num", sort=False)["amt"]
    rolled = timed.rolling("24h", closed="left").agg(["count", "sum"])
    month = timed.rolling("30D", closed="left").count()
    # same row order as by_card; an empty window comes back NaN and means none
    out["card_txns_24h"] = np.nan_to_num(rolled["count"].to_numpy(), nan=0.0)
    out["card_spend_24h"] = np.nan_to_num(rolled["sum"].to_numpy(), nan=0.0)
    out["card_txns_30d"] = np.nan_to_num(month.to_numpy(), nan=0.0)
    return out.sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/sparkov")
    ap.add_argument("--cutoff", default="2020-12-21", help="drop transactions from this date on")
    args = ap.parse_args()

    parts = [pd.read_csv(os.path.join(args.data_dir, f), index_col=0)
             for f in ("fraudTrain.csv", "fraudTest.csv")]
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    n_all = len(df)
    df = df[df["ts"] < pd.Timestamp(args.cutoff)].reset_index(drop=True)

    out = pd.DataFrame({
        "TransactionID": np.arange(len(df)),
        "TransactionDT": (df["ts"] - df["ts"].iloc[0]).dt.total_seconds().astype("int64"),
        "isFraud": df["is_fraud"].astype("int8"),
        "TransactionAmt": df["amt"].astype("float32"),
        "city_pop": df["city_pop"].astype("int32"),
        "age_years": ((df["ts"] - pd.to_datetime(df["dob"])).dt.days / 365.25).astype("float32"),
        "hour": df["ts"].dt.hour.astype("int8"),
        "weekday": df["ts"].dt.weekday.astype("int8"),
        "distance_km": haversine_km(df["lat"], df["long"], df["merch_lat"],
                                    df["merch_long"]).astype("float32"),
    })
    for c in CATEGORICAL:
        out[c] = df[c].astype("category")
    hist = card_history_features(df)
    for c in hist.columns:
        out[c] = hist[c].astype("float32")

    path = os.path.join(args.data_dir, "transactions.parquet")
    out.to_parquet(path)
    days = out["TransactionDT"].iloc[-1] / 86400
    print(f"Kept {len(out):,} of {n_all:,} transactions (cut at {args.cutoff}), {days:.0f} days, "
          f"{int(out['isFraud'].sum()):,} frauds ({out['isFraud'].mean():.3%})")
    print(f"Features ({out.shape[1] - 3}): {', '.join(out.columns[3:])}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
