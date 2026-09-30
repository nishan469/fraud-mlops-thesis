"""Loading IEEE-CIS data and slicing it by time (days since the first transaction)."""

import os

import numpy as np
import pandas as pd

TARGET = "isFraud"
TIME = "TransactionDT"
NON_FEATURES = {TARGET, "TransactionID", TIME}


def reduce_memory(df):
    """Downcast numerics so the ~590k x 434 table fits in laptop RAM."""
    for col in df.columns:
        dt = df[col].dtype
        if dt == "float64":
            df[col] = df[col].astype("float32")
        elif dt == "int64" and col not in ("TransactionID", TIME):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def feature_columns(df):
    """All model features, and the subset that is categorical (any non-numeric column)."""
    feats = [c for c in df.columns if c not in NON_FEATURES]
    cats = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]
    return feats, cats


def load_transactions(data_dir, cache_path=""):
    """train_transaction.csv (+ train_identity.csv if present), time-sorted, categoricals as
    pandas `category`. Cached to parquet after the first load when cache_path is set.
    A data_dir holding transactions.parquet (already in this layout) is read directly."""
    if cache_path and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    prepared = os.path.join(data_dir, "transactions.parquet")   # e.g. prepare_sparkov.py
    if os.path.exists(prepared):
        return pd.read_parquet(prepared).sort_values(TIME).reset_index(drop=True)

    df = reduce_memory(pd.read_csv(os.path.join(data_dir, "train_transaction.csv")))
    id_path = os.path.join(data_dir, "train_identity.csv")
    if os.path.exists(id_path):
        idf = reduce_memory(pd.read_csv(id_path))
        idf.columns = [c.replace("-", "_") for c in idf.columns]   # id-01 vs id_01
        df = df.merge(idf, on="TransactionID", how="left")
    df = df.sort_values(TIME).reset_index(drop=True)
    _, cats = feature_columns(df)
    for c in cats:
        df[c] = df[c].astype("category")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_parquet(cache_path)
    return df


class TransactionStore:
    """Time-indexed view of the data. Rows must be sorted by TransactionDT."""

    def __init__(self, df):
        self.df = df
        self.day = df[TIME].to_numpy() / 86400.0
        self.y = df[TARGET].to_numpy()
        self.features, self.categorical = feature_columns(df)

    def row(self, day):
        """First row at or after `day`."""
        return int(np.searchsorted(self.day, day, side="left"))

    def rows(self, start_day, end_day):
        return slice(self.row(start_day), self.row(end_day))

    def X(self, rows):
        return self.df.iloc[rows][self.features]

    def day_at_fraction(self, frac):
        return float(self.day[min(int(len(self.day) * frac), len(self.day) - 1)])
