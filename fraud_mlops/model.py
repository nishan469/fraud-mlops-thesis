"""The deployable model: a LightGBM booster plus the spec needed to score raw input."""

import numpy as np
import pandas as pd


class FraudModel:
    """`spec` holds everything besides the booster that scoring and monitoring need:
    features, categorical, threshold, train_lo, train_hi, val_pr_auc, best_iteration, ...
    It is stored as feature_spec.json next to the model in MLflow."""

    def __init__(self, booster, spec, version=None):
        self.booster = booster
        self.spec = spec
        self.version = version
        # training-time extras kept in memory only (not persisted)
        self.val_rows = None
        self.val_scores = None
        self.feature_ref = None     # {feature: sample of training values}, for drift PSI

    @property
    def features(self):
        return self.spec["features"]

    @property
    def threshold(self):
        return self.spec["threshold"]

    def _prepare(self, X):
        X = X.reindex(columns=self.features)       # missing columns -> NaN, extras dropped
        for c in self.spec["categorical"]:
            if not isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = X[c].astype("category")     # booster remaps to training categories
        for c in self.features:
            if c not in self.spec["categorical"] and not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def _layout(self):
        """Column positions and category -> code maps, built once per model. The codes are
        the booster's own training categories (pandas_categorical), in the order LightGBM
        assigned them, so scoring a numeric matrix gives the same result as a DataFrame."""
        if getattr(self, "_lay", None) is None:
            cats = set(self.spec["categorical"])
            cat_cols = [c for c in self.features if c in cats]
            levels = self.booster.pandas_categorical or []
            if len(levels) != len(cat_cols):
                self._lay = False                  # unexpected model layout: use _prepare
            else:
                self._lay = {
                    "num_idx": [i for i, c in enumerate(self.features) if c not in cats],
                    "num_cols": [c for c in self.features if c not in cats],
                    "cat_idx": [i for i, c in enumerate(self.features) if c in cats],
                    "cat_cols": cat_cols,
                    "codes": [{v: float(k) for k, v in enumerate(lv)} for lv in levels]}
        return self._lay

    def _matrix(self, X):
        """Raw input -> float matrix in feature order (missing or unknown values -> NaN)."""
        lay = self._layout()
        X = X.reindex(columns=self.features)
        out = np.empty((len(X), len(self.features)))
        num = X[lay["num_cols"]]
        try:
            out[:, lay["num_idx"]] = num.to_numpy(dtype="float64", na_value=np.nan)
        except (TypeError, ValueError):            # text in a numeric column
            out[:, lay["num_idx"]] = num.apply(pd.to_numeric, errors="coerce").to_numpy("float64")
        for j, c, codes in zip(lay["cat_idx"], lay["cat_cols"], lay["codes"]):
            out[:, j] = X[c].astype(object).map(codes).to_numpy("float64", na_value=np.nan)
        return out

    def _matrix_records(self, records):
        """Same as _matrix for a list of dicts (JSON input), without building a
        DataFrame; per-request overhead dominates for real-time requests of a few rows."""
        lay = self._layout()
        cat_pos = dict(zip(lay["cat_idx"], lay["codes"]))
        out = np.full((len(records), len(self.features)), np.nan)
        for r, rec in enumerate(records):
            for j, f in enumerate(self.features):
                v = rec.get(f)
                if v is None:
                    continue
                codes = cat_pos.get(j)
                if codes is not None:
                    out[r, j] = codes.get(v, np.nan)
                else:
                    try:
                        out[r, j] = float(v)
                    except (TypeError, ValueError):
                        pass
        return out

    def decide_records(self, records):
        """decide() for a list of dicts, as received by the scoring API."""
        if len(records) >= 1000 or not self._layout():
            return self.decide(pd.DataFrame(records))
        p = self.booster.predict(self._matrix_records(records),
                                 num_iteration=self.spec["best_iteration"], num_threads=1)
        return p, (p >= self.threshold).astype(int)

    def predict(self, X):
        it = self.spec["best_iteration"]
        if self._layout():
            # one thread for small requests: no OpenMP start-up cost, and concurrent requests
            # do not compete for the same cores
            threads = 1 if len(X) < 1000 else 0
            return self.booster.predict(self._matrix(X), num_iteration=it, num_threads=threads)
        return self.booster.predict(self._prepare(X), num_iteration=it)

    def decide(self, X):
        p = self.predict(X)
        return p, (p >= self.threshold).astype(int)

    def top_numeric_features(self, k):
        gain = pd.Series(self.booster.feature_importance("gain"), index=self.booster.feature_name())
        cats = set(self.spec["categorical"])
        return [f for f in gain.sort_values(ascending=False).index if f not in cats][:k]
