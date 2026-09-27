"""The deployable model: a LightGBM booster plus the spec needed to score raw input."""

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

    def predict(self, X):
        return self.booster.predict(self._prepare(X), num_iteration=self.spec["best_iteration"])

    def decide(self, X):
        p = self.predict(X)
        return p, (p >= self.threshold).astype(int)

    def top_numeric_features(self, k):
        gain = pd.Series(self.booster.feature_importance("gain"), index=self.booster.feature_name())
        cats = set(self.spec["categorical"])
        return [f for f in gain.sort_values(ascending=False).index if f not in cats][:k]
