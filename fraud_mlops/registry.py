"""MLflow tracking + Model Registry: every trained model becomes a registered version;
the one serving traffic carries the `champion` alias, its predecessor `previous`."""

import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from .model import FraudModel

CHAMPION, PREVIOUS = "champion", "previous"


class ModelRegistry:
    def __init__(self, cfg):
        self.cfg = cfg
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        mlflow.set_tracking_uri(cfg.tracking_uri)
        self.client = MlflowClient(cfg.tracking_uri)
        self.name = cfg.model_name
        artifacts = Path(cfg.artifact_dir).resolve()
        self.training_exp = self._experiment(cfg.training_experiment, artifacts)
        self.monitoring_exp = self._experiment(cfg.monitoring_experiment, artifacts)

    def _experiment(self, name, artifacts):
        exp = self.client.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        return self.client.create_experiment(name, artifact_location=(artifacts / name).as_uri())

    # ------------------------------------------------------------ write
    def register(self, model, tags=None):
        """Log a training run (params, metrics, spec, model) and register a new version."""
        s = model.spec
        with mlflow.start_run(experiment_id=self.training_exp,
                              run_name=f"train_d{s['train_lo']:.0f}-{s['train_hi']:.0f}") as run:
            mlflow.log_params({"train_lo_day": round(s["train_lo"], 2),
                               "train_hi_day": round(s["train_hi"], 2),
                               "n_train": s["n_train"], "n_val": s["n_val"], "seed": s["seed"],
                               "n_features": len(s["features"])})
            mlflow.log_metrics({"val_pr_auc": s["val_pr_auc"], "threshold": s["threshold"],
                                "best_iteration": s["best_iteration"],
                                "train_seconds": s["train_seconds"],
                                "val_fraud_rate": s["val_fraud_rate"]})
            mlflow.set_tags(tags or {})
            mlflow.log_dict(s, "feature_spec.json")
            info = mlflow.lightgbm.log_model(model.booster, name="model",
                                             registered_model_name=self.name)
        version = str(info.registered_model_version)
        for k, v in {"run_id": run.info.run_id, "train_hi_day": f"{s['train_hi']:.2f}",
                     "val_pr_auc": f"{s['val_pr_auc']:.4f}", **(tags or {})}.items():
            self.client.set_model_version_tag(self.name, version, k, str(v))
        model.version = version
        return version

    def promote(self, version, reason=""):
        current = self.champion_version()
        if current is not None and current != str(version):
            self.client.set_registered_model_alias(self.name, PREVIOUS, current)
        self.client.set_registered_model_alias(self.name, CHAMPION, str(version))
        self.client.set_model_version_tag(self.name, str(version), "promotion_reason", reason)

    def reject(self, version, reason):
        self.client.set_model_version_tag(self.name, str(version), "rejected", reason)

    # ------------------------------------------------------------ read
    def champion_version(self):
        try:
            return str(self.client.get_model_version_by_alias(self.name, CHAMPION).version)
        except mlflow.exceptions.MlflowException:
            return None

    def load(self, alias=CHAMPION):
        mv = self.client.get_model_version_by_alias(self.name, alias)
        booster = mlflow.lightgbm.load_model(f"models:/{self.name}@{alias}")
        spec = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/feature_spec.json")
        return FraudModel(booster, spec, version=str(mv.version))
