"""Typed configuration loaded from a TOML file (see configs/default.toml)."""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: str = "data"
    cache_path: str = ""


@dataclass
class MlflowConfig:
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_dir: str = "mlruns"
    training_experiment: str = "fraud-training"
    monitoring_experiment: str = "fraud-monitoring"
    model_name: str = "ieee-cis-fraud"


@dataclass
class ModelConfig:
    seed: int = 42
    val_frac: float = 0.15
    learning_rate: float = 0.05
    num_leaves: int = 256
    min_child_samples: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.5
    reg_lambda: float = 1.0
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100


@dataclass
class PolicyConfig:
    label_delay_days: float = 30
    step_days: float = 7
    retrain_every_days: float = 14
    train_window_days: float = 60
    # windows tried at every retrain (days, 0 = expanding); the gate keeps the best. Empty:
    # only train_window_days
    candidate_windows: list = field(default_factory=list)
    safety_net_tol: float = 0.30
    history_window: int = 4
    min_lift: float = 3.0
    monitor_days: float = 14
    cooldown_days: float = 7
    min_positives: int = 30


@dataclass
class GateConfig:
    max_pr_auc_drop: float = 0.01
    min_positives: int = 30


@dataclass
class MonitoringConfig:
    psi_features: int = 10
    psi_alert: float = 0.25
    psi_sample: int = 50000


@dataclass
class ReplayConfig:
    start_frac: float = 0.6
    out_dir: str = "outputs/replay"


@dataclass
class Config:
    data: DataConfig
    mlflow: MlflowConfig
    model: ModelConfig
    policy: PolicyConfig
    gate: GateConfig
    monitoring: MonitoringConfig
    replay: ReplayConfig


def _build(cls, values):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for [{cls.__name__}]: {sorted(unknown)}")
    return cls(**values)


def load_config(path=None, overrides=None):
    """Load a TOML config; `overrides` is {"section": {"key": value}} applied on top."""
    raw = {}
    if path is not None:
        with open(Path(path), "rb") as f:
            raw = tomllib.load(f)
    for section, values in (overrides or {}).items():
        raw.setdefault(section, {}).update(values)
    return Config(**{f.name: _build(f.type, raw.get(f.name, {})) for f in fields(Config)})
