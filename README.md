# A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Thesis code on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
data: offline experiments that decide *how* a fraud model should be kept up to date, and
the `fraud_mlops` framework that implements that decision end to end.

## Layout

| Path | What |
|---|---|
| `fraud_mlops/` | The framework (monitoring, retrain policy, promotion gate, registry, replay, API) |
| `configs/default.toml` | Framework configuration |
| `tests/` | Unit and end-to-end tests on synthetic data (`pytest`) |
| `ieee_cis_eda_baseline.py` | EDA, static LightGBM baseline, performance/PSI decay over time |
| `retraining_simulation.py` | Retraining strategies under label delay (static / periodic / drift-triggered) |
| `sweep_label_delay.py` | Strategies across label delays with paired day-block bootstrap CIs |
| `tune_drift_trigger.py` | Grid tuning of the drift trigger on a held-out earlier period |
| `walkforward_tuning.py` | Walk-forward re-tuning every 4 weeks, delays 0/15/30 days |
| `seed_robustness.py` | Key comparisons repeated over 5 training seeds |

Put `train_transaction.csv` and `train_identity.csv` in `data/`. Install: `pip install -r requirements.txt`.

## What the experiments found (and why the framework looks like it does)

* Model performance decays after deployment (weekly PR-AUC 0.66 -> 0.42), but **feature and
  score PSI stay low** (score PSI <= 0.02): label-free drift detection never fired.
* Every retraining strategy beats a static model; the benefit shrinks with label delay
  (+0.08 PR-AUC at 0 days, +0.05 at 30, +0.01 at 60 days).
* **A fixed schedule on a recent-data window (30-60 days) beats a tuned performance-drop
  trigger at every label delay** (walk-forward: -0.07 / -0.03 / -0.03 PR-AUC for the trigger at
  0 / 15 / 30 days; all orderings hold in 5/5 training seeds).

So the framework's primary policy is **scheduled retraining on a sliding window**; the
performance trigger is kept only as a **safety net** for sharp drops, and PSI is logged for
visibility, not used to act.

## The framework

```
            every step_days (simulated clock in replay)
 ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐
 │ monitor  │──▶│ decide   │──▶│ retrain   │──▶│ gate     │──▶│ serve     │
 │ live PR- │   │ schedule │   │ sliding   │   │ challenger│  │ champion  │
 │ AUC on   │   │ or safety│   │ window,   │   │ vs champ. │  │ scores    │
 │ matured  │   │ net      │   │ register  │   │ on unseen │  │ next step │
 │ labels;  │   │          │   │ in MLflow │   │ data      │  │           │
 │ PSI info │   └──────────┘   └───────────┘   └──────────┘   └───────────┘
 └──────────┘
```

| Module | Role |
|---|---|
| `data.py` | Load/cache data, `TransactionStore` for time-window slicing |
| `training.py` | Train LightGBM on a day window; time-ordered hold-out for early stopping, threshold, gate |
| `model.py` | `FraudModel`: booster + feature spec; scores raw input (missing/extra columns, string categoricals) |
| `monitoring.py` | Live PR-AUC on matured labels the model never saw; score/feature PSI |
| `policy.py` | `RetrainPolicy` (schedule + safety net + cooldown), `PromotionGate` |
| `registry.py` | MLflow runs + Model Registry; `champion` / `previous` aliases |
| `pipeline.py` | `ReplayRunner`: the loop above over the historical stream |
| `serving.py` | FastAPI app serving the current champion |
| `dashboard.py` | Self-contained HTML dashboard of a replay (performance, drift, decisions, versions) |

### Run it

```bash
python -m fraud_mlops replay                      # full loop, 30-day label delay
python -m fraud_mlops replay --label_delay 0      # other delay
python -m fraud_mlops champion                    # current production model
python -m fraud_mlops dashboard                   # monitoring dashboard -> outputs/replay/dashboard.html
python -m fraud_mlops serve --port 8000           # scoring API (POST /predict, GET /health, POST /reload)
mlflow ui --backend-store-uri sqlite:///mlflow.db # training runs, monitoring curves, registry
python -m pytest tests                            # tests (synthetic data, ~30 s)
```

Replay outputs go to `outputs/replay/` (`decisions.csv`, `served_weekly.csv`, `summary.json`)
and to MLflow (experiment `fraud-monitoring`, one metric point per step).
