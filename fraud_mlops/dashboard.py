"""Static monitoring dashboard for a replay: one self-contained HTML file built from the
replay outputs (decisions.csv, served_weekly.csv, summary.json) and the MLflow registry.

  python -m fraud_mlops dashboard  ->  outputs/replay/dashboard.html
"""

import json
import math
import os
import re
import time
from pathlib import Path

import pandas as pd

from .registry import CHAMPION, PREVIOUS, ModelRegistry

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
GATE_RE = re.compile(r"challenger ([\d.]+) vs champion ([\d.]+)")


def _clean(v):
    """JSON-safe scalar: NaN/inf -> None, numpy -> python."""
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _records(df):
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def _version_str(v):
    v = _clean(v)
    if v is None or v == "":
        return None
    return str(int(float(v)))


def collect(cfg):
    out_dir = cfg.replay.out_dir
    with open(os.path.join(out_dir, "summary.json")) as f:
        summary = json.load(f)
    decisions = pd.read_csv(os.path.join(out_dir, "decisions.csv"))
    weekly = pd.read_csv(os.path.join(out_dir, "served_weekly.csv"))
    p = cfg.policy

    steps = []
    psi_cols = [c for c in decisions.columns if c.startswith("psi_")]
    for r in decisions.to_dict("records"):
        psis = {c[4:]: r[c] for c in psi_cols if pd.notna(r[c])}
        top = max(psis, key=psis.get) if psis else None
        ch, cp = r.get("gate_challenger_pr_auc"), r.get("gate_champion_pr_auc")
        gate_text = r.get("gate") if isinstance(r.get("gate"), str) else ""
        if (ch is None or pd.isna(ch)) and gate_text:          # older replays: parse the text
            m = GATE_RE.search(gate_text)
            ch, cp = (float(m.group(1)), float(m.group(2))) if m else (None, None)
        ref = r.get("ref_pr_auc")
        alerts = r.get("alerts") if isinstance(r.get("alerts"), str) else ""
        steps.append({k: _clean(v) for k, v in {
            "day": r["day"], "champion": _version_str(r["champion"]),
            "live_pr_auc": r.get("live_pr_auc"), "ref_pr_auc": ref,
            "threshold": ref * (1 - p.safety_net_tol) if ref is not None and pd.notna(ref) else None,
            "score_psi": r.get("score_psi"), "max_feature_psi": r.get("max_feature_psi"),
            "top_feature": top, "alerts": [a for a in alerts.split(";") if a],
            "retrain": bool(r["retrain"]),
            "reason": r["reason"] if isinstance(r.get("reason"), str) else "",
            "challenger": _version_str(r.get("challenger")),
            "promoted": None if pd.isna(r.get("promoted", float("nan"))) else bool(r["promoted"]),
            "gate_text": gate_text, "gate_challenger": ch, "gate_champion": cp,
            "fault": r["fault"] if isinstance(r.get("fault"), str) else "",
            "matured_positives": r.get("matured_positives")}.items()})

    weekly_rows = _records(weekly.rename(columns={"day_start": "day"}))
    for w in weekly_rows:
        w["model_version"] = _version_str(w["model_version"])

    # registry: only versions that took part in this replay
    reg = ModelRegistry(cfg.mlflow)
    used = {s["champion"] for s in steps} | {s["challenger"] for s in steps if s["challenger"]}
    aliases = {}
    for alias in (CHAMPION, PREVIOUS):
        try:
            aliases[str(reg.client.get_model_version_by_alias(reg.name, alias).version)] = alias
        except Exception:
            pass
    served_days = weekly.groupby(weekly["model_version"].map(_version_str))["n"].count() \
        * p.step_days
    versions = []
    for mv in reg.client.search_model_versions(f"name='{reg.name}'"):
        v = str(mv.version)
        if v not in used:
            continue
        params = reg.client.get_run(mv.run_id).data.params if mv.run_id else {}
        versions.append({
            "version": v, "alias": aliases.get(v),
            "train_lo": _clean(float(params["train_lo_day"])) if "train_lo_day" in params else None,
            "train_hi": _clean(float(params["train_hi_day"])) if "train_hi_day" in params else None,
            "n_train": int(params["n_train"]) if "n_train" in params else None,
            "val_pr_auc": _clean(float(mv.tags["val_pr_auc"])) if "val_pr_auc" in mv.tags else None,
            "trigger": mv.tags.get("trigger", ""),
            "rejected": mv.tags.get("rejected"),
            "served_days": _clean(float(served_days.get(v, 0))),
            "run_id": mv.run_id})
    versions.sort(key=lambda d: int(d["version"]))

    reasons = [s["reason"] for s in steps if s["retrain"]]
    return {
        "meta": {"model_name": reg.name, "run_id": summary.get("mlflow_run_id"),
                 "label_delay": p.label_delay_days, "step_days": p.step_days,
                 "retrain_every": p.retrain_every_days, "window": p.train_window_days,
                 "safety_net_tol": p.safety_net_tol, "psi_alert": cfg.monitoring.psi_alert,
                 "psi_features": cfg.monitoring.psi_features,
                 "start_day": steps[0]["day"] if steps else None,
                 "end_day": (steps[-1]["day"] + p.step_days) if steps else None,
                 "final_champion": summary.get("final_champion"),
                 "tracking_uri": cfg.mlflow.tracking_uri,
                 "generated": time.strftime("%Y-%m-%d %H:%M")},
        "kpis": {"mean_pr": _clean(summary.get("served_mean_weekly_pr_auc")),
                 "min_pr": _clean(summary.get("served_min_weekly_pr_auc")),
                 "retrains": len(reasons),
                 "scheduled": sum(r.startswith("scheduled") for r in reasons),
                 "safety_net": sum(r.startswith("safety net") for r in reasons),
                 "promotions": sum(1 for s in steps if s["promoted"] is True),
                 "rejections": sum(1 for s in steps if s["promoted"] is False),
                 "drift_alerts": sum(1 for s in steps for a in s["alerts"]
                                     if a.startswith(("feature_drift", "score_drift")))},
        "weekly": weekly_rows, "steps": steps, "versions": versions}


def build_dashboard(cfg, out_path=None):
    data = collect(cfg)
    payload = json.dumps(data, allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DASHBOARD_DATA__", payload)
    out_path = out_path or os.path.join(cfg.replay.out_dir, "dashboard.html")
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path
