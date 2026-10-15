"""
Load test of the fraud scoring API (fraud_mlops serve).
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Starts the FastAPI service with the registry's champion model in a separate process, then
sends real IEEE-CIS transactions (raw columns, as a payment system would) to POST /predict.

  1. Concurrency test: N simulated clients (1, 10, 50, 100), each sending one transaction
     per request and waiting for the answer before sending the next (closed loop), for
     --duration seconds per level after a short warm-up.
  2. Batch test: one client sending batches of 1, 10, 100 and 1000 transactions per request.

Reported per level: requests and transactions per second, latency p50/p95/p99/max (measured
at the client, so it includes JSON encoding and the HTTP round trip), and errors. The scores
returned by the API are also checked against the model loaded directly from the registry.

  python load_test.py --config configs/default.toml
Outputs (in --out_dir, default ./outputs/loadtest): loadtest_concurrency.csv,
loadtest_batch.csv, loadtest_machine.json, loadtest.png
"""

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time

import aiohttp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from fraud_mlops.config import load_config


def sample_transactions(cfg, n, seed=0):
    """n raw transactions from the replayed stream, as JSON-ready dicts (NaN -> null)."""
    df = pd.read_parquet(cfg.data.cache_path)
    df = df.sort_values("TransactionDT").iloc[int(len(df) * 0.6):]       # the stream part
    rows = df.sample(n, random_state=seed).drop(columns=["isFraud"])
    for c in rows.columns:
        if isinstance(rows[c].dtype, pd.CategoricalDtype):
            rows[c] = rows[c].astype(object)
    rows = rows.astype(object).where(rows.notna(), None)
    return rows.to_dict(orient="records")


def wait_ready(url, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        try:
            r = requests.get(url + "/health", timeout=2)
            if r.ok and r.json().get("champion_version"):
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1)
    raise SystemExit("Scoring API did not become ready (is there a champion model?)")


async def run_level(url, payloads, users, duration, warmup, batch=1):
    """Closed-loop clients; returns latencies (s) of requests finished after the warm-up."""
    lat, errors = [], 0
    t_start = time.perf_counter()
    t_measure, t_end = t_start + warmup, t_start + warmup + duration
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as s:
        async def client(k):
            nonlocal errors
            i = k * 7919
            while True:
                start = time.perf_counter()
                if start >= t_end:
                    return
                body = {"transactions": [payloads[(i + j) % len(payloads)] for j in range(batch)]}
                i += batch
                try:
                    async with s.post(url + "/predict", json=body) as r:
                        await r.read()
                        ok = r.status == 200
                except aiohttp.ClientError:
                    ok = False
                done = time.perf_counter()
                if done >= t_measure and start >= t_measure:
                    if ok:
                        lat.append(done - start)
                    else:
                        errors += 1
        await asyncio.gather(*(client(k) for k in range(users)))
    return np.array(lat), errors


def summarise(lat, errors, duration, batch):
    ms = lat * 1000
    return {"requests": len(lat), "errors": errors,
            "requests_per_sec": len(lat) / duration,
            "transactions_per_sec": len(lat) * batch / duration,
            "p50_ms": np.percentile(ms, 50), "p95_ms": np.percentile(ms, 95),
            "p99_ms": np.percentile(ms, 99), "max_ms": ms.max(), "mean_ms": ms.mean()}


def check_scores(cfg, url, payloads):
    """Scores from the API must equal scores from the champion loaded in-process."""
    from fraud_mlops.registry import ModelRegistry
    m = ModelRegistry(cfg.mlflow).load()
    api = requests.post(url + "/predict", json={"transactions": payloads[:200]}, timeout=60).json()
    api_p = np.array([p["fraud_probability"] for p in api["predictions"]])
    local_p, _ = m.decide(pd.DataFrame(payloads[:200]))
    return float(np.max(np.abs(api_p - np.asarray(local_p))))


def plot(conc, batch, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    x = conc["users"].astype(str)
    ax = axes[0]
    ax.bar(x, conc["requests_per_sec"], color="#2A78D6")
    for i, v in enumerate(conc["requests_per_sec"]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Throughput (1 transaction per request)", fontweight="bold")
    ax.set_xlabel("concurrent clients")
    ax.set_ylabel("requests per second")
    ax = axes[1]
    for col, c, lab in [("p50_ms", "#2A78D6", "p50"), ("p95_ms", "#EB6834", "p95"),
                        ("p99_ms", "#B4501A", "p99")]:
        ax.plot(conc["users"], conc[col], marker="o", color=c, label=lab)
    ax.set_xscale("log")
    ax.set_xticks(conc["users"])
    ax.set_xticklabels(conc["users"])
    ax.set_title("Latency (1 transaction per request)", fontweight="bold")
    ax.set_xlabel("concurrent clients")
    ax.set_ylabel("milliseconds")
    ax.legend(frameon=False)
    ax = axes[2]
    xb = batch["batch"].astype(str)
    ax.bar(xb, batch["transactions_per_sec"], color="#1BB27B")
    for i, (v, p) in enumerate(zip(batch["transactions_per_sec"], batch["p50_ms"])):
        ax.text(i, v, f"{v:,.0f}\n({p:.0f} ms)", ha="center", va="bottom", fontsize=8)
    ax.set_title("Batch scoring (1 client)", fontweight="bold")
    ax.set_xlabel("transactions per request")
    ax.set_ylabel("transactions per second")
    ax.set_ylim(0, batch["transactions_per_sec"].max() * 1.25)
    for a in axes:
        a.grid(axis="y", alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.toml")
    ap.add_argument("--out_dir", default="./outputs/loadtest")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--users", default="1,10,50,100")
    ap.add_argument("--batches", default="1,10,100,1000")
    ap.add_argument("--duration", type=float, default=30, help="measured seconds per level")
    ap.add_argument("--warmup", type=float, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config(args.config)
    url = f"http://127.0.0.1:{args.port}"

    print("Sampling transactions ...")
    payloads = sample_transactions(cfg, 5000)
    print(f"  {len(payloads)} transactions, {len(payloads[0])} raw fields, "
          f"{len(json.dumps({'transactions': payloads[:1]})) / 1024:.1f} KB per single request")

    env = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1")
    server = subprocess.Popen([sys.executable, "-m", "fraud_mlops", "--config", args.config,
                               "serve", "--port", str(args.port)], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        health = wait_ready(url)
        print(f"API ready: {health}")
        diff = check_scores(cfg, url, payloads)
        print(f"Max |API score - direct score| on 200 transactions: {diff:.2e}")

        conc = []
        for u in [int(v) for v in args.users.split(",")]:
            lat, err = asyncio.run(run_level(url, payloads, u, args.duration, args.warmup))
            row = {"users": u, **summarise(lat, err, args.duration, 1)}
            conc.append(row)
            print(f"  {u:>4} clients: {row['requests_per_sec']:7.1f} req/s   p50 {row['p50_ms']:6.1f}  "
                  f"p95 {row['p95_ms']:6.1f}  p99 {row['p99_ms']:6.1f} ms   errors {err}")
        batch = []
        for b in [int(v) for v in args.batches.split(",")]:
            lat, err = asyncio.run(run_level(url, payloads, 1, args.duration, args.warmup, b))
            row = {"batch": b, **summarise(lat, err, args.duration, b)}
            batch.append(row)
            print(f"  batch {b:>5}: {row['transactions_per_sec']:9.1f} tx/s   p50 {row['p50_ms']:7.1f}  "
                  f"p95 {row['p95_ms']:7.1f} ms   errors {err}")
    finally:
        server.terminate()
        server.wait(timeout=30)

    conc, batch = pd.DataFrame(conc), pd.DataFrame(batch)
    conc.to_csv(os.path.join(args.out_dir, "loadtest_concurrency.csv"), index=False)
    batch.to_csv(os.path.join(args.out_dir, "loadtest_batch.csv"), index=False)
    machine = {"cpu": platform.processor(), "logical_cpus": os.cpu_count(),
               "os": platform.platform(), "python": platform.python_version(),
               "server": "uvicorn, 1 worker process", "client": "aiohttp, same machine",
               "champion_version": health.get("champion_version"),
               "raw_fields_per_transaction": len(payloads[0]),
               "max_abs_score_diff_api_vs_direct": diff,
               "duration_s": args.duration, "warmup_s": args.warmup}
    with open(os.path.join(args.out_dir, "loadtest_machine.json"), "w") as f:
        json.dump(machine, f, indent=2)
    plot(conc, batch, os.path.join(args.out_dir, "loadtest.png"))
    print("\n" + conc.round(1).to_string(index=False) + "\n\n" + batch.round(1).to_string(index=False))
    print(f"\nDone. Outputs are in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
