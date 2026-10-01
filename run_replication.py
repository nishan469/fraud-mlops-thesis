"""
Replicate the IEEE-CIS experiments on another dataset, end to end.
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Runs, in order, the same scripts and settings as for IEEE-CIS:

  prepare -> baseline -> label-delay sweep -> walk-forward -> seeds
          -> framework replay (60-day window) -> framework replay (expanding window)

  python run_replication.py --dataset baf                       # everything
  python run_replication.py --dataset baf --from sweep          # resume from a step
  python run_replication.py --dataset baf --to sweep            # stop after a step
  python run_replication.py --dataset baf --in_dir /kaggle/input/...   # where the raw data is

Outputs go to outputs/<dataset>/<step>/ and a log per step to outputs/<dataset>/logs/.
After an interruption, rerun with --from <the unfinished step>: the long steps run with
--resume and keep every label delay or seed already saved.
"""

import argparse
import os
import subprocess
import sys
import time

DATASETS = {
    # stream starts at row fraction 0.5: half the data is history before the first deployment
    "sparkov": {"prepare": ["prepare_sparkov.py"], "in_flag": "--data_dir", "start": "0.5",
                "configs": ["configs/sparkov.toml", "configs/sparkov_expanding.toml"]},
    "baf": {"prepare": ["prepare_baf.py", "--out_dir", "data/baf"], "in_flag": "--in_dir",
            "start": "0.5",
            "configs": ["configs/baf.toml", "configs/baf_expanding.toml"]},
}


def steps(name, in_dir=None):
    d = DATASETS[name]
    data, out, start = f"data/{name}", f"outputs/{name}", d["start"]
    prepare = d["prepare"] + ([d["in_flag"], in_dir] if in_dir else [])
    sliding, expanding = d["configs"]
    return [
        ("prepare", prepare),
        ("baseline", ["ieee_cis_eda_baseline.py", "--data_dir", data, "--out_dir",
                      f"{out}/baseline", "--n_bins", "10"]),
        ("sweep", ["sweep_label_delay.py", "--data_dir", data, "--stream_start_frac", start,
                   "--out_dir", f"{out}/sweep", "--delays", "0,15,30,60", "--n_boot", "500",
                   "--resume"]),
        ("walkforward", ["walkforward_tuning.py", "--data_dir", data, "--stream_start_frac", start,
                         "--tune_start_frac", "0.35", "--out_dir", f"{out}/walkforward",
                         "--delays", "0,15,30", "--resume"]),
        ("seeds", ["seed_robustness.py", "--data_dir", data, "--stream_start_frac", start,
                   "--out_dir", f"{out}/seeds", "--seeds", "42,1,2,3,4", "--delays", "0,30",
                   "--resume"]),
        ("replay", ["-m", "fraud_mlops", "--config", sliding, "replay"]),
        ("dashboard", ["-m", "fraud_mlops", "--config", sliding, "dashboard"]),
        ("replay_expanding", ["-m", "fraud_mlops", "--config", expanding, "replay"]),
        ("dashboard_expanding", ["-m", "fraud_mlops", "--config", expanding, "dashboard"]),
    ]


def run(dataset, start=None, stop=None, in_dir=None):
    plan = steps(dataset, in_dir)
    names = [s for s, _ in plan]
    plan = plan[names.index(start or names[0]): names.index(stop or names[-1]) + 1]
    out = f"outputs/{dataset}"
    os.makedirs(f"{out}/logs", exist_ok=True)
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "GIT_PYTHON_REFRESH": "quiet",
           "PYTHONUNBUFFERED": "1"}
    for name, cmd in plan:
        t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
        with open(f"{out}/logs/{name}.log", "w", encoding="utf-8") as log:
            rc = subprocess.call([sys.executable, "-W", "ignore", *cmd], stdout=log,
                                 stderr=subprocess.STDOUT, env=env)
        print(f"[{time.strftime('%H:%M:%S')}] {name} {'done' if rc == 0 else f'FAILED ({rc})'}"
              f" in {(time.time() - t) / 60:.1f} min", flush=True)
        if rc != 0:
            sys.exit(f"Stopped at {name}; see {out}/logs/{name}.log")
    print("All steps done.")


def main():
    names = [s for s, _ in steps("baf")]
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--from", dest="start", choices=names)
    ap.add_argument("--to", dest="stop", choices=names)
    ap.add_argument("--in_dir", help="folder with the raw files (default: data/<dataset>)")
    args = ap.parse_args()
    run(args.dataset, args.start, args.stop, args.in_dir)


if __name__ == "__main__":
    main()
