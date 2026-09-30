"""
Replicate the IEEE-CIS experiments on the Sparkov dataset, end to end.
Thesis: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Needs data/sparkov/fraudTrain.csv and fraudTest.csv (see prepare_sparkov.py). Runs, in order,
the same scripts and settings as for IEEE-CIS, with the stream starting at row fraction 0.5
(about 1 Jan 2020, one year of history before it):

  prepare -> baseline -> label-delay sweep -> walk-forward -> seeds -> framework replay

Outputs go to outputs/sparkov/<step>/ and a log per step to outputs/sparkov/logs/.
  python run_sparkov_replication.py            # everything (a few hours)
  python run_sparkov_replication.py --from sweep   # resume from a step

After an interruption (e.g. a power cut), rerun with --from <the unfinished step>. The long
steps (sweep, walkforward, seeds) run with --resume: they keep every label delay or seed
already saved and continue with the rest, so little work is lost. To recompute a step from
scratch, delete its folder under outputs/sparkov/ first.
"""

import argparse
import os
import subprocess
import sys
import time

DATA, OUT, START = "data/sparkov", "outputs/sparkov", "0.5"
STEPS = [
    ("prepare", ["prepare_sparkov.py"]),
    ("baseline", ["ieee_cis_eda_baseline.py", "--data_dir", DATA, "--out_dir", f"{OUT}/baseline",
                  "--n_bins", "10"]),
    ("sweep", ["sweep_label_delay.py", "--data_dir", DATA, "--stream_start_frac", START,
               "--out_dir", f"{OUT}/sweep", "--delays", "0,15,30,60", "--n_boot", "500", "--resume"]),
    ("walkforward", ["walkforward_tuning.py", "--data_dir", DATA, "--stream_start_frac", START,
                     "--tune_start_frac", "0.35", "--out_dir", f"{OUT}/walkforward",
                     "--delays", "0,15,30", "--resume"]),
    ("seeds", ["seed_robustness.py", "--data_dir", DATA, "--stream_start_frac", START,
               "--out_dir", f"{OUT}/seeds", "--seeds", "42,1,2,3,4", "--delays", "0,30", "--resume"]),
    ("replay", ["-m", "fraud_mlops", "--config", "configs/sparkov.toml", "replay"]),
    ("dashboard", ["-m", "fraud_mlops", "--config", "configs/sparkov.toml", "dashboard"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0][0], choices=[s for s, _ in STEPS])
    args = ap.parse_args()
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "GIT_PYTHON_REFRESH": "quiet",
           "PYTHONUNBUFFERED": "1"}
    names = [s for s, _ in STEPS]
    for name, cmd in STEPS[names.index(args.start):]:
        t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
        with open(f"{OUT}/logs/{name}.log", "w", encoding="utf-8") as log:
            rc = subprocess.call([sys.executable, "-W", "ignore", *cmd], stdout=log,
                                 stderr=subprocess.STDOUT, env=env)
        print(f"[{time.strftime('%H:%M:%S')}] {name} {'done' if rc == 0 else f'FAILED ({rc})'}"
              f" in {(time.time() - t) / 60:.1f} min", flush=True)
        if rc != 0:
            sys.exit(f"Stopped at {name}; see {OUT}/logs/{name}.log")
    print("All steps done.")


if __name__ == "__main__":
    main()
