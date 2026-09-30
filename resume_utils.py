"""Resuming the long experiment scripts after an interruption (for example a power cut).

The scripts save their results after every finished unit of work (a label delay, or a
seed and delay). With --resume they read those files back and skip the finished units.
check_settings() stops a resume from mixing in results made with different settings.
"""

import json
import os

import pandas as pd

# arguments that legitimately differ between the original run and a resume
NOT_SETTINGS = {"resume", "delays", "seeds", "label_delay", "verbose", "smoke"}


def check_settings(out_dir, args):
    """Record this run's settings in <out_dir>/run_settings.json. On --resume, refuse to
    continue if the recorded settings differ from the current ones."""
    path = os.path.join(out_dir, "run_settings.json")
    now = {k: v for k, v in sorted(vars(args).items())
           if k not in NOT_SETTINGS and isinstance(v, (str, int, float, bool, type(None)))}
    if getattr(args, "resume", False) and os.path.exists(path):
        with open(path) as f:
            before = json.load(f)
        diff = {k: (before.get(k), now.get(k)) for k in sorted(set(before) | set(now))
                if before.get(k) != now.get(k)}
        if diff:
            lines = "\n".join(f"  {k}: saved {a!r}, now {b!r}" for k, (a, b) in diff.items())
            raise SystemExit(f"Cannot resume in {out_dir}: settings differ from the saved run:\n"
                             f"{lines}\nUse the original settings, or delete the folder to start over.")
    with open(path, "w") as f:
        json.dump(now, f, indent=2)


def load_saved(out_dir, name, resume):
    """Previously saved rows of <out_dir>/<name> when resuming, else an empty DataFrame."""
    path = os.path.join(out_dir, name)
    if resume and os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return pd.DataFrame()
