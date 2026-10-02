"""Chapter 4 figures, drawn from the experiment outputs in ../../outputs (not in git; the PNGs
are). Colours: the first three slots of the validated reference palette, each with one fixed
meaning in every figure: blue = all labelled history, orange = last 60 days, aqua = automatic
window selection (aqua is below 3:1 contrast, so its marks carry direct labels).

  python results_figures.py   ->  figures/results_*.png
"""

import json
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "outputs"))
FIG = os.path.join(HERE, "figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
NAMES = {"ieee": "IEEE-CIS", "sparkov": "Sparkov", "baf": "BAF"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#9a9893",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK})


def style(ax, zero=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if zero:
        ax.axhline(0, color="#9a9893", lw=1)


def path(dataset, *parts):
    base = {"ieee": OUT, "sparkov": os.path.join(OUT, "sparkov"), "baf": os.path.join(OUT, "baf")}
    sub = {"ieee": {"baseline": ""}}.get(dataset, {}).get(parts[0], parts[0])
    return os.path.join(base[dataset], sub, *parts[1:])


def decay():
    """Small multiples: weekly PR-AUC of the static model (top) and the largest feature PSI
    (bottom) per time bin. Separate rows, never two scales on one axis."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True)
    for j, d in enumerate(NAMES):
        m = pd.read_csv(path(d, "baseline", "metrics_by_time.csv"))
        ps = pd.read_csv(path(d, "baseline", "psi_by_time.csv")).set_index("bin")
        ps = ps.drop(columns=[c for c in ("day_start",) if c in ps])
        ax = axes[0, j]
        ax.plot(m["bin"], m["pr_auc"], color=BLUE, lw=2, marker="o", ms=4)
        lo, hi = m["pr_auc"].min(), m["pr_auc"].max()
        pad = max(0.02, (hi - lo) * 0.35)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(NAMES[d])
        first, last = m["pr_auc"].iloc[0], m["pr_auc"].iloc[-1]
        imin = int(m["pr_auc"].idxmin())
        ax.annotate(f"{first:.2f}", (m["bin"].iloc[0], first), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color=INK)
        ax.annotate(f"min {lo:.2f}", (m["bin"].iloc[imin], lo), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=8, color=INK)
        style(ax)
        ax2 = axes[1, j]
        mx = ps.max(axis=1)
        ax2.bar(mx.index, mx.values, color=BLUE, width=0.62, edgecolor=SURFACE, lw=2)
        ax2.axhline(0.25, color=INK, lw=1, ls="--")
        ax2.text(10.4, 0.25, "0.25: major\nshift", va="bottom", ha="right", fontsize=7.5, color=MUTED)
        ax2.set_xlabel("time bin after deployment (1-10)")
        ax2.set_xticks(range(1, 11))
        style(ax2)
    axes[0, 0].set_ylabel("PR-AUC (static model)")
    axes[1, 0].set_ylabel("largest feature PSI")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "results_decay.png"), dpi=200)
    plt.close(fig)


def delay():
    """Gain over the static model by label delay, with 95% paired bootstrap intervals."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    dirs = {"ieee": os.path.join(OUT, "sweep"), "sparkov": os.path.join(OUT, "sparkov", "sweep"),
            "baf": os.path.join(OUT, "baf", "sweep")}
    for ax, d in zip(axes, NAMES):
        s = pd.read_csv(os.path.join(dirs[d], "sweep_results.csv"))
        for k, (strategy, label, color) in enumerate((
                ("periodic_expanding", "all history", BLUE),
                ("periodic_sliding", "last 60 days", ORANGE))):
            r = s[s["strategy"] == strategy]
            x = r["label_delay"].to_numpy() + (k - 0.5) * 2.2
            y = r["delta_pr_auc_vs_static"].to_numpy()
            err = [y - r["delta_lo"].to_numpy(), r["delta_hi"].to_numpy() - y]
            ax.errorbar(x, y, yerr=err, color=color, lw=2, marker="o", ms=5, capsize=3,
                        label=label)
        ax.set_title(NAMES[d])
        ax.set_xticks([0, 15, 30, 60])
        ax.set_xlabel("label delay (days)")
        style(ax, zero=True)
    axes[0].set_ylabel("PR-AUC gain over static")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "results_delay.png"), dpi=200)
    plt.close(fig)


def window():
    """Framework replays: gain over the static model with each window choice (30-day delay),
    95% bootstrap interval over weeks."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    variants = (("sliding", "last 60 days", ORANGE), ("expanding", "all history", BLUE),
                ("adaptive", "automatic", AQUA))
    for ax, d in zip(axes, NAMES):
        base = os.path.join(OUT, d, "window_selection")
        weekly = {v: pd.read_csv(os.path.join(base, v, "served_weekly.csv"))["pr_auc"]
                  for v in ("static", "sliding", "expanding", "adaptive")}
        for i, (v, label, color) in enumerate(variants):
            diff = (weekly[v] - weekly["static"]).dropna().to_numpy()
            boots = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
            mean, lo, hi = diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)
            ax.bar(i, mean, color=color, width=0.62, edgecolor=SURFACE, lw=2)
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], color=INK, lw=1, capsize=3)
            ax.annotate(f"{mean:+.3f}", (i, hi if mean >= 0 else lo), textcoords="offset points",
                        xytext=(0, 4 if mean >= 0 else -11), ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(3), [lab for _, lab, _ in variants])
        ax.set_title(NAMES[d])
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(lo_y - (hi_y - lo_y) * 0.08, hi_y + (hi_y - lo_y) * 0.08)
        style(ax, zero=True)
    axes[0].set_ylabel("PR-AUC gain over static")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "results_window.png"), dpi=200)
    plt.close(fig)


def gate():
    """Fault injection on IEEE-CIS: loss against the clean replay with the gate on and off."""
    r = pd.read_csv(os.path.join(OUT, "gate_test", "gate_results.csv"))
    r = r[r["scenario"] != "clean"]
    names = {"label_shuffle": "label shuffle", "label_loss": "label loss (80%)",
             "feature_unit": "feature unit bug", "upstream_label_shuffle": "upstream label\nshuffle"}
    order = list(names)
    fig, ax = plt.subplots(figsize=(8, 3.3))
    for k, (g, label, color) in enumerate((("on", "gate on", AQUA), ("off", "gate off", ORANGE))):
        sub = r[r["gate"] == g].set_index("scenario").loc[order]
        y = np.arange(len(order)) + (k - 0.5) * 0.36
        ax.barh(y, sub["delta_vs_clean"], height=0.34, color=color, edgecolor=SURFACE, lw=2,
                label=label)
        for yi, v in zip(y, sub["delta_vs_clean"]):
            ax.text(v - 0.002, yi, f"{v:+.3f}", va="center", ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)), [names[s] for s in order])
    ax.invert_yaxis()
    ax.set_xlim(r["delta_vs_clean"].min() * 1.35, 0.004)
    ax.axvline(0, color="#9a9893", lw=1)
    ax.set_xlabel("change in mean weekly PR-AUC against the clean replay")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "results_gate.png"), dpi=200)
    plt.close(fig)


def equal_budget():
    """Best mean weekly PR-AUC reachable with at most k retrains, per family: one row per
    dataset whose equal-budget run is available."""
    sources = [(n, p) for n, p in (("IEEE-CIS", os.path.join(OUT, "equal_budget")),
                                   ("Sparkov", os.path.join(OUT, "sparkov", "equal_budget")),
                                   ("BAF", os.path.join(OUT, "baf", "equal_budget")))
               if os.path.exists(os.path.join(p, "eb_runs.csv"))]
    keys = [(0.0, "60d"), (0.0, "all"), (30.0, "60d"), (30.0, "all")]
    fig, axes = plt.subplots(len(sources), 4, figsize=(11, 3.0 * len(sources)), squeeze=False)
    for row, (name, path) in zip(axes, sources):
        runs = pd.read_csv(os.path.join(path, "eb_runs.csv"))
        for ax, (delay, window) in zip(row, keys):
            g = runs[(runs["label_delay"] == delay) & (runs["window"] == window)]
            for fam, colour, label in (("schedule", BLUE, "schedule"),
                                       ("trigger", ORANGE, "drift trigger")):
                f = g[g["family"] == fam]
                ax.scatter(f["retrains"], f["mean_pr"], s=16, color=colour, alpha=0.45, lw=0)
                best = f.groupby("retrains")["mean_pr"].max().sort_index().cummax()
                ax.step(best.index, best.values, where="post", color=colour, lw=2, label=label)
            ax.axhline(g[g["family"] == "static"]["mean_pr"].iloc[0], color=MUTED, lw=1,
                       ls="--", label="static")
            ax.set_title(f"{name}: delay {delay:g} d, "
                         f"{'last 60 days' if window == '60d' else 'all history'}", fontsize=9)
            ax.set_xlabel("number of retrains")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            style(ax)
        row[0].set_ylabel("mean weekly PR-AUC")
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "results_equal_budget.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    equal_budget()
    decay()
    delay()
    window()
    gate()
    print("results figures written")
