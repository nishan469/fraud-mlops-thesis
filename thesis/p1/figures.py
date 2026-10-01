"""Figures for the P1: methodology flow (Fig. 1.1) and framework loop (Fig. 1.2)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, ACCENT, FILL, FILL2 = "#15233B", "#4A5363", "#2F6BD6", "#EEF2F8", "#FDF3EA"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, title, body, fill=FILL, edge=ACCENT):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8.6, color=MUTED,
            linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=MUTED, lw=1.4, connectionstyle=f"arc3,rad={rad}"))


def methodology(path):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    w, h = 2.25, 1.4
    top = [("1. Literature review", "Fraud detection, concept\ndrift, label delay,\nretraining, MLOps"),
           ("2. Data preparation", "IEEE-CIS and Sparkov,\nstrict time order,\nleak-free card features"),
           ("3. Static baseline", "Weekly PR-AUC after\ndeployment; PSI drift of\nfeatures and scores"),
           ("4. Retraining simulation", "Static, scheduled and\ndrift-triggered policies;\nlabel delay 0-60 days")]
    bottom = [("8. Results and thesis", "Answer RQ1-RQ4;\nlimitations and\nfuture work"),
              ("7. Fault injection and\nreplication", "Corrupted labels, outage,\nunit bug; repeat key\nexperiments on Sparkov"),
              ("6. MLOps framework", "Monitor, decide, retrain,\npromotion gate, serve;\nMLflow, API, dashboard"),
              ("5. Statistical validation", "Day-block bootstrap CIs,\nwalk-forward tuning,\n5 training seeds")]
    xs = [0.15, 2.65, 5.15, 7.65]
    for (t, b), x in zip(top, xs):
        box(ax, x, 2.8, w, h, t, b)
    for (t, b), x in zip(bottom, xs):
        box(ax, x, 0.4, w, h, t, b, fill=FILL2 if t.startswith(("6", "7")) else FILL)
    for i in range(3):
        arrow(ax, xs[i] + w, 2.8 + h / 2, xs[i + 1], 2.8 + h / 2)
        arrow(ax, xs[i + 1], 0.4 + h / 2, xs[i] + w, 0.4 + h / 2)
    arrow(ax, xs[3] + w / 2, 2.8, xs[3] + w / 2, 0.4 + h)
    ax.text(5.0, 4.4, "Offline experiments", ha="center", fontsize=9, color=ACCENT, style="italic")
    ax.text(5.0, 0.08, "Framework and evaluation", ha="center", fontsize=9, color="#C8581E",
            style="italic")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def framework_loop(path):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    w, h = 1.72, 1.35
    steps = [("Monitor", "PR-AUC on labels\nthat have arrived;\ndrift logged only"),
             ("Decide", "Every 14 days, or\nearly if the safety\nnet fires"),
             ("Retrain", "LightGBM on the\nlast 60 days of\nlabelled data"),
             ("Promotion gate", "New model must\nmatch the live one\non unseen data"),
             ("Serve", "Live model scores\nthe next week of\ntransactions")]
    xs = [0.1 + i * 2.0 for i in range(5)]
    for (t, b), x in zip(steps, xs):
        box(ax, x, 2.85, w, h, t, b, edge=ACCENT if t != "Promotion gate" else "#C8581E")
    for i in range(4):
        arrow(ax, xs[i] + w, 2.85 + h / 2, xs[i + 1], 2.85 + h / 2)
    arrow(ax, xs[4] + w / 2, 2.85, xs[0] + w / 2, 2.85, rad=-0.12)
    ax.text(5.0, 1.72, "repeated every 7 days on the transaction stream", ha="center", fontsize=9,
            color=MUTED, style="italic")
    support = [("MLflow registry", "Versioned models;\nlive and previous\nversion for rollback"),
               ("Scoring API", "FastAPI service for\nthe live model"),
               ("Monitoring dashboard", "Performance, drift,\ndecisions, versions")]
    for i, (t, b) in enumerate(support):
        box(ax, 0.9 + i * 2.9, 0.1, 2.4, 1.3, t, b, fill="#F6F5F1", edge="#8A96A8")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def drift_types(path):
    """Four panels: the underlying fraud concept over time for each type of drift."""
    import numpy as np
    rng = np.random.default_rng(3)
    t = np.linspace(0, 10, 400)
    concepts = {
        "Sudden": np.where(t < 5, 0.0, 1.0),
        "Gradual": None,
        "Incremental": np.clip((t - 3) / 4, 0, 1),
        "Recurring": ((t // 2.5) % 2).astype(float),
    }
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.3), sharey=True)
    for ax, (name, c) in zip(axes, concepts.items()):
        if name == "Gradual":
            # samples switch between the old and the new concept, more often the new one over time
            p_new = np.clip((t - 2.5) / 5, 0, 1)
            pts = (rng.uniform(size=t.size) < p_new).astype(float)
            ax.scatter(t[::4], pts[::4] + rng.normal(0, 0.04, t[::4].size), s=6, color=ACCENT,
                       alpha=0.8)
        else:
            ax.plot(t, c, color=ACCENT, lw=2.2)
        ax.set_title(name, fontsize=11, color=INK, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["old\npattern", "new\npattern"], fontsize=8.5, color=MUTED)
        ax.set_ylim(-0.3, 1.3)
        ax.set_xlabel("time", fontsize=9, color=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#9AA3AF")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def label_delay(path):
    """Timeline: scoring is immediate, labels mature after the delay; what monitoring and
    retraining can use at time 'now'."""
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(0, 3.25)
    ax.axis("off")
    now, delay = 8.6, 3.6
    matured = now - delay
    y = 1.35
    # time axis
    ax.add_patch(FancyArrowPatch((0, y), (10.1, y), arrowstyle="-|>", mutation_scale=16,
                                 color=INK, lw=1.4))
    ax.text(10.1, y - 0.3, "time", ha="right", fontsize=9, color=MUTED)
    # regions
    ax.add_patch(plt.Rectangle((0, y), matured, 0.55, fc=FILL, ec=ACCENT, lw=1.2))
    ax.text(matured / 2, y + 0.275, "labels known (matured)", ha="center", va="center",
            fontsize=9.5, color=INK)
    ax.add_patch(plt.Rectangle((matured, y), delay, 0.55, fc=FILL2, ec="#C8581E", lw=1.2))
    ax.text(matured + delay / 2, y + 0.275, "scored, labels not yet known", ha="center",
            va="center", fontsize=9.5, color=INK)
    # retraining window and monitoring window above
    win0 = matured - 3.2
    ax.annotate("", xy=(win0, 2.45), xytext=(matured, 2.45),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.3))
    ax.text((win0 + matured) / 2, 2.6, "training window for a new model\n(sliding, most recent labels)",
            ha="center", va="bottom", fontsize=8.8, color=ACCENT)
    mon0 = matured - 1.2
    ax.annotate("", xy=(mon0, 2.05), xytext=(matured, 2.05),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
    ax.text(mon0 - 0.1, 2.05, "latest monitoring window", ha="right", va="center",
            fontsize=8.5, color=MUTED)
    # markers
    for x, label in ((matured, "now - label delay"), (now, "now")):
        ax.plot([x, x], [y - 0.12, y + 0.7], color=INK, lw=1.2)
        ax.text(x, y - 0.3, label, ha="center", va="top", fontsize=9.5, color=INK,
                fontweight="bold")
    ax.annotate("", xy=(matured, 0.55), xytext=(now, 0.55),
                arrowprops=dict(arrowstyle="<->", color="#C8581E", lw=1.3))
    ax.text((matured + now) / 2, 0.25, "label delay (e.g. 30 days until a chargeback is known)",
            ha="center", fontsize=8.8, color="#C8581E")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def protocol(path):
    """Evaluation protocol on one dataset: history, tuning stream, evaluated stream, weekly
    steps, walk-forward folds, and what one step of the simulation does."""
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    x0, x1, xt, xs = 0.3, 9.7, 3.6, 5.2      # start, end, tuning start, stream start
    y = 3.0
    ax.add_patch(plt.Rectangle((x0, y), xs - x0, 0.5, fc=FILL, ec=ACCENT, lw=1.2))
    ax.add_patch(plt.Rectangle((xs, y), x1 - xs, 0.5, fc=FILL2, ec="#C8581E", lw=1.2))
    ax.text((x0 + xs) / 2, y + 0.25, "history: initial training data", ha="center", va="center",
            fontsize=9.5, color=INK)
    ax.text((xs + x1) / 2, y + 0.25, "evaluated stream (replayed week by week)", ha="center",
            va="center", fontsize=9.5, color=INK)
    for x in np.arange(xs, x1 + 1e-9, 0.36):
        ax.plot([x, x], [y - 0.08, y], color="#C8581E", lw=0.8)
    ax.annotate("", xy=(xt, y + 0.78), xytext=(x1, y + 0.78),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
    ax.text((xt + x1) / 2, y + 0.9, "walk-forward tuning stream (all settings run)", ha="center",
            fontsize=8.6, color=MUTED)
    for x, lab in ((x0, "first\ntransaction"), (xs, "stream start t0\n(50-60% of rows)"),
                   (x1, "last\ntransaction")):
        ax.text(x, y - 0.2, lab, ha="center", va="top", fontsize=8.4, color=INK)
    # walk-forward folds
    fy = 2.0
    fw = (x1 - xs) / 4
    for i, x in enumerate(np.arange(xs, x1 - 1e-9, fw)):
        w = fw
        ax.add_patch(plt.Rectangle((x, fy), w - 0.05, 0.32, fc="#F6F5F1", ec="#8A96A8", lw=1))
        ax.text(x + w / 2, fy + 0.16, f"fold {i + 1}" if i < 3 else "...", ha="center", va="center", fontsize=8,
                color=MUTED)
    ax.text(xs - 0.1, fy + 0.16, "4-week folds: settings chosen\nfrom matured labels only",
            ha="right", va="center", fontsize=8.4, color=MUTED)
    # one step
    steps = [("1. Labels available", "transactions older than\nnow - label delay"),
             ("2. Decide", "strategy rule: schedule,\nscore PSI or PR-AUC drop"),
             ("3. Retrain (if decided)", "on all or the last W days\nof labelled data"),
             ("4. Serve and score", "next 7 days; weekly\nPR-AUC recorded")]
    for i, (t, b) in enumerate(steps):
        bx = 0.3 + i * 2.4
        ax.add_patch(FancyBboxPatch((bx, 0.12), 2.1, 1.25, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=FILL, ec=ACCENT, lw=1.2))
        ax.text(bx + 1.05, 1.22, t, ha="center", va="top", fontsize=9.3, fontweight="bold", color=INK)
        ax.text(bx + 1.05, 0.88, b, ha="center", va="top", fontsize=8.3, color=MUTED, linespacing=1.3)
        if i < 3:
            arrow(ax, bx + 2.1, 0.75, bx + 2.4, 0.75)
    ax.text(5.0, 1.52, "every 7 days of the stream, for each strategy and label delay:",
            ha="center", fontsize=8.8, color=ACCENT, style="italic")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def architecture(path):
    """Components of the framework and the data that flows between them."""
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    def comp(x, y, w, h, title, body, edge=ACCENT, fill=FILL):
        box(ax, x, y, w, h, title, body, fill=fill, edge=edge)

    comp(0.1, 3.5, 1.9, 1.3, "Transactions", "incoming stream,\nscored at once", edge="#8A96A8",
         fill="#F6F5F1")
    comp(2.6, 3.5, 2.1, 1.3, "Scoring API", "FastAPI; champion\nmodel scores and\nflags", )
    comp(5.4, 3.5, 2.0, 1.3, "Label store", "chargebacks arrive\nafter the label delay",
         edge="#8A96A8", fill="#F6F5F1")
    comp(8.0, 3.5, 1.9, 1.3, "Monitor", "live PR-AUC on\nmatured labels;\nPSI logged")
    comp(8.0, 1.6, 1.9, 1.3, "Retrain policy", "schedule (14 d)\n+ safety net")
    comp(5.4, 1.6, 2.0, 1.3, "Trainer", "LightGBM on the\ntraining window;\nchallenger")
    comp(2.6, 1.6, 2.1, 1.3, "Promotion gate", "challenger vs\nchampion on\nunseen data", edge="#C8581E")
    comp(0.1, 1.6, 1.9, 1.3, "MLflow registry", "versions; champion\nand previous\naliases")
    comp(2.6, 0.05, 4.8, 1.05, "Monitoring dashboard", "performance, drift, decisions and versions,\nread from the MLflow logs of every step",
         edge="#8A96A8", fill="#F6F5F1")
    arrow(ax, 2.0, 4.15, 2.6, 4.15)
    arrow(ax, 4.7, 4.15, 5.4, 4.15)
    arrow(ax, 7.4, 4.15, 8.0, 4.15)
    arrow(ax, 8.95, 3.5, 8.95, 2.9)
    arrow(ax, 8.0, 2.25, 7.4, 2.25)
    arrow(ax, 5.4, 2.25, 4.7, 2.25)
    arrow(ax, 2.6, 2.25, 2.0, 2.25)
    arrow(ax, 1.05, 2.9, 2.6, 3.75, rad=-0.2)
    ax.text(1.0, 3.25, "promote:\nnew champion", ha="center", fontsize=8, color="#C8581E")
    ax.annotate("", xy=(6.4, 2.9), xytext=(6.4, 3.5),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.text(6.5, 3.2, "labelled data", ha="left", va="center", fontsize=7.8, color=MUTED)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    protocol(os.path.join(HERE, "figures", "protocol.png"))
    architecture(os.path.join(HERE, "figures", "architecture.png"))
    methodology(os.path.join(HERE, "figures", "methodology.png"))
    framework_loop(os.path.join(HERE, "figures", "framework_loop.png"))
    drift_types(os.path.join(HERE, "figures", "drift_types.png"))
    label_delay(os.path.join(HERE, "figures", "label_delay.png"))
    print("figures written")
