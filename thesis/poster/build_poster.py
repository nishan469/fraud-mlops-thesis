"""P2 poster (48 x 36 in, landscape), following the layout of the BRAC P2 poster template:
navy header with title, presenters and supervisors, three columns of sections (pink problem
statement box, grey analysis box), and the BRAC footer. Names, IDs and supervisors come from
the thesis content (../p1/content.py), so they always match the P1/P2 documents.

  python build_poster.py   ->  P2_Poster_Drift_Aware_MLOps.pdf
"""

import os
import sys

import matplotlib
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "..", "p1")
sys.path.insert(0, P1)
import content as C  # noqa: E402

OUT = os.path.join(HERE, "P2_Poster_Drift_Aware_MLOps.pdf")
W, H = 48 * inch, 36 * inch
NAVY, PINK, GREY, INK = HexColor("#003C77"), HexColor("#FFF5F2"), HexColor("#F0F0F0"), HexColor("#1A1A1A")
MARGIN, GAP, HEADER, FOOTER = 1.0 * inch, 0.9 * inch, 6.3 * inch, 1.0 * inch
COL_W = (W - 2 * MARGIN - 2 * GAP) / 3

ttf = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("Sans", os.path.join(ttf, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", os.path.join(ttf, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Sans-It", os.path.join(ttf, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-It",
                              boldItalic="Sans-Bold")

BODY = ParagraphStyle("body", fontName="Sans", fontSize=24, leading=33, alignment=TA_JUSTIFY,
                      textColor=INK, spaceAfter=12)
ITEM = ParagraphStyle("item", parent=BODY, alignment=TA_LEFT, leftIndent=34, firstLineIndent=-34,
                      spaceAfter=6)
HEAD = ParagraphStyle("head", fontName="Sans-Bold", fontSize=34, leading=42, alignment=TA_CENTER,
                      textColor=NAVY, spaceBefore=14, spaceAfter=4)
CAP = ParagraphStyle("cap", fontName="Sans", fontSize=19, leading=24, alignment=TA_CENTER,
                     textColor=HexColor("#444444"), spaceBefore=4, spaceAfter=16)
REF = ParagraphStyle("ref", fontName="Sans", fontSize=18, leading=23, textColor=HexColor("#333333"),
                     leftIndent=44, firstLineIndent=-44, spaceAfter=8)
CELL = ParagraphStyle("cell", fontName="Sans", fontSize=19, leading=24, textColor=INK)
CELLH = ParagraphStyle("cellh", parent=CELL, fontName="Sans-Bold", textColor=white)


def heading(text, inside=0):
    """Section title with a rule underneath, as in the template."""
    t = Table([[Paragraph(text, HEAD)]], colWidths=[COL_W - 2 * inside])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 2.2, HexColor("#555555")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    return [Spacer(1, 10), t, Spacer(1, 14)]


def para(text):
    return Paragraph(text, BODY)


def figure(name, caption, width=1.0, inside=0):
    """`inside`: horizontal padding of an enclosing box (then no KeepTogether, which a table
    cell cannot hold)."""
    img = Image(os.path.join(P1, "figures", name))
    w = (COL_W - 2 * inside) * width
    img.drawWidth, img.drawHeight = w, img.imageHeight * w / img.imageWidth
    parts = [img, Paragraph(caption, CAP)]
    return parts if inside else [KeepTogether(parts)]


def box(flowables, colour, pad=22):
    t = Table([[flowables]], colWidths=[COL_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colour),
                           ("LEFTPADDING", (0, 0), (-1, -1), pad), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
                           ("TOPPADDING", (0, 0), (-1, -1), pad * 0.6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), pad * 0.8)]))
    return [t]


def numbered(items):
    return [Paragraph(f"{i}.&nbsp;&nbsp;&nbsp;{t}", ITEM) for i, t in enumerate(items, 1)]


def bullets(items):
    return [Paragraph(f"•&nbsp;&nbsp;&nbsp;{t}", ITEM) for t in items]


def table(rows, widths):
    data = [[Paragraph(c, CELLH if r == 0 else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * COL_W for w in widths])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F3F6FA")]),
                           ("LINEBELOW", (0, -1), (-1, -1), 1.5, NAVY),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return [t, Spacer(1, 14)]


# ------------------------------------------------------------------------------ content
PRE_W, METH_W, DECAY_W, RES_W, GATE_W = 0.78, 0.8, 0.9, 1.0, 0.9   # figure widths

# The thesis references in a compact IEEE form (full entries with DOIs are in P1 and P2).
# Only those cited on the poster are listed, numbered in order of first citation.
REFS = {
    "dalpozzolo2018": "A. Dal Pozzolo et al., \"Credit card fraud detection: A realistic modeling and a novel learning strategy,\" <i>IEEE Trans. Neural Netw. Learn. Syst.</i>, vol. 29, no. 8, 2018.",
    "menezes2025": "R. S. Menezes and R. H. Filho, \"Semantic and structural drift in financial knowledge graphs: A robustness analysis of GNN-based fraud detectors,\" in <i>Proc. IEEE ICKG</i>, 2025.",
    "amekoe2024": "K. M. Amekoe et al., \"Evaluating the efficacy of instance incremental vs. batch learning in delayed label environments,\" arXiv:2409.10111, 2024.",
    "shakil2025": "M. N. P. Shakil et al., \"Feature drift-guided adaptive ML retraining: An MLOps approach for big data analytics,\" in <i>Proc. IEEE BigData</i>, 2025.",
    "wong2025": "H. M. Wong and S. Perumal, \"AI-driven model-retraining architecture to sustain operational accuracy in data-drifting environments,\" in <i>Proc. IEEE ISWTA</i>, 2025.",
    "uddin2026": "M. N. Uddin and M. M. Aziz, \"Shapley value-guided adaptive ensemble learning for explainable financial fraud detection,\" arXiv:2604.14231, 2026.",
    "yelleti2025": "V. Yelleti, \"ROSFD: Robust online streaming fraud detection with resilience to concept drift in data streams,\" arXiv:2504.10229, 2025.",
    "aldaoud2025": "K. I. Al-Daoud and I. A. Abu-AlSondos, \"Robust AI for financial fraud detection in the GCC,\" <i>J. Theor. Appl. Electron. Commer. Res.</i>, vol. 20, no. 2, 2025.",
    "somasundaram2019": "A. Somasundaram and S. Reddy, \"Parallel and incremental credit card fraud detection model to handle concept drift and data imbalance,\" <i>Neural Comput. Appl.</i>, vol. 31, 2019.",
    "alessi2026": "G. Alessi and M. Fugini, \"Adaptive real-time financial fraud detection with explainable AI tools,\" <i>Digital Threats: Res. Pract.</i>, vol. 7, no. 1, 2026.",
    "hassan2026": "Y. Hassan, \"DriftGuard-TriAudit: Concept-drift-aware continual multimodal learning for evolving financial statement fraud detection,\" <i>J. Comput. Electron. Inf. Manage.</i>, 2026.",
    "reda2025": "A. Reda et al., \"Hybrid MLOps framework for automated lifecycle management of adaptive phishing detection models,\" <i>Scientific Reports</i>, vol. 15, 2025.",
    "pulicharla2019": "M. R. Pulicharla, \"Detecting and addressing model drift: Automated monitoring and real-time retraining in ML pipelines,\" <i>World J. Adv. Res. Rev.</i>, vol. 3, 2019.",
    "kodakandla2024": "N. Kodakandla, \"Data drift detection and mitigation: A comprehensive MLOps approach for real-time systems,\" <i>Int. J. Sci. Res. Arch.</i>, vol. 12, 2024.",
    "kaushik2025": "S. Kaushik et al., \"Real-time analytics with intelligent data pipelines and ML detection,\" in <i>Proc. ICSIT</i>, 2025.",
    "john2025": "S. John, \"Fair and explainable credit-scoring under concept drift: Adaptive explanation frameworks for evolving populations,\" arXiv:2511.03807, 2025.",
    "mienye2023": "I. D. Mienye and Y. Sun, \"A deep learning ensemble with data resampling for credit card fraud detection,\" <i>IEEE Access</i>, vol. 11, 2023.",
    "sohony2018": "I. Sohony, R. Pratap, and U. Nambiar, \"Ensemble learning for credit card fraud detection,\" in <i>Proc. CoDS-COMAD</i>, 2018.",
    "abdelnaby2023": "A. Abd El-Naby et al., \"An efficient fraud detection framework with credit card imbalanced data in financial services,\" <i>Multimed. Tools Appl.</i>, vol. 82, 2023.",
    "dang2021": "T. Dang et al., \"Machine learning based on resampling approaches and deep reinforcement learning for credit card fraud detection systems,\" <i>Applied Sciences</i>, vol. 11, 2021.",
}
N = {}


def cite(*keys):
    """IEEE-style citation; numbers follow the order of first use on the poster."""
    for k in keys:
        if k not in REFS:
            raise KeyError(f"Unknown reference: {k}")
        N.setdefault(k, len(N) + 1)
    return "[" + ", ".join(str(N[k]) for k in keys) + "]"


def references():
    """The references cited on the poster, in order of first citation."""
    return [Paragraph(f"[{n}]&nbsp;&nbsp;{REFS[k]}", REF)
            for k, n in sorted(N.items(), key=lambda kv: kv[1])]


def story():
    N.clear()
    s = []
    # ---------------------------------------------------------------- column 1
    s += heading("Abstract")
    s.append(para(
        "Fraud models are usually trained once, but in production fraud patterns change "
        "(<b>concept drift</b>) and true labels arrive weeks late (<b>label delay</b>). We build "
        "and evaluate a <b>drift-aware continuous learning MLOps framework</b> that monitors a "
        "deployed model on delayed labels, retrains it on a schedule with an automatically "
        "chosen training window, and deploys a new model only if it passes a "
        "<b>promotion gate</b>. It is tested on three public datasets with label delays of "
        "0 to 60 days."))
    s += heading("Problem Statement")
    s += box([para("Existing studies test fraud models on <b>random splits</b>, assume "
                   "<b>immediate labels</b> and trigger retraining with <b>drift detectors</b> "
                   "on artificial drift. When should a real fraud model be retrained, and how "
                   "can a bad model be kept out of production?")], PINK)
    s += heading("Research Objectives")
    s += numbered([
        "Measure how a deployed fraud model decays and whether drift measures detect it.",
        "Measure how label delay limits the benefit of retraining.",
        "Compare scheduled and drift-triggered retraining fairly.",
        "Build an MLOps framework with monitoring, retraining, window selection and a "
        "promotion gate, and test it with injected faults.",
        "Replicate on three public datasets."])
    s += heading("Literature Review")
    s += bullets([
        "<b>Dal Pozzolo et al.</b> " + cite("dalpozzolo2018") + ": fraud labels arrive late "
        "(verification latency), which limits how a model can be updated.",
        "<b>Menezes and Filho</b> " + cite("menezes2025") + ": fraud models on IEEE-CIS lose "
        "up to 40% F1 under natural drift; no retraining tested.",
        "<b>Shakil et al.</b> " + cite("shakil2025") + ": drift-triggered retraining for "
        "MLOps, tested only on injected drift with immediate labels."])
    s.append(para("<b>Gap:</b> no fair test of retraining policies for fraud under real "
                  "drift and label delay."))
    s += heading("Data set")
    s += table([["", "IEEE-CIS", "Sparkov", "BAF"],
                ["Type", "Real e-commerce", "Simulated cards", "Bank accounts"],
                ["Records", "590,540", "1,801,969", "1,000,000"],
                ["Time span", "182 days", "720 days", "8 months"],
                ["Fraud rate", "3.50%", "0.53%", "1.10%"],
                ["Features", "431", "16", "29"]], [0.25, 0.25, 0.25, 0.25])
    s.append(Paragraph("<b>Preprocessing</b>", ParagraphStyle("sub", parent=BODY, spaceAfter=4)))
    s += bullets([
        "<b>IEEE-CIS:</b> transaction and identity tables joined on TransactionID; text "
        "columns as categories; missing values kept.",
        "<b>Sparkov:</b> personal fields removed; cut at 21 Dec 2020 (fraud stops); age, hour, "
        "distance and card-history features from <i>earlier</i> transactions only.",
        "<b>BAF:</b> -1 codes set to missing; month not used as a feature.",
        "<b>All:</b> sorted by time, never shuffled; no oversampling (fraud class weighted)."])
    s += figure("preprocessing.png", "Preprocessing pipeline from raw files to the time-ordered "
                "table used by every experiment", PRE_W)
    s.append(FrameBreak())
    # ---------------------------------------------------------------- column 2
    s += heading("Methodology")
    s += figure("methodology_3ds.png", "Workflow of the study", METH_W)
    s += figure("model_architecture.png", "Model architecture: LightGBM gradient-boosted "
                "decision trees", METH_W)
    s += figure("architecture.png", "Proposed framework: each week it monitors, decides, "
                "retrains, gates and serves", METH_W)
    first = heading("Result Analysis", inside=20)
    first += figure("results_decay.png", "<b>Drift measures do not track performance:</b> "
                    "IEEE-CIS decays while drift stays low; Sparkov and BAF drift without "
                    "decaying", DECAY_W, inside=20)
    s += box(first, GREY, pad=20)
    s.append(FrameBreak())
    # ---------------------------------------------------------------- column 3
    rest = []
    rest += figure("results_delay.png", "<b>Label delay erodes retraining:</b> on IEEE-CIS "
                   "the gain falls from +0.078 to +0.010 PR-AUC", RES_W, inside=20)
    rest += figure("results_window.png", "<b>Automatic window selection</b> matches the best "
                   "fixed window without hindsight", RES_W, inside=20)
    rest += figure("results_gate.png", "<b>The promotion gate</b> blocked every model "
                   "trained on corrupted data", GATE_W, inside=20)
    s += box(rest, GREY, pad=20)
    s += heading("Conclusion & Future Work")
    s += bullets([
        "<b>Drift indicators can miss real degradation under delayed labels</b>, while "
        "scheduled retraining was more reliable. Here, <i>drift-aware</i> means tracking "
        "drift through delayed-label performance, not label-free alarms.",
        "With delayed labels, the schedule matched or beat the drift trigger even at an "
        "<b>equal retraining budget</b> (IEEE-CIS, BAF): not only from retraining more often.",
        "<b>Window selection</b> and a <b>promotion gate</b> make retraining adaptive and safe; "
        "results replicated on <b>two more timestamped datasets</b> (Sparkov, BAF).",
        "<b>Next (supervisor feedback):</b> equal-budget comparison on Sparkov, and an "
        "<b>API load test</b> of the scoring service (throughput, p95/p99 latency).",
        "<b>Later:</b> faster investigator feedback, more model families, explanation "
        "monitoring and a live deployment."])
    s += heading("References")
    s += references()
    return s


# ------------------------------------------------------------------------------ page
def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, H - HEADER, W, HEADER, stroke=0, fill=1)
    canvas.rect(0, 0, W, FOOTER, stroke=0, fill=1)
    logo = os.path.join(HERE, "brac_logo.png")
    lh = 3.6 * inch
    canvas.drawImage(logo, W - MARGIN - lh * 587 / 534, H - HEADER + 1.3 * inch, lh * 587 / 534, lh)
    title = Paragraph(C.TITLE, ParagraphStyle("t", fontName="Sans-Bold", fontSize=66, leading=80,
                                              alignment=TA_CENTER, textColor=white))
    people = [f"{n} ({i})" for n, i in C.AUTHORS]
    half = (len(people) + 1) // 2
    names = ", ".join(people[:half]) + ",<br/>" + ", ".join(people[half:])
    sup, co = C.META["supervisor"][0], C.META["cosupervisor"][0]
    sub = Paragraph(f"Presented by {names}<br/>Supervisor: {sup}&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
                    f"Co-Supervisor: {co}",
                    ParagraphStyle("s", fontName="Sans", fontSize=34, leading=46,
                                   alignment=TA_CENTER, textColor=white))
    uni = Paragraph(C.META["university"].replace("Brac", "BRAC"),
                    ParagraphStyle("u", fontName="Sans", fontSize=26, leading=32,
                                   alignment=TA_CENTER, textColor=white))
    tw = W - 2 * MARGIN - 2 * 4.6 * inch
    y = H - 0.55 * inch
    for p, gap in ((title, 0.25 * inch), (sub, 0.2 * inch), (uni, 0)):
        _, ph = p.wrap(tw, H)
        y -= ph
        p.drawOn(canvas, (W - tw) / 2, y)
        y -= gap
    canvas.setFillColor(white)
    canvas.setFont("Sans", 25)
    fy = 0.38 * inch
    canvas.drawString(MARGIN, fy, "https://www.bracu.ac.bd")
    canvas.drawCentredString(W / 2, fy, "BRAC University | Kha 224 Pragati Sarani, Merul Badda, "
                             "Dhaka 1212 | Tel: +88 09638464646")
    canvas.drawRightString(W - MARGIN, fy, f"© {C.META['copyright_year']} Brac University. "
                           "All rights reserved.")
    canvas.restoreState()


def column_heights():
    """Height each column needs (points), to check the layout fits."""
    cols, cur = [], []
    for f in story():
        if isinstance(f, type(FrameBreak)):
            cols.append(cur); cur = []
        else:
            cur.append(f)
    cols.append(cur)
    out = []
    for col in cols:
        h = 0
        flat = [g for f in col for g in (f._content if isinstance(f, KeepTogether) else [f])]
        for f in flat:
            _, fh = f.wrap(COL_W, H)
            h += fh + f.getSpaceBefore() + f.getSpaceAfter()
        out.append(round(h / inch, 1))
    return out


def main():
    top, bottom = H - HEADER - 0.5 * inch, FOOTER + 0.5 * inch
    frames = [Frame(MARGIN + i * (COL_W + GAP), bottom, COL_W, top - bottom, id=f"c{i}",
                    leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
              for i in range(3)]
    doc = BaseDocTemplate(OUT, pagesize=(W, H), title=f"P2 Poster: {C.TITLE}",
                          author=", ".join(n for n, _ in C.AUTHORS))
    doc.addPageTemplates([PageTemplate(id="poster", frames=frames, onPage=header_footer)])
    avail = (top - bottom) / inch
    print(f"column heights (in): {column_heights()}  available: {avail:.1f}")
    doc.build(story())
    print(f"Wrote {OUT} ({doc.page} page{'s' if doc.page > 1 else ''})")


if __name__ == "__main__":
    main()
