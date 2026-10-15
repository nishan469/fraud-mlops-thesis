"""Presentation speech (5-6 minutes) for the P2 poster, with a plain-language explanation of
every poster block and figure. Output: P2_Presentation_Speech.pdf next to this file."""

import os
import sys

import matplotlib
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "..", "p1")
sys.path.insert(0, P1)
import content as C  # noqa: E402

OUT = os.path.join(HERE, "P2_Presentation_Speech.pdf")
FIG = os.path.join(P1, "figures")
NAVY, HOT, INK, MUTED = HexColor("#003C77"), HexColor("#B4501A"), HexColor("#1A1A1A"), HexColor("#555555")
SOFT, SPEECH, TIP = HexColor("#F3F6FA"), HexColor("#FBF7EF"), HexColor("#EEF6F0")

ttf = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("Sans", os.path.join(ttf, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", os.path.join(ttf, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Sans-It", os.path.join(ttf, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("Sans-BoldIt", os.path.join(ttf, "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-It",
                              boldItalic="Sans-BoldIt")

BODY = ParagraphStyle("body", fontName="Sans", fontSize=10, leading=14.5, textColor=INK, spaceAfter=6)
SAY = ParagraphStyle("say", parent=BODY, fontSize=11, leading=16.5, spaceAfter=7)
CUE = ParagraphStyle("cue", parent=BODY, fontName="Sans-It", fontSize=9, textColor=HOT, spaceAfter=3)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.8, leading=12, textColor=MUTED)
H1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=17, leading=21, textColor=NAVY,
                    spaceBefore=4, spaceAfter=10)
H2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=12.5, leading=16, textColor=NAVY,
                    spaceBefore=10, spaceAfter=5)
STAMP = ParagraphStyle("stamp", fontName="Sans-Bold", fontSize=10, leading=13, textColor=white)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=9, leading=12, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Sans-Bold")
CELLH = ParagraphStyle("cellh", parent=CELLB, textColor=white)
FW = A4[0] - 4 * cm   # frame width


def p(t, st=BODY):
    return Paragraph(t, st)


def boxed(flows, bg, pad=9, border=None):
    t = Table([[flows]], colWidths=[FW])
    st = [("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), pad),
          ("RIGHTPADDING", (0, 0), (-1, -1), pad), ("TOPPADDING", (0, 0), (-1, -1), pad),
          ("BOTTOMPADDING", (0, 0), (-1, -1), pad - 3)]
    if border:
        st.append(("LINEBEFORE", (0, 0), (0, -1), 3, border))
    t.setStyle(TableStyle(st))
    return t


def table(rows, widths, head=True):
    data = [[p(c, CELLH if head and r == 0 else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * FW for w in widths], repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#C9D3DF")),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
        st.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, SOFT]))
    t.setStyle(TableStyle(st))
    return [t, Spacer(1, 8)]


def image(name, width=FW):
    path = os.path.join(FIG, name)
    img = Image(path)
    img.drawHeight = width * img.imageHeight / img.imageWidth
    img.drawWidth = width
    return img


# ------------------------------------------------------------------------------ the speech
# (time stamp, section, [cue, paragraph, cue, paragraph ...]); cues start with ">".
SPEECH_PARTS = [
    ("0:00 - 0:45", "Opening and the problem", [
        "> Stand beside the title. Smile, look at the audience, not the poster.",
        "Good morning. We are Mashud, Nirnoy, Sabbir, Imran and Sami. Our thesis is a "
        "<b>drift-aware continuous learning MLOps framework for financial fraud detection</b>, "
        "supervised by Md. Sabbir Ahmed and co-supervised by Naima Tahsin Nodi.",
        "Banks use machine learning models to catch fraudulent transactions. Normally the model "
        "is trained once and tested once. But in real life two things go wrong. First, "
        "fraudsters keep changing their tricks, so the data slowly changes and a good model "
        "becomes weaker. This is called <b>concept drift</b>. Second, the bank only finds out "
        "that a transaction was fraud weeks later, when the customer complains. This is called "
        "<b>label delay</b>. So the model cannot see its own mistakes in time.",
        "> Point at the pink Problem Statement box.",
        "So our question is simple: <b>when should a fraud model be retrained, and how do we "
        "stop a bad new model from going live?</b>",
    ]),
    ("0:45 - 1:35", "Objectives and literature review", [
        "> Point at Research Objectives, then Literature Review.",
        "We had five objectives: measure how a deployed model gets worse and whether drift "
        "alarms notice it; measure how label delay reduces the benefit of retraining; compare "
        "scheduled and drift-triggered retraining fairly; build a full MLOps framework and test "
        "it with injected faults; and repeat everything on three datasets.",
        "From the literature: Dal Pozzolo and colleagues showed that fraud labels arrive late. "
        "Menezes and Filho showed that fraud models lose up to 40 percent of their F1-score under "
        "natural drift, but they did not test retraining. Shakil and colleagues built "
        "drift-triggered retraining for MLOps, but only on artificial drift with instant labels. "
        "So the <b>gap</b> is that nobody had fairly tested retraining policies for fraud with "
        "real drift and late labels.",
    ]),
    ("1:35 - 2:25", "Datasets and preprocessing", [
        "> Point at the dataset table, then the preprocessing diagram.",
        "We used three public datasets from Kaggle. <b>IEEE-CIS</b> is real online payments: "
        "about 590 thousand transactions over six months, 3.5 percent fraud. <b>Sparkov</b> is "
        "simulated card payments: 1.8 million transactions over two years, only half a percent "
        "fraud. <b>BAF</b> is bank account applications: one million rows over eight months, "
        "about one percent fraud. Using three different datasets shows our results are not "
        "luck on one dataset.",
        "For preprocessing, we merged and cleaned the files, removed personal information, and "
        "built features like hour, age, distance and card history, using only "
        "<b>earlier</b> transactions so the model never sees the future. Then we sorted "
        "everything by time and never shuffled. The first half is used to train the first "
        "model, and the rest is replayed week by week, just like real life.",
    ]),
    ("2:25 - 3:40", "Methodology, model and framework", [
        "> Point at the workflow diagram (8 boxes).",
        "Our study had eight steps: literature review, data preparation, a static baseline, a "
        "retraining simulation with label delays from 0 to 60 days, statistical checks with "
        "confidence intervals and five random seeds, then building the framework, testing it "
        "with faults, and replicating on all datasets.",
        "> Point at the model architecture diagram.",
        "Our model is <b>LightGBM</b>. It builds many small decision trees one after another, "
        "and each new tree fixes the mistakes of the earlier ones. Their outputs are added and "
        "turned into a fraud probability between 0 and 1. We chose it because it is fast, "
        "strong on table data, and handles missing values. We trained it ourselves; no "
        "pre-trained model is used.",
        "> Point at the framework diagram and follow the arrows with your hand.",
        "Our framework runs as a weekly loop. Transactions are scored by an API. Labels arrive "
        "late into a label store. The <b>monitor</b> checks performance on the labels that have "
        "arrived. The <b>retrain policy</b> retrains every 14 days, with a safety net if "
        "performance suddenly crashes. The trainer tries two training windows, the last 60 days "
        "and all history, and keeps the better one. Then the <b>promotion gate</b> compares the "
        "new model with the current one on unseen data, and only lets it go live if it is not "
        "worse. MLflow keeps every version, so we can roll back, and a dashboard shows every "
        "decision.",
    ]),
    ("3:40 - 5:15", "Results", [
        "> Point at the first result figure (top: lines, bottom: bars).",
        "Our main score is <b>PR-AUC</b>: how well the model puts the real frauds at the top of "
        "the list. First result: drift alarms do not track performance. On IEEE-CIS, "
        "performance falls from 0.58 to 0.39, but the drift score never crosses the alarm "
        "line. That is a <b>missed alarm</b>. On BAF, the drift score is huge, but performance "
        "stays flat. That is a <b>false alarm</b>.",
        "> Point at the label-delay figure.",
        "Second: label delay eats the benefit of retraining. On IEEE-CIS, retraining improves "
        "PR-AUC by 0.078 when labels arrive at once, but only by 0.010 when they arrive 60 days "
        "late. The window also matters: on Sparkov, a short window actually hurts.",
        "> Point at the window-selection bars.",
        "Third: our automatic window choice matched the best fixed window on every dataset, "
        "without knowing in advance which one was right.",
        "> Point at the gate figure.",
        "Fourth: we deliberately broke the data, for example by shuffling labels. Without the "
        "gate, performance dropped by up to 0.080. With the gate, only 0.012, because the bad "
        "models were blocked.",
        "And as our supervisor asked, we compared the two policies with the <b>same number of "
        "retrains</b>. With delayed labels, the schedule still matched or beat the drift "
        "trigger, so it does not win just because it retrains more often.",
    ]),
    ("5:15 - 6:00", "Conclusion and future work", [
        "> Point at Conclusion and Future Work.",
        "To conclude: when labels are late, drift alarms can miss real damage, and a simple "
        "schedule with the right training window is more reliable. In our work, "
        "<b>drift-aware</b> means watching drift through delayed-label performance, not through "
        "label-free alarms. Window selection and the promotion gate make retraining adaptive "
        "and safe, and the results held on three datasets.",
        "Our supervisor also asked us to load test the scoring API. The first test found a "
        "bottleneck; after fixing it, the API handles about <b>1,300 requests per second</b> "
        "and scores one transaction in about a millisecond, with exactly the same scores.",
        "Next, we will finish the equal-budget test on Sparkov. Later we want to try faster "
        "investigator feedback, other models and a live deployment. Thank you. We are happy "
        "to take your questions.",
    ]),
]


def speech():
    s = [p("Part B. The speech", H1),
         p("Read at a calm speed (about 150 words a minute). Orange lines are stage cues: do "
           "not read them aloud. Bold words are the ones to stress.", SMALL)]
    words = 0
    for stamp, title, items in SPEECH_PARTS:
        head = Table([[p(f"{stamp}&nbsp;&nbsp;|&nbsp;&nbsp;{title}", STAMP)]], colWidths=[FW])
        head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 9),
                                  ("TOPPADDING", (0, 0), (-1, -1), 5),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        body = []
        for it in items:
            if it.startswith(">"):
                body.append(p("[" + it[1:].strip() + "]", CUE))
            else:
                body.append(p(it, SAY))
                words += len(it.replace("<b>", "").replace("</b>", "").split())
        s += [Spacer(1, 6), KeepTogether([head, boxed(body, SPEECH, pad=10)])]
    return s, words


# ------------------------------------------------------------------------------ figure guides
def guide(title, img, width, rows, takeaway):
    """One poster figure: the picture, then what it shows / how to read it / numbers."""
    tbl = table([[f"<b>{k}</b>", v] for k, v in rows], [0.2, 0.8], head=False)
    take = boxed([p("<b>Say it in one line:</b> " + takeaway, BODY)], TIP, pad=8,
                 border=HexColor("#2E8B57"))
    head = [p(title, H2)]
    if img:
        head += [image(img, width), Spacer(1, 6)]
    return [KeepTogether(head + tbl[:1] + [Spacer(1, 4), take]), Spacer(1, 10)]


def explain():
    s = [p("Part C. The poster explained, block by block", H1),
         p("Each block of the poster in simple words. For every figure: what it shows, how to "
           "read it, the numbers to remember, and one line you can say about it.", SMALL)]

    s += [p("1. Abstract and Problem Statement", H2),
          p("<b>What the work is:</b> a system that keeps a fraud model healthy after it is "
            "deployed. It watches the model, retrains it on a schedule, picks how much past "
            "data to learn from, and only lets a new model go live if it passes a check (the "
            "promotion gate)."),
          p("<b>Why it is needed:</b> most papers (1) test on a random split, which mixes past "
            "and future and hides drift, (2) assume labels arrive instantly, which never happens "
            "with fraud, and (3) trigger retraining with drift detectors tested on fake, "
            "injected drift. We test under realistic conditions instead.")]

    s += [p("2. Research Objectives", H2)]
    s += table([["#", "Objective", "Where the answer is on the poster"],
                ["1", "How a deployed model decays, and whether drift measures detect it",
                 "Result figure 1 (decay)"],
                ["2", "How label delay limits the benefit of retraining", "Result figure 2 (delay)"],
                ["3", "Compare scheduled and drift-triggered retraining fairly",
                 "Conclusion (equal-budget point)"],
                ["4", "Build the framework and test it with injected faults",
                 "Framework diagram, result figures 3 and 4"],
                ["5", "Replicate on three datasets", "Every result figure has three panels"]],
                [0.06, 0.56, 0.38])

    s += [p("3. Literature Review (three papers on the poster)", H2)]
    s += table([["Paper", "What they found", "What was missing (our gap)"],
                ["Dal Pozzolo et al. (2018)", "Fraud labels arrive late (verification latency)",
                 "Did not test MLOps retraining policies"],
                ["Menezes and Filho (2025)", "Fraud models on IEEE-CIS lose up to 40% F1 under "
                 "natural drift", "Measured the damage, did not test retraining"],
                ["Shakil et al. (2025)", "Drift-triggered retraining inside an MLOps pipeline",
                 "Only artificial drift and instant labels"]], [0.27, 0.4, 0.33])
    s.append(p("<b>Gap in one line:</b> no fair test of retraining policies for fraud under "
               "real drift and label delay."))

    s += [p("4. Data set table", H2)]
    s += table([["", "IEEE-CIS", "Sparkov", "BAF"],
                ["What it is", "Real e-commerce payments (Vesta, Kaggle competition)",
                 "Simulated credit-card payments (Sparkov generator)",
                 "Bank account opening applications (NeurIPS 2022)"],
                ["Rows", "590,540", "1,801,969", "1,000,000"],
                ["Time span", "182 days", "720 days", "8 months"],
                ["Fraud rate", "3.50%", "0.53%", "1.10%"],
                ["Features", "431", "16", "29"],
                ["Why we chose it", "Real drift; the model really decays",
                 "Long time span with real timestamps (supervisor asked for a second "
                 "timestamped dataset)", "Built to contain monthly drift; a different kind of "
                 "fraud"]], [0.16, 0.28, 0.28, 0.28])
    s.append(p("<b>How to read it:</b> each column is one dataset. The fraud rate shows how rare "
               "fraud is: in Sparkov only about 5 in every 1,000 transactions. 'Features' is the "
               "number of input columns the model uses. All three are table data with a yes/no "
               "fraud label."))

    s += guide("5. Preprocessing diagram", "preprocessing.png", FW, [
        ("What it shows", "The five steps from raw downloaded files to the clean table that every "
                          "experiment uses."),
        ("How to read", "Left to right. (1) Raw files. (2) Merge and clean: join the two IEEE "
                        "tables, remove names and card numbers, treat -1 in BAF as missing, cut "
                        "Sparkov at 21 Dec 2020 because fraud stops there. (3) Make features "
                        "(hour, weekday, age, distance, card history) from earlier transactions "
                        "only. (4) Sort by time, never shuffle. (5) Split: the first 50-60% trains "
                        "the first model, the rest is replayed week by week."),
        ("Why it matters", "Shuffling would let the model see the future and would hide drift. "
                           "Using only earlier transactions for features avoids data leakage."),
    ], "We kept strict time order so the experiment behaves like a real bank.")

    s += guide("6. Workflow of the study", "methodology_3ds.png", FW, [
        ("What it shows", "The eight steps of the whole project."),
        ("How to read", "Top row left to right (offline experiments): literature, data, static "
                        "baseline, retraining simulation. Then down and right to left (framework "
                        "and evaluation): statistical validation, MLOps framework, fault "
                        "injection and replication, results."),
        ("Key terms", "<b>Static baseline</b>: train once, never update, and watch it decay. "
                      "<b>Bootstrap CI</b>: resample the days many times to see how sure a "
                      "difference is. <b>Walk-forward tuning</b>: choose settings using only the "
                      "past, then test on the next period. <b>5 seeds</b>: repeat training with "
                      "5 random starts to check the result is not luck."),
    ], "First we measured the problem, then built the framework, then tried to break it.")

    s += guide("7. Model architecture (LightGBM)", "model_architecture.png", FW, [
        ("What it shows", "How one transaction becomes a fraud decision."),
        ("How to read", "Input features (16 to 431 columns) go into many small decision trees. "
                        "Tree 2 learns from the mistakes of tree 1, tree 3 from tree 2, and so on "
                        "(this is 'gradient boosting'). The outputs are added into a score, the "
                        "score is turned into a probability between 0 and 1, and the transaction "
                        "is flagged if the probability is above a threshold."),
        ("Settings", "Up to 2,000 trees, 256 leaves each, learning rate 0.05, stops early when "
                     "validation stops improving. The rare fraud class gets a bigger weight "
                     "instead of creating fake fraud rows (no SMOTE)."),
        ("Why LightGBM", "Fast enough to retrain hundreds of times, best-known performer on "
                         "table data, handles missing values and categories by itself. Trained "
                         "from scratch, not pre-trained."),
    ], "Many small trees, each fixing the last one's mistakes, give a fraud probability.")

    s += guide("8. Proposed framework", "architecture.png", FW, [
        ("What it shows", "Our MLOps system and how data flows through it every week."),
        ("How to read", "Follow the arrows. <b>Transactions</b> are scored by the <b>Scoring "
                        "API</b> (FastAPI). True labels arrive late in the <b>Label store</b>. The "
                        "<b>Monitor</b> measures live PR-AUC on labels that have arrived. The "
                        "<b>Retrain policy</b> retrains every 14 days, plus a safety net. The "
                        "<b>Trainer</b> builds a new model (the challenger). The <b>Promotion "
                        "gate</b> compares it with the live model (the champion). If it passes, "
                        "<b>MLflow</b> registers it and it becomes the new champion. The "
                        "<b>Dashboard</b> shows everything."),
        ("Safety net", "Retrains early if live PR-AUC falls 30% below normal, or close to the "
                       "level of random guessing."),
        ("Window selection", "Each retrain trains two challengers, last 60 days and all history, "
                             "and keeps the one that scores better on the newest unseen labels."),
        ("Promotion gate", "Promotes only if the new model is at most 0.01 PR-AUC worse than the "
                           "current one, measured again through the real production features."),
    ], "A weekly loop: monitor, decide, retrain, check, then serve.")

    s += [KeepTogether([p("Part C (continued). The result figures", H1),
          boxed([p("<b>Before the graphs: what is PR-AUC?</b> The model gives every transaction "
                   "a fraud score. Sort them from highest to lowest. PR-AUC (between 0 and 1) "
                   "says how well the real frauds sit at the top of that list. 1 is perfect. A "
                   "model that guesses randomly gets about the fraud rate (0.035 on IEEE-CIS, "
                   "0.011 on BAF). That is why BAF's 0.16 is actually about 15 times better than "
                   "guessing. We use PR-AUC instead of accuracy because with 99% normal "
                   "transactions, a model that says 'never fraud' is 99% accurate but useless.",
                   BODY),
                 p("<b>What is PSI?</b> Population Stability Index: how much a feature's values "
                   "have changed compared to the training period. Below 0.1 = stable, 0.1 to 0.25 "
                   "= moderate, above 0.25 = major change (the usual alarm line).", BODY),
                 p("<b>Error bars</b> (the thin vertical lines with caps): the 95% confidence "
                   "interval. If the bar does not cross zero, the difference is reliable.", BODY)],
                SOFT, pad=10, border=NAVY)]), Spacer(1, 10)]

    s += guide("Result 1. Drift measures do not track performance", "results_decay.png", FW, [
        ("Question", "When the model gets worse, does the drift alarm notice?"),
        ("What it shows", "One column per dataset. <b>Top row</b>: PR-AUC of a model trained "
                          "once and never updated, over 10 time periods after deployment. "
                          "<b>Bottom row</b>: the PSI of the most changed feature. The dashed "
                          "line at 0.25 is the 'major change' alarm."),
        ("How to read", "Compare top and bottom in the same column. If the alarm worked, bars "
                        "would rise above the dashed line exactly when the line goes down."),
        ("Numbers", "<b>IEEE-CIS</b>: PR-AUC drops from 0.58 to 0.39, but PSI stays below 0.17, "
                    "never reaching 0.25: a <b>missed alarm</b>. <b>BAF</b>: PSI is 3 to 4.5, "
                    "about 15 times the alarm line, but PR-AUC stays around 0.15-0.19: a "
                    "<b>false alarm</b>. <b>Sparkov</b>: PR-AUC stays high (0.89-0.98); PSI "
                    "crosses the line only in the last period."),
        ("Why this chart", "Lines show change over time; bars show an amount at each time. Two "
                           "rows because PR-AUC and PSI have different scales."),
    ], "Drift alarms missed real damage on IEEE-CIS and raised false alarms on BAF.")

    s += guide("Result 2. Label delay erodes retraining", "results_delay.png", FW, [
        ("Question", "If labels arrive late, how much does retraining still help?"),
        ("What it shows", "X axis: label delay (0, 15, 30, 60 days). Y axis: how much better "
                          "retraining is than the never-updated model (PR-AUC gain). Blue = train "
                          "on all history, orange = train on the last 60 days. Error bars = 95% "
                          "confidence."),
        ("How to read", "Above the zero line means retraining helps; below means it hurts. A "
                        "falling line means the benefit shrinks as labels get later."),
        ("Numbers", "<b>IEEE-CIS</b>, last 60 days: +0.078 at 0 days, falling to +0.010 at 60 "
                    "days. All history: +0.033 to +0.008. <b>Sparkov</b>: last 60 days is far "
                    "below zero (-0.17 to -0.26) because 60 days hold too few frauds; all history "
                    "is about +0.02. <b>BAF</b>: all history about +0.01; last 60 days around "
                    "zero or slightly negative."),
        ("Why this chart", "A line shows the trend as delay grows; error bars show whether "
                           "differences are real."),
    ], "The later the labels, the less retraining helps, and the right window depends on the "
       "dataset.")

    s += guide("Result 3. Automatic window selection", "results_window.png", FW, [
        ("Question", "Can the framework pick the training window by itself?"),
        ("What it shows", "Full framework replays with a 30-day label delay. Three bars per "
                          "dataset: always last 60 days (orange), always all history (blue), "
                          "automatic choice (green). Height = PR-AUC gain over the "
                          "never-updated model."),
        ("How to read", "The best fixed window differs per dataset. Look whether green is close "
                        "to the taller of orange and blue."),
        ("Numbers", "<b>IEEE-CIS</b>: 60 days +0.050, all history +0.025, automatic +0.050. "
                    "<b>Sparkov</b>: 60 days -0.172, all history +0.019, automatic +0.018. "
                    "<b>BAF</b>: 60 days -0.008, all history +0.008, automatic +0.004 (its error "
                    "bar crosses zero, so roughly equal to no change)."),
        ("Why this chart", "Bars are the clearest way to compare a few options side by side."),
    ], "Automatic selection matched the best fixed window without knowing it in advance, and "
       "avoided the big loss on Sparkov.")

    s += guide("Result 4. The promotion gate", "results_gate.png", FW * 0.92, [
        ("Question", "Does the gate stop a bad model from going live?"),
        ("What it shows", "We broke the training data on purpose in four ways. For each, the "
                          "loss in average weekly PR-AUC compared with a clean run. Green = gate "
                          "on, orange = gate off. Longer bar = more damage."),
        ("The faults", "<b>Label shuffle</b>: fraud labels mixed up randomly. <b>Label loss "
                       "(80%)</b>: most labels missing. <b>Feature unit bug</b>: a feature in the "
                       "wrong unit (like cents instead of dollars). <b>Upstream label "
                       "shuffle</b>: labels corrupted at the source, so both models are judged "
                       "on wrong labels."),
        ("Numbers", "Gate off: -0.080, -0.039, -0.019. Gate on: only -0.012 each time. Upstream "
                    "shuffle: -0.080 either way."),
        ("Why the last row is the same", "The gate checks models with the labels it has. If the "
                                          "labels themselves are wrong at the source, the gate "
                                          "is fooled too. That case needs the safety net and "
                                          "data-quality checks. Be honest about this; it shows "
                                          "you understand the limit."),
    ], "The gate blocked every bad model trained on broken data, cutting the damage from up "
       "to 0.080 to 0.012.")

    s += [KeepTogether([p("9. Conclusion and Future Work", H2)] + table([["Poster point", "Meaning in simple words"],
                ["Drift indicators can miss real degradation under delayed labels",
                 "The usual alarms watch the inputs, not the results. They missed the real "
                 "drop on IEEE-CIS and cried wolf on BAF."],
                ["Drift-aware = tracking drift through delayed-label performance",
                 "We still care about drift, but we judge it by how the model actually performs "
                 "on the labels that have arrived."],
                ["Equal retraining budget",
                 "Supervisor's question: does the schedule win only because it retrains more? We "
                 "gave both the same number of retrains. On IEEE-CIS (60-day window) the schedule "
                 "was better in 18 of 27 pairs and the trigger was never reliably better; on BAF "
                 "with a 30-day delay the schedule also led."],
                ["Window selection and promotion gate", "Retraining adapts to each dataset and "
                 "stays safe from broken data."],
                ["API load test (done)", "Supervisor's request. The first run found that "
                 "converting the input took almost all of the time (about 14 requests per "
                 "second). After the fix, with identical scores: about 1,300 requests per "
                 "second, about 1 ms for one transaction, p95 latency 82 ms with 100 clients "
                 "at the same time, no errors."],
                ["Next", "Equal-budget test on Sparkov; repeat the load test on a server with "
                 "several worker processes."],
                ["Later", "Faster investigator feedback, other model families, monitoring of "
                 "explanations (SHAP), a live deployment."]], [0.38, 0.62]))]
    return s


def extras():
    s = [PageBreak(), p("Part D. Numbers to remember", H1)]
    s += table([["What", "Number"],
                ["Datasets", "3: IEEE-CIS (590,540 rows), Sparkov (1,801,969), BAF (1,000,000)"],
                ["Fraud rates", "3.5%, 0.53%, 1.1%"],
                ["Label delays tested", "0, 15, 30, 60 days"],
                ["Retrain schedule", "every 14 days; safety net at most once per 7 days"],
                ["Training windows", "last 60 days, or all history; chosen automatically"],
                ["Safety net", "30% drop below normal, or near random-guess level"],
                ["Promotion gate", "new model may be at most 0.01 PR-AUC worse"],
                ["Decay on IEEE-CIS", "PR-AUC 0.58 to 0.39, PSI never above 0.25"],
                ["Retraining gain, IEEE-CIS", "+0.078 (no delay) to +0.010 (60-day delay)"],
                ["Gate effect", "damage -0.080 without gate, -0.012 with gate"],
                ["API load test", "about 1,300 requests/s; 1 ms per transaction; p95 82 ms at "
                                  "100 clients (before the fix: 14 requests/s)"],
                ["Robustness", "5 random seeds, walk-forward tuning, bootstrap confidence "
                               "intervals"],
                ["Tools", "Python, LightGBM, pandas, scikit-learn, MLflow, FastAPI, Kaggle"]],
                [0.3, 0.7])

    s += [p("Part E. Words in simple language", H1)]
    s += table([["Word", "Simple meaning"],
                ["Concept drift", "The data or the fraud patterns change over time, so an old "
                                  "model gets worse."],
                ["Label delay", "We learn whether a transaction was fraud only weeks later."],
                ["MLOps", "Running machine learning models in real use: deploy, watch, update, "
                          "roll back."],
                ["Retraining", "Training a fresh model on newer data."],
                ["Scheduled retraining", "Retrain at fixed times (every 14 days) no matter what."],
                ["Drift-triggered retraining", "Retrain only when an alarm says the data or "
                                               "performance changed."],
                ["Training window", "How much past data the new model learns from."],
                ["Champion / challenger", "The live model / the newly trained model that wants "
                                          "to replace it."],
                ["Promotion gate", "A check that the challenger is not worse before it goes "
                                   "live."],
                ["MLflow", "A tool that stores every model version and its scores."],
                ["FastAPI", "A tool for building the web service that scores transactions."],
                ["Replay", "Feeding old data week by week as if it were happening now."],
                ["Fault injection", "Breaking the data on purpose to test safety."]], [0.3, 0.7])

    s += [PageBreak(), p("Part F. Questions you may get (short answers)", H1)]
    qa = [
        ("Why not use accuracy?", "Fraud is about 1%, so 'never fraud' is 99% accurate but "
         "useless. PR-AUC focuses on catching frauds at the top of the alert list."),
        ("Why is BAF's PR-AUC so low (0.16)?", "BAF fraud is very hard to separate and only 1.1%. "
         "Random guessing gets 0.011, so 0.16 is about 15 times better."),
        ("Did you use a pre-trained model?", "No. LightGBM is trained from scratch on each "
         "dataset, hundreds of times."),
        ("Why LightGBM and not deep learning?", "On table data boosted trees are usually the "
         "strongest, and they are fast enough to retrain many times. Other models are future work."),
        ("Why did you not shuffle the data?", "Shuffling mixes the future into training and "
         "hides drift. Real banks always learn from the past to predict the future."),
        ("Why does the schedule beat the drift trigger?", "The trigger needs proof that "
         "performance dropped, and that proof comes from labels that are weeks old. The schedule "
         "keeps the model fresh without waiting."),
        ("Is the schedule better only because it retrains more?", "No. With the same number of "
         "retrains, the schedule still matched or beat the trigger under delayed labels."),
        ("Why 60 days on IEEE-CIS but all history on Sparkov?", "IEEE-CIS changes fast, so old "
         "data misleads. Sparkov is stable and fraud is rare, so 60 days have too few frauds."),
        ("What happens if the gate is fooled?", "If labels are corrupted at the source, both "
         "models look the same. The safety net and data checks handle that case."),
        ("What is new in your work?", "A realistic test with label delay, a fair equal-budget "
         "comparison, and a framework with delayed-label monitoring, automatic window "
         "selection and a gate that re-scores on production features."),
        ("Is it real time?", "We replay historical data week by week, but the scoring API is "
         "real: in the load test it answered one transaction in about 1 ms and handled about "
         "1,300 requests per second. A live stream is the next step."),
        ("What did the load test find?", "At first only 14 requests per second, because "
         "turning the JSON into a pandas table took 57 ms. We now convert the input straight "
         "into numbers: same scores, about 100 times faster."),
        ("What is p95 latency?", "The time within which 95 of every 100 requests finish. It "
         "matters more than the average, because the slowest requests keep customers waiting."),
        ("Was there any data leakage? Did you check?", "We kept strict time order and used "
         "only labels the bank would have, with unit tests. One known leak is in IEEE-CIS's "
         "own labelling: fraud is spread to the same client's later transactions. We rebuilt "
         "a client ID and checked: without those clients the retraining gain halves (0.056 to "
         "0.027) but stays reliable, and the ranking of the strategies does not change."),
    ]
    s += table([["Question", "Answer"]] + [[f"<b>{q}</b>", a] for q, a in qa], [0.34, 0.66])
    return s


def cover():
    t = ParagraphStyle("t", fontName="Sans-Bold", fontSize=20, leading=26, textColor=NAVY,
                       alignment=TA_CENTER, spaceAfter=10)
    c = ParagraphStyle("c", parent=BODY, alignment=TA_CENTER, fontSize=11, leading=16)
    sup, cos = C.META["supervisor"], C.META["cosupervisor"]
    s = [Spacer(1, 1.2 * cm), p("P2 Poster Presentation: Speech and Guide", t),
         p(C.TITLE, ParagraphStyle("st", parent=c, fontName="Sans-It", fontSize=12, leading=17)),
         Spacer(1, 8),
         p("<br/>".join(f"{n} ({i})" for n, i in C.AUTHORS), c), Spacer(1, 6),
         p(f"Supervisor: {sup[0]}, {sup[1]}<br/>Co-supervisor: {cos[0]}, {cos[1]}", c),
         p(f"{C.META['department']}, {C.META['university']}", c), Spacer(1, 16)]
    return s


def plan(words):
    s = [p("Part A. How to use this file", H1),
         p(f"The speech in Part B is about {words} words, which is about 6 minutes at a steady "
           "pace. Part C explains every block and every figure of the poster in simple words, so "
           "anyone on the team can answer questions about any part. Parts D to F are a "
           "cheat-sheet: numbers, simple meanings of terms, and likely questions.")]
    s += table([["Time", "Section", "Poster block to point at", "Suggested speaker"],
                ["0:00 - 0:45", "Opening and problem", "Title, Abstract, Problem Statement", "Speaker 1"],
                ["0:45 - 1:35", "Objectives and literature", "Research Objectives, Literature Review", "Speaker 1"],
                ["1:35 - 2:25", "Datasets and preprocessing", "Data set table, preprocessing diagram", "Speaker 2"],
                ["2:25 - 3:40", "Method, model, framework", "Three Methodology diagrams", "Speaker 3"],
                ["3:40 - 5:15", "Results", "Grey Result Analysis boxes", "Speaker 4"],
                ["5:15 - 6:00", "Conclusion and future work", "Conclusion & Future Work", "Speaker 5"]],
                [0.15, 0.27, 0.36, 0.22])
    s += [p("Speakers are only a suggestion; one person can give the whole speech.", SMALL),
          p("<b>Tips:</b> stand beside the poster, not in front of it. Point at each figure "
            "before you talk about it. Say the one-line takeaway of a graph first, then one "
            "number. If you forget a number, the takeaway is enough. If a question is about a "
            "limitation, admit it and say how the future work handles it.")]
    return s


def build():
    sp, words = speech()
    story = cover() + plan(words) + [PageBreak()] + sp + [PageBreak()] + explain() + extras()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Sans", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.2 * cm, "P2 poster presentation: speech and guide")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title="P2 Poster Presentation: Speech and Guide",
                            author=", ".join(n for n, _ in C.AUTHORS))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("words in speech:", words, "| written", OUT)


if __name__ == "__main__":
    build()
