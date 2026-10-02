"""The full thesis report, structured like the BRAC CSE report template (demo P2 report):

  1 Introduction                       (P1 Chapter 1, sections renamed to the template)
  2 Literature Review                  (P1 Chapter 2)
  3 Requirements, Impacts and Constraints   (left empty on purpose)
  4 Methodology and Design             4.1 Design Process, 4.2 Design (Model) Specification,
                                       4.3 Data Collection, 4.4 Dataset Overview,
                                       4.5 Project Management Plan, 4.6 Implementation
  5 Result Analysis                    5.1 Performance Evaluation, 5.2 Analyse Design
                                       Solutions, 5.3 Final Design Adjustments, 5.4 Summary
                                       of Results, 5.5 Statistical Analysis, 5.6 Discussions
  6 Conclusion                         6.1 Summary of Findings, 6.2 Contributions,
                                       6.3 Recommendations for Future Work

Reuses the text of content.py (Chapters 1-2), chapter3.py (methodology and design) and
chapter4.py (results); only the material the template asks for and the thesis did not yet
have is written here. P1 and P2 are not affected.
"""

import copy

import chapter3 as M
import chapter4 as R

# ------------------------------------------------------------------------------ helpers


def _sub_blocks(sections, key):
    """Blocks of the ("sub", title, key) subsection, without its heading."""
    for sec in sections:
        blocks = sec["blocks"]
        for i, b in enumerate(blocks):
            if b[0] == "sub" and b[2] == key:
                out = []
                for nb in blocks[i + 1:]:
                    if nb[0] == "sub":
                        break
                    out.append(nb)
                return copy.deepcopy(out)
    raise KeyError(key)


def _section_blocks(chapter, key):
    for sec in chapter["sections"]:
        if sec.get("key") == key:
            return copy.deepcopy(sec["blocks"])
    raise KeyError(key)


def _pick(blocks, *starts):
    """Paragraph blocks whose text starts with one of `starts`, in the order given."""
    out = []
    for st in starts:
        hits = [b for b in blocks if b[0] == "p" and b[1].startswith(st)]
        if not hits:
            raise KeyError(st)
        out.append(hits[0])
    return out


def _edit(block, old, new):
    assert old in block[1], old[:60]
    return (block[0], block[1].replace(old, new)) + tuple(block[2:])


def sub(title, key):
    return ("sub", title, key)


# ------------------------------------------------------------------------------ front matter
ABSTRACT = [
    "Machine-learning models for financial transaction fraud detection are usually trained once "
    "and evaluated on a random split of historical data. In production this picture breaks down: "
    "fraud patterns and customer behaviour change over time (concept drift), and the true label "
    "of a transaction becomes known only weeks later, when a chargeback is filed (label delay). "
    "MLOps platforms usually decide when to retrain with statistical drift detectors on the "
    "input data, but whether such detectors reflect the performance of a fraud model when labels "
    "arrive late had not been tested.",
    "This thesis designs and evaluates a drift-aware continuous learning MLOps framework for "
    "fraud detection. Here, drift-aware means that the framework tracks drift through the "
    "performance of the model on labels as they mature, rather than relying on label-free drift "
    "alarms. The framework monitors a LightGBM model on delayed labels, retrains it on a fixed "
    "schedule with a performance safety net, selects the training window automatically, and "
    "promotes a new model only if a promotion gate confirms that it is at least as good as the "
    "live model on unseen data. It is built with MLflow, a FastAPI scoring service and a "
    "monitoring dashboard, and is evaluated by replaying three public datasets (IEEE-CIS, "
    "Sparkov and Bank Account Fraud) as live streams under label delays of 0 to 60 days, with "
    "paired bootstrap confidence intervals, walk-forward tuning, five training seeds, an "
    "equal-budget comparison and fault injection.",
    "Drift indicators proved unreliable in both directions: on IEEE-CIS the PR-AUC of a static "
    "model fell from 0.58 to 0.39 while the drift of its scores stayed below 0.02, whereas on "
    "Sparkov and BAF strong input drift (PSI up to 4.6) caused no loss of performance. Retraining "
    "helped on every dataset when its window suited the data, but its benefit on IEEE-CIS fell "
    "from +0.078 to +0.010 PR-AUC as the label delay grew to 60 days. Tuned fairly, "
    "drift-triggered retraining was reliably worse than scheduled retraining in seven of nine "
    "comparisons and never better; with delayed labels the schedule matched or beat it also at "
    "an equal number of retrains. Automatic window selection matched the best fixed window on two datasets and came "
    "within 0.005 PR-AUC on the third, and the promotion gate blocked every model trained on "
    "corrupted data that it could observe.",
]

KEYWORDS = ("Financial fraud detection, concept drift, label delay, continuous learning, MLOps, "
            "scheduled retraining, promotion gate, LightGBM, PR-AUC")

# ------------------------------------------------------------------------------ figures, tables
FIGURES = {
    "method3": ("figures/methodology_3ds.png",
                "Methodology overview: the experiments on three datasets, the framework and its "
                "evaluation"),
    "preproc": ("figures/preprocessing.png",
                "Preprocessing pipeline from the raw files of each dataset to the common "
                "time-ordered table"),
    "plan": ("figures/project_plan.png", "Project management plan over the three thesis stages"),
    "dashtop": ("figures/dashboard_top.png",
                "Front end: the monitoring dashboard after a replay of IEEE-CIS (summary figures "
                "and the performance of the served model)"),
    "dashlog": ("figures/dashboard_log.png",
                "Front end: decision log and model versions on the monitoring dashboard"),
    "budget": ("figures/results_equal_budget.png",
               "Equal-budget comparison on IEEE-CIS (top) and BAF (bottom): the best mean weekly "
               "PR-AUC each family reaches with at most a given number of retrains (dots: "
               "individual settings)"),
}

TABLES = {
    "sources": {
        "caption": "Sources of the datasets",
        "widths": [0.16, 0.34, 0.24, 0.26],
        "header": ["Dataset", "Source", "Files used", "Records in the source"],
        "rows": [
            ["IEEE-CIS", "Kaggle competition IEEE-CIS Fraud Detection (IEEE Computational "
             "Intelligence Society and Vesta Corporation)", "train_transaction.csv, "
             "train_identity.csv", "590,540 transactions"],
            ["Sparkov", "Kaggle dataset Credit Card Transactions Fraud Detection (generated "
             "with the Sparkov simulator)", "fraudTrain.csv, fraudTest.csv",
             "1,852,394 transactions"],
            ["BAF", "Kaggle dataset Bank Account Fraud Dataset Suite (NeurIPS 2022)",
             "Base.csv (of six variants)", "1,000,000 applications"],
        ],
    },
    "preprocessed": {
        "caption": "Summary of the preprocessed data",
        "widths": [0.34, 0.22, 0.22, 0.22],
        "header": ["", "IEEE-CIS", "Sparkov", "BAF"],
        "rows": [
            ["Records after preprocessing", "590,540", "1,801,969", "1,000,000"],
            ["Frauds (rate)", "20,663 (3.50%)", "9,634 (0.53%)", "11,029 (1.10%)"],
            ["Legitimate per fraud", "27.6", "186.1", "89.7"],
            ["Features (of which categorical)", "431 (31)", "16 (5)", "29 (5)"],
            ["History: training the first model", "354,324 (60%)", "900,984 (50%)",
             "500,000 (50%)"],
            ["Stream: replayed week by week", "236,216 (12 weeks)", "900,985 (52 weeks)",
             "500,000 (19 weeks)"],
        ],
    },
    "modules": {
        "caption": "Modules of the framework (Python package fraud_mlops)",
        "widths": [0.22, 0.78],
        "header": ["Module", "Responsibility"],
        "rows": [
            ["config", "Typed configuration loaded from one TOML file per dataset"],
            ["data", "Loading, caching and time-window access to the transaction table"],
            ["training", "LightGBM training on a window with a time-ordered validation split"],
            ["model", "Scoring of raw input, including missing columns and text categories"],
            ["monitoring", "Live PR-AUC on matured labels; PSI of scores and features"],
            ["policy", "Retraining policy, safety net, candidate windows and promotion gate"],
            ["registry", "MLflow runs and model registry with champion and previous aliases"],
            ["pipeline", "The replay loop: monitor, decide, retrain, gate and serve"],
            ["faults", "Fault injection for testing the safeguards"],
            ["serving", "FastAPI scoring service for the champion model"],
            ["dashboard", "Self-contained HTML monitoring dashboard"],
            ["cli", "Command line: replay, champion, dashboard and serve"],
        ],
    },
    "budget": {
        "caption": "Equal-budget comparison: each trigger setting paired with the schedule that "
                   "uses the same window and no more retrains (95% intervals; retrains: trigger "
                   "/ schedule)",
        "widths": [0.12, 0.08, 0.15, 0.08, 0.17, 0.15, 0.11, 0.14],
        "header": ["Dataset", "Delay", "Window", "Pairs", "Schedule better (reliably)",
                   "Trigger reliably better", "Mean diff.", "Retrains"],
        "rows": [
            ["IEEE-CIS", "0 d", "last 60 days", "27", "18 (9)", "0", "+0.016", "1.7 / 1.7"],
            ["", "30 d", "last 60 days", "30", "23 (7)", "0", "+0.011", "2.2 / 2.2"],
            ["", "0 d", "all history", "28", "15 (14)", "0", "+0.005", "1.7 / 1.6"],
            ["", "30 d", "all history", "30", "12 (0)", "0", "0.000", "2.3 / 2.2"],
            ["BAF", "0 d", "last 60 days", "9", "9 (2)", "0", "+0.003", "2.2 / 2.2"],
            ["", "30 d", "last 60 days", "0", "-", "-", "-", "trigger never fired"],
            ["", "0 d", "all history", "10", "2 (1)", "3", "-0.002", "1.9 / 1.9"],
            ["", "30 d", "all history", "2", "2 (1)", "0", "+0.004", "2.5 / 2.5"],
        ],
    },
}

# ------------------------------------------------------------------------------ chapter 1


def chapter1(ch1):
    ch = copy.deepcopy(ch1)
    titles = {"Rationale of the Study or Motivation": "Motivation", "Objectives": "Objective"}
    for sec in ch["sections"]:
        sec["title"] = titles.get(sec["title"], sec["title"])
        blocks = sec["blocks"]
        for i, b in enumerate(blocks):
            if b[0] == "p" and b[1].startswith("Preliminary experiments on IEEE-CIS have been"):
                blocks[i] = ("p",
                             "The experiments were run on IEEE-CIS and repeated on Sparkov and "
                             "on the Bank Account Fraud (BAF) dataset. They show that drift "
                             "indicators can miss real degradation under delayed labels, that a "
                             "fixed retraining schedule is more reliable than drift-triggered "
                             "retraining when labels are delayed, also at an equal retraining "
                             "budget, and that the "
                             "framework reproduces the offline results when replayed. The "
                             "details are given in Chapters 4 and 5.")
            if b[0] == "list":
                items = [it.replace("To prepare two public fraud datasets, IEEE-CIS and Sparkov,",
                                    "To prepare three public fraud datasets, IEEE-CIS, Sparkov "
                                    "and BAF,")
                         .replace("the second dataset.", "the other two datasets.")
                         for it in b[1]]
                blocks[i] = ("list", items) + tuple(b[2:])
    bg = ch["sections"][0]["blocks"]
    bg.append(("p", "In this thesis, a <i>drift-aware</i> system is one that is aware of drift "
                    "through its effect on the model: it tracks the performance of the deployed "
                    "model on labels as they mature, and treats label-free drift measures as "
                    "diagnostic information rather than as a reason to retrain. As the results "
                    "will show, this distinction matters, because drift indicators alone can "
                    "both miss real degradation and raise alarms about drift that does no "
                    "harm."))
    return ch


# ------------------------------------------------------------------------------ chapter 4
def chapter4():
    S = M.CH3_SECTIONS
    data = _sub_blocks(S, "data")
    ieee, sparkov, baf, order = _pick(data, "<b>IEEE-CIS Fraud Detection</b>", "<b>Sparkov</b>",
                                       "<b>Bank Account Fraud (BAF)</b>", "All three datasets")
    design = _sub_blocks(S, "design")
    design = [b for b in design if not (b[0] == "p" and b[1].startswith("The rest of this section"))]
    design.append(("p", "The rest of this section describes the simulated deployment under label "
                        "delay (Section [@sec:deploy]), the retraining strategies (Section "
                        "[@sec:strategies]), the drift and performance measures (Section "
                        "[@sec:measures]), the statistical validation (Section [@sec:validation]) "
                        "and the evaluation of the framework, including fault injection (Section "
                        "[@sec:faults]). Section [@sec:spec] specifies the model and the "
                        "framework, Sections [@sec:collection] and [@sec:data] describe the data, "
                        "Section [@sec:plan] the project plan, and Section [@sec:impl] the "
                        "implementation."))
    serving = _sub_blocks(S, "serving")
    impl = _sub_blocks(S, "impl")

    design_process = (
        [("p", "The study follows the design process shown in Figure [@fig:method3]: experiments "
               "that measure how a deployed fraud model behaves under drift and label delay, a "
               "framework designed around their results, and an evaluation of that framework on "
               "three datasets."),
         ("fig", "method3"),
         sub("Research Design", "design")] + design
        + [sub("Simulated Deployment under Label Delay", "deploy")] + _sub_blocks(S, "deploy")
        + [sub("Retraining Strategies", "strategies")] + _sub_blocks(S, "strategies")
        + [sub("Drift and Performance Measures", "measures")] + _sub_blocks(S, "measures")
        + [sub("Statistical Validation", "validation")] + _sub_blocks(S, "validation")
        + [sub("Framework Evaluation and Fault Injection", "faults")] + _sub_blocks(S, "faults"))

    specification = (
        [("p", "This section specifies the fraud detection model used throughout the study and "
               "the components of the drift-aware continuous learning framework.")]
        + [sub("Fraud Detection Model", "model")] + _sub_blocks(S, "model")
        + [sub("Framework Architecture", "arch")] + _sub_blocks(S, "arch")
        + [sub("Monitoring on Delayed Labels", "monitor")] + _sub_blocks(S, "monitor")
        + [sub("Retraining Policy and Safety Net", "safety")] + _sub_blocks(S, "safety")
        + [sub("Training and Automatic Window Selection", "window")] + _sub_blocks(S, "window")
        + [sub("Promotion Gate", "gate")] + _sub_blocks(S, "gate")
        + [sub("Framework Configuration", "config")] + _sub_blocks(S, "config"))

    collection = [
        ("p", "The study uses three public fraud datasets, summarised in Table [@tab:sources]. "
              "Public data with reliable timestamps is essential for the research questions, "
              "because drift and label delay can only be studied when transactions can be "
              "replayed in the order in which they happened. Commonly used fraud datasets that "
              "cover only a few days [@dang2021,@mienye2023] were therefore not suitable."),
        ("table", "sources"),
        ("p", "All three datasets were downloaded from Kaggle. IEEE-CIS is distributed as a "
              "competition dataset, which requires accepting the competition rules; its test "
              "files have no labels and were not used. Sparkov is distributed as two files that "
              "split one continuous period, and both were combined. BAF is distributed as a base "
              "dataset and five variants with controlled biases; the base dataset was used. The "
              "raw data is not redistributed with the code: the repository contains the scripts "
              "that prepare it, so that every result can be regenerated from the original "
              "sources."),
        ("p", "Two practical choices made the data collection reproducible. First, every "
              "dataset is converted by a preparation script into the same table layout, so all "
              "experiments run unchanged on any of them. Second, the long experiments on the two "
              "larger replication datasets were run as Kaggle notebooks, next to the data, with "
              "the code taken directly from the public repository."),
    ]

    overview = (
        [("p", "This section describes each dataset, the target variable, the cleaning and "
               "feature engineering applied to it, and the resulting preprocessed data.")]
        + [sub("IEEE-CIS Fraud Detection", "ieeecis"), _edit(ieee, "<b>IEEE-CIS Fraud Detection</b> is", "IEEE-CIS is")]
        + [sub("Sparkov Credit Card Transactions", "sparkovds"), _edit(sparkov, "<b>Sparkov</b> is", "Sparkov is")]
        + [sub("Bank Account Fraud (BAF)", "bafds"), _edit(baf, "<b>Bank Account Fraud (BAF)</b> is", "BAF is")]
        + [sub("Target Variable and Class Imbalance", "target"),
           ("p", "The target is a binary label that marks a transaction, or an application in "
                 "the case of BAF, as fraudulent (1) or legitimate (0). It is called isFraud in "
                 "IEEE-CIS, is_fraud in Sparkov and fraud_bool in BAF, and is renamed isFraud in "
                 "all three. Fraud is rare in every dataset: there are 27.6 legitimate "
                 "transactions per fraud in IEEE-CIS, 186.1 in Sparkov and 89.7 in BAF. For this "
                 "reason accuracy is not used as a measure (a model that never predicts fraud "
                 "would be 96.5% to 99.5% accurate), and PR-AUC is the main measure (Section "
                 "[@sec:measures]). The imbalance is handled by weighting the fraud class in "
                 "training; no oversampling such as SMOTE is used, because resampling can make "
                 "results look better than they are [@dang2021].")]
        + [sub("Data Cleaning and Feature Engineering", "cleaning"),
           ("p", "The preprocessing pipeline is shown in Figure [@fig:preproc]. In IEEE-CIS, the "
                 "transaction and identity tables are joined on the transaction identifier, "
                 "numeric columns are stored in the smallest suitable type to reduce memory, and "
                 "the 31 text columns are converted to categorical features. Missing values are "
                 "kept as missing, because LightGBM handles them natively and because "
                 "missingness itself can carry information: 214 of the 434 columns are missing "
                 "for more than half of the transactions."),
           ("p", "In Sparkov, names, street addresses, card numbers and transaction numbers are "
                 "removed, and the data is cut at 21 December 2020, after which the simulator "
                 "produces almost no fraud; this keeps 1,801,969 of 1,852,394 transactions. "
                 "Time-of-day, weekday, age and the distance between customer and merchant are "
                 "derived, and five per-card behaviour features are computed only from each "
                 "card's earlier transactions, so that no information leaks from the future. In "
                 "BAF, values documented as missing (negative values in six columns) are "
                 "converted to missing, a constant column is dropped, and the month is not used "
                 "as a feature."),
           ("fig", "preproc")]
        + [sub("Time Ordering and History-Stream Split", "split"), order, ("fig", "protocol")]
        + [sub("Summary of Preprocessed Data", "summarydata"),
           ("p", "Table [@tab:preprocessed] summarises the data after preprocessing, and Table "
                 "[@tab:datasets] compares the three datasets. Every prepared dataset is a single "
                 "table with a transaction identifier, a time column in seconds, the isFraud "
                 "label and the model features, sorted by time."),
           ("table", "preprocessed"), ("table", "datasets")])

    plan = [
        ("p", "The project was organised in three stages that match the thesis deliverables, as "
              "shown in Figure [@fig:plan]. The first stage covered the literature review, the "
              "research questions and the preparation of IEEE-CIS. The second stage covered the "
              "experiments, their statistical validation and the design of the framework. The "
              "third stage covered the framework's evaluation, the replication on two further "
              "datasets, automatic window selection and the changes requested in supervisor "
              "feedback."),
        ("fig", "plan"),
        ("p", "Work proceeded in short iterations: each experiment was planned, run, reviewed "
              "and, when its result raised a new question, followed by a further experiment. "
              "Several design decisions came out of this cycle. The walk-forward tuning was "
              "added when fixed tuning periods proved too short to rank the drift triggers; the "
              "promotion gate was changed to re-score models on production features after a "
              "fault test exposed a weakness; and automatic window selection was added when the "
              "replication showed that the best window depends on the dataset. Supervisor "
              "feedback was incorporated in the same way, for example the equal-budget "
              "comparison in Section [@sec:budget]."),
        ("p", "All code and configuration were kept under version control in a Git repository "
              "hosted on GitHub, with an automated test suite run before each change. Because "
              "single experiments ran for several hours and power cuts were frequent, the long "
              "experiments save their results after every completed step and resume where they "
              "stopped; the longest runs were moved to Kaggle notebooks."),
    ]

    implementation = (
        [("p", "The selected design was implemented as a Python package with a back end that "
               "runs the continuous learning loop, a front end for monitoring, and a model "
               "registry that connects them. Table [@tab:modules] lists the modules.")]
        + [sub("Back End", "backend")]
        + [("table", "modules")]
        + impl[:1]
        + [("p", "The back end is driven from the command line. <i>replay</i> runs the loop "
                 "over a dataset as if it were live, <i>champion</i> reports the model in "
                 "service, <i>dashboard</i> builds the monitoring dashboard and <i>serve</i> "
                 "starts the scoring service. Every training run is logged to MLflow with its "
                 "parameters, metrics and training window, and every model is registered as a "
                 "new version.")]
        + [sub("Front End", "frontend"),
           ("p", "The front end is a monitoring dashboard, generated as a single HTML page from "
                 "the MLflow logs of a replay, that can be opened in any browser without a "
                 "server. It shows the summary figures of the run, the weekly performance of the "
                 "served model together with what the loop could measure on delayed labels and "
                 "the safety-net line (Figure [@fig:dashtop]), the drift measures, every "
                 "retraining decision with its reason, and the history of model versions with "
                 "the champion and the version kept for rollback (Figure [@fig:dashlog])."),
           ("fig", "dashtop"), ("fig", "dashlog")]
        + [sub("Frontend-Backend Integration", "integration")]
        + serving
        + [("p", "The MLflow model registry is the contract between the parts: the back end "
                 "writes model versions and moves the <i>champion</i> alias when the promotion "
                 "gate accepts a challenger, the scoring service loads whatever carries that "
                 "alias and reloads it on request, and the dashboard reads the same registry "
                 "and run logs. No component needs to know how the others work, so the "
                 "scoring service always serves the model the gate last approved.")])

    return {"title": "Methodology and Design", "sections": [
        {"title": "Design Process", "key": "overview", "blocks": design_process},
        {"title": "Design (Model) Specification", "key": "spec", "blocks": specification},
        {"title": "Data Collection", "key": "collection", "blocks": collection},
        {"title": "Dataset Overview", "key": "data", "blocks": overview},
        {"title": "Project Management Plan", "key": "plan", "blocks": plan},
        {"title": "Implementation of Selected Design", "key": "impl", "blocks": implementation},
    ]}


# ------------------------------------------------------------------------------ chapter 5
def chapter5():
    C = R.CH4
    rq1, rq2, rq3, rq4 = (_section_blocks(C, k) for k in ("rq1", "rq2", "rq3", "rq4"))
    summary = [_edit(b, "The next chapter discusses", "Section [@sec:discussion] discusses")
               if b[0] == "p" and "The next chapter discusses" in b[1] else b
               for b in _section_blocks(C, "summary")]
    seeds_start = next(i for i, b in enumerate(rq3) if b == ("table", "seeds"))
    rq3_main, seeds = rq3[:seeds_start], rq3[seeds_start:]
    answer3 = [b for b in seeds if b[0] == "p" and b[1].startswith("<b>Answer to RQ3.")]
    seeds = [b for b in seeds if b not in answer3]
    i_window = next(i for i, b in enumerate(rq4) if b[0] == "p" and b[1].startswith("<b>Automatic window selection."))
    i_faults = next(i for i, b in enumerate(rq4) if b[0] == "p" and b[1].startswith("<b>Safeguards under fault injection."))
    replay, window, faults = rq4[:i_window], rq4[i_window:i_faults], rq4[i_faults:]
    answer4 = [b for b in faults if b[0] == "p" and b[1].startswith("<b>Answer to RQ4.")]
    faults = [b for b in faults if b not in answer4]

    evaluation = (
        [sub("Testing Method", "testing"),
         ("p", "The framework and the retraining strategies are tested by replaying each dataset "
               "in strict time order, as described in Section [@sec:overview]. A model is "
               "trained on the history, deployed at the start of the stream and scores each "
               "following week before the labels of that week are known; the labels become "
               "usable only after the label delay of 0, 15, 30 or 60 days. Every strategy starts "
               "from the same initial model and sees exactly the same data, so differences "
               "between strategies come only from their retraining decisions. The framework "
               "itself is tested by a full replay with its registry, monitor and promotion gate "
               "(Section [@sec:replay5]) and by injecting faults into its retraining jobs "
               "(Section [@sec:faults5])."),
         sub("Evaluation Metrics", "metrics"),
         ("p", "The main metric is the mean weekly PR-AUC (Equation [@eq:weekly]), computed per "
               "week so that scores of different model versions are never pooled. ROC-AUC, "
               "precision, recall and F1 at each model's validation threshold, the lowest "
               "weekly PR-AUC and the number of retrains are reported as secondary measures. "
               "Drift is measured with the PSI (Equation [@eq:psi]). Differences between "
               "strategies are reported with 95% paired day-block bootstrap intervals, and a "
               "difference is called reliable when its interval excludes zero. Because the "
               "datasets differ in difficulty, a model without skill would score 0.035 on "
               "IEEE-CIS, 0.005 on Sparkov and 0.011 on BAF, so PR-AUC values are compared "
               "within a dataset only."),
         sub("Model Performance and Drift after Deployment", "rq1")] + rq1
        + [sub("Retraining under Label Delay", "rq2")] + rq2)

    budget = [
        ("p", "Scheduled retraining retrained more often than the drift triggers in the "
              "comparisons above (for example 12 against 2 to 5 retrains in walk-forward "
              "tuning on IEEE-CIS), so part of its advantage could come simply from retraining "
              "more often. To separate the two, both families were run over many settings on "
              "IEEE-CIS and BAF, schedules every 7 days or more and 64 trigger settings, and "
              "every trigger setting "
              "was paired with the schedule that uses the same training window and the largest "
              "number of retrains that does not exceed the trigger's. The schedule therefore "
              "never has more retrains than the trigger it is compared with, which favours the "
              "trigger. Table [@tab:budget] summarises the pairs and Figure [@fig:budget] shows "
              "the best PR-AUC each family reaches within a given budget."),
        ("table", "budget"), ("fig", "budget"),
        ("p", "With the 60-day window, the best window on IEEE-CIS, the schedule was better in "
              "18 of 27 pairs with immediate labels and 23 of 30 pairs at a 30-day delay, "
              "reliably so in 9 and 7 pairs, and the trigger was never reliably better; on "
              "average the schedule was 0.016 and 0.011 PR-AUC ahead at the same budget. With "
              "two retrains, for example, the best schedule reached 0.568 and the best trigger "
              "0.546. With all history as training data the two were close: the schedule was "
              "ahead by 0.005 with immediate labels and level at a 30-day delay. The label-free "
              "score-PSI trigger fired only once across all its settings. Finally, the triggers "
              "rarely used more than two or three retrains, while the schedule kept improving "
              "as it retrained more often (0.599 with eleven retrains)."),
        ("p", "On BAF the triggers rarely fired at all, because performance hardly changed: "
              "most settings never retrained, and at a 30-day delay no setting with the 60-day "
              "window fired. Where pairs exist, the picture depends on the label delay. With "
              "immediate labels and all history as training data, the trigger was the more "
              "economical policy: it was reliably better in 3 of 10 pairs, and its best setting "
              "reached 0.173 with two retrains, which the schedule matched only with nine. With "
              "the 60-day window the schedule was better in all 9 pairs. At a realistic 30-day "
              "delay the schedule was ahead again: with two retrains it reached 0.168 against "
              "0.161 for the best trigger, and 0.170 against 0.164 at best."),
        ("p", "The advantage of scheduled retraining is therefore not only a matter of retraining "
              "more often. When labels were delayed by 30 days, the schedule matched or beat the "
              "drift trigger at an equal budget on both datasets, and it was never reliably worse "
              "on IEEE-CIS. A trigger can be economical when labels arrive immediately, as on BAF "
              "with all history, which is consistent with earlier work that assumed immediate "
              "labels [@yelleti2025]; under label delay it has to wait for evidence of "
              "degradation and loses that advantage. Part of the schedule's overall advantage "
              "also comes from the larger number of retrains, which a delayed-label trigger "
              "cannot match. The comparison on Sparkov is being run in the same way."),
    ]

    analyse = (
        [sub("Drift-Triggered versus Scheduled Retraining", "rq3")] + rq3_main + answer3
        + [sub("Equal-Budget Comparison", "budget")] + budget
        + [sub("Framework Replay", "replay5")] + replay
        + [sub("Safeguards under Fault Injection", "faults5")] + faults + answer4)

    adjustments = [
        ("p", "Three changes were made to the design during the evaluation, each prompted by a "
              "result."),
        sub("Promotion Gate on Production Features", "adj_gate"),
        ("p", "The first version of the promotion gate compared the challenger and the champion "
              "using scores stored when each model was trained. A fault test with a feature "
              "unit bug showed the weakness of this: a model trained on wrongly scaled features "
              "looks good on equally wrong validation data. The gate was changed to re-score "
              "both models through the production feature path (Section [@sec:gate]), after "
              "which it rejected the faulty models."),
        sub("Safety-Net Reference from Live Performance", "adj_safety"),
        ("p", "Originally the safety net compared live PR-AUC with the model's own validation "
              "PR-AUC. When the label store itself was corrupted, a faulty model reported a "
              "poor validation score and was then judged against that low bar, so the fault "
              "went undetected and cost 0.149 PR-AUC. The reference was changed to the better "
              "of the validation score and the median of the last four live measurements, and "
              "a no-skill check was added (Section [@sec:safety]); the same fault was then "
              "caught after seven days and cost 0.080."),
        sub("Automatic Training-Window Selection", "adj_window"),
    ] + window

    statistics = [
        ("p", "Three layers of statistical analysis support the results. First, every "
              "comparison between strategies is paired: all strategies are evaluated on the "
              "same weeks and the same bootstrap resamples, which removes most of the noise "
              "they share. Second, the paired day-block bootstrap resamples whole days within "
              "each week, which respects the correlation between transactions of the same day, "
              "and yields the 95% intervals reported throughout. Third, training randomness is "
              "covered by repeating the key comparisons with five LightGBM seeds."),
    ] + seeds + [
        ("p", "Because many comparisons are reported, a single interval that excludes zero "
              "should not be over-interpreted. The conclusions of this thesis rest instead on "
              "patterns that hold across label delays, datasets, seeds and analyses: for "
              "example, the drift trigger was reliably worse than the schedule in seven of nine "
              "walk-forward comparisons and never reliably better in any of them, and the "
              "equal-budget analysis pointed the same way whenever labels were delayed."),
    ]

    discussion = [
        ("p", "<b>Drift indicators and delayed labels.</b> The central finding is that "
              "label-free drift indicators did not reflect what mattered. They stayed quiet "
              "while the IEEE-CIS model lost a third of its PR-AUC, and they signalled strong "
              "drift on Sparkov and BAF while the models were unaffected. Shakil et al. noted "
              "that statistical drift does not always mean lower performance [@shakil2025]; on "
              "natural drift in fraud data we observed this in both directions. For a fraud "
              "system this means that drift alarms should inform investigation, not trigger "
              "retraining, and that the performance on matured labels is the signal to watch, "
              "even though it arrives late. This is the sense in which the framework is "
              "drift-aware."),
        ("p", "<b>Schedules and triggers.</b> Earlier work reported that drift-triggered "
              "retraining matches or beats a schedule with fewer retrains "
              "[@wong2025,@yelleti2025]. Those results were obtained with injected drift or "
              "immediate labels. Under realistic label delay a performance trigger can only "
              "react after a drop has been confirmed on labels that are weeks old, and a "
              "label-free trigger reacts to the wrong signal. A schedule needs no evidence and "
              "keeps the model fresh, and the equal-budget analysis shows that under label delay "
              "its advantage is not merely a matter of retraining more often. With immediate "
              "labels, a trigger could be the more economical policy, as on BAF, which explains "
              "why earlier studies that assumed immediate labels favoured triggers. The result is consistent with "
              "Dal Pozzolo et al., who stressed that verification latency shapes how a fraud "
              "model can be updated [@dalpozzolo2018], and with Amekoe et al., who found that "
              "batch models retrained on delayed labels remain competitive [@amekoe2024]."),
        ("p", "<b>The training window.</b> The choice of training window changed results by up "
              "to 0.26 PR-AUC, far more than the choice of trigger. Recent data was best on "
              "IEEE-CIS, which decays, and all history was best on Sparkov and BAF, which are "
              "stable and have few frauds. Because no fixed window suits every deployment, "
              "selecting the window from the data, as the framework now does, is a practical "
              "way to avoid a costly wrong choice."),
        ("p", "<b>Safe automation.</b> Automated retraining will eventually train on bad data. "
              "The fault tests show that a promotion gate that re-scores models on production "
              "data stops faults it can observe, and that a safety net based on live "
              "performance limits the damage of faults it cannot. Closed-loop MLOps frameworks "
              "such as HAMF [@reda2025] automate the loop; this thesis adds evidence on how such "
              "safeguards behave when labels arrive late."),
        ("p", "<b>Limitations.</b> The deployment is replayed from historical data, so live "
              "latency and the behaviour of a real chargeback feed are not observed, and every "
              "label arrives after the same delay, without faster investigator feedback. One "
              "model family is used throughout. IEEE-CIS covers only six months, Sparkov is "
              "simulated, and BAF records only the month of each application, so drift within "
              "a month cannot appear. The equal-budget comparison is complete on IEEE-CIS and "
              "BAF and under way on Sparkov, and the scoring service has not yet been load "
              "tested."),
    ]

    return {"title": "Result Analysis", "sections": [
        {"title": "Performance Evaluation", "key": "evaluation", "blocks": evaluation},
        {"title": "Analyse Design Solutions", "key": "analyse", "blocks": analyse},
        {"title": "Final Design Adjustments", "key": "adjust", "blocks": adjustments},
        {"title": "Summary of Results", "key": "summary", "blocks": summary},
        {"title": "Statistical Analysis", "key": "statistics", "blocks": statistics},
        {"title": "Discussions", "key": "discussion", "blocks": discussion},
    ]}


# ------------------------------------------------------------------------------ chapter 6
CH6 = {"title": "Conclusion", "sections": [
    {"title": "Summary of Findings", "key": "findings", "blocks": [
        ("p", "This thesis set out to design and evaluate a drift-aware continuous learning MLOps "
              "framework for financial transaction fraud detection that identifies when a "
              "deployed model is becoming outdated, decides when retraining is necessary, and "
              "deploys a better model safely. Replaying three public datasets under realistic "
              "label delay led to five main findings:"),
        ("list", [
            "Label-free drift indicators did not reflect model performance: they missed a "
            "loss of a third of the PR-AUC on IEEE-CIS and signalled strong drift on Sparkov "
            "and BAF, where performance did not fall.",
            "Retraining improved performance on every dataset when its training window suited "
            "the data, but label delay eroded its benefit: on IEEE-CIS the gain fell from "
            "+0.078 PR-AUC with immediate labels to +0.010 with a 60-day delay.",
            "Tuned fairly, drift-triggered retraining was never reliably better than a fixed "
            "schedule and was reliably worse in seven of nine comparisons; with delayed labels "
            "the schedule matched or beat it also at an equal retraining budget, while with "
            "immediate labels a trigger could retrain more economically.",
            "The training window mattered more than the trigger, and automatic window selection "
            "matched the best fixed window on two datasets and came within 0.005 PR-AUC on the "
            "third.",
            "The framework reproduced the offline results in a full replay, its promotion gate "
            "blocked every faulty model it could observe, and its safety net limited the "
            "damage of the fault it could not observe to one week.",
        ], True),
    ]},
    {"title": "Contributions to the Field", "key": "contributions", "blocks": [
        ("list", [
            "An evaluation of retraining policies for fraud detection on public data in strict "
            "time order, with explicit label delay, paired bootstrap intervals, walk-forward "
            "tuning, repeated seeds and three datasets.",
            "Evidence, on natural drift, that label-free drift indicators are an unreliable "
            "trigger for retraining fraud models, and a definition of drift-awareness based on "
            "delayed-label performance instead.",
            "An equal-budget comparison showing that, under label delay, the advantage of "
            "scheduled retraining is not only due to retraining more often.",
            "A drift-aware MLOps framework with delayed-label monitoring, a scheduled policy "
            "with a safety net, automatic window selection and a promotion gate that re-scores "
            "models on production features, tested by fault injection.",
            "Open, reproducible code with resumable experiments and Kaggle notebooks, so that "
            "every result can be regenerated from the public datasets.",
        ], True),
    ]},
    {"title": "Recommendations for Future Work", "key": "future", "blocks": [
        ("list", [
            "Complete the equal-budget comparison on Sparkov, and load test the scoring "
            "service (throughput and p95/p99 latency at increasing concurrency), as recommended "
            "in supervisor feedback.",
            "Model faster investigator feedback alongside delayed chargebacks, which could make "
            "performance-based triggers react sooner.",
            "Repeat the study with other model families, such as neural and online learners, to "
            "test whether the findings depend on gradient boosted trees.",
            "Monitor the stability of model explanations under drift [@john2025,@uddin2026] "
            "and drift in several data sources separately [@hassan2026].",
            "Deploy the framework on a live transaction stream with real label feeds, for "
            "example behind a streaming platform [@kaushik2025].",
        ], True),
    ]},
]}

CH3_EMPTY = {"title": "Requirements, Impacts and Constraints", "sections": []}


def chapters(ch1, ch2):
    return [chapter1(ch1), copy.deepcopy(ch2), CH3_EMPTY, chapter4(), chapter5(), CH6]


def chapters_empty123(ch1, ch2):
    """Variant with Chapters 1-3 left empty (title pages only) and Chapters 4-6 unchanged.
    Figures 1.1 and 1.2 are referred to from Chapter 4, so they are shown there instead."""
    ch4 = chapter4()
    for sec in ch4["sections"]:
        for fig in ("method", "loop"):
            for i, b in enumerate(sec["blocks"]):
                if b[0] == "p" and f"[@fig:{fig}]" in b[1]:
                    sec["blocks"].insert(i + 1, ("fig", fig))
                    break
    return [{"title": ch1["title"], "sections": []}, {"title": ch2["title"], "sections": []},
            CH3_EMPTY, ch4, chapter5(), CH6]
