"""Chapter 3 (Methodology): text, figures, tables and equations. Merged into content.py.

Same markup as content.py. Equations: ("eq", key) blocks, defined in EQUATIONS as
(latex, pdf_html), numbered per chapter and referenced as [@eq:key]. Every number here is
taken from the code and configuration files of the repository (configs/default.toml,
retraining_simulation.py, walkforward_tuning.py, seed_robustness.py, gate_fault_test.py).
"""

SYM = '<font name="Symbol">{}</font>'          # Greek letters in the PDF (Symbol font)
SUM, TAU, DELTA, LAMBDA, PI, KAPPA = (SYM.format(c) for c in "Στδλπκ")
GE, LE, MINUS = (SYM.format(c) for c in "≥≤−")

FIGURES = {
    "protocol": ("figures/protocol.png",
                 "Evaluation protocol: each dataset is split in time into a history and a "
                 "stream that is replayed week by week; settings are tuned walk-forward in "
                 "4-week folds"),
    "architecture": ("figures/architecture.png",
                     "Components of the proposed framework and the flow of data between them"),
}

EQUATIONS = {
    "psi": (r"\mathrm{PSI} = \sum_{i=1}^{B} (c_i - r_i)\,\ln\frac{c_i}{r_i}",
            f"PSI = {SUM}<sub><i>i</i>=1</sub><super><i>B</i></super> "
            f"(<i>c<sub>i</sub></i> - <i>r<sub>i</sub></i>) ln(<i>c<sub>i</sub></i> / "
            f"<i>r<sub>i</sub></i>)"),
    "ap": (r"\mathrm{AP} = \sum_{n} (R_n - R_{n-1})\,P_n",
           f"AP = {SUM}<sub><i>n</i></sub> (<i>R<sub>n</sub></i> - <i>R</i><sub><i>n</i>-1</sub>) "
           f"<i>P<sub>n</sub></i>"),
    "weekly": (r"\overline{\mathrm{AP}} = \frac{1}{W} \sum_{w=1}^{W} \mathrm{AP}_w",
               f"mean AP = (1/<i>W</i>) {SUM}<sub><i>w</i>=1</sub><super><i>W</i></super> "
               f"AP<sub><i>w</i></sub>"),
    "labelled": (r"\mathcal{L}(t) = \{\, x_j : t_j \le t - d \,\}",
                 "<i>L</i>(<i>t</i>) = { <i>x<sub>j</sub></i> : <i>t<sub>j</sub></i> "
                 f"{LE} <i>t</i> {MINUS} <i>d</i> }}"),
    "perftrigger": (r"\mathrm{AP}_{\mathrm{live}}(t) < (1 - \tau)\, \mathrm{AP}_{\mathrm{ref}}",
                    f"AP<sub>live</sub>(<i>t</i>) &lt; (1 - {TAU}) AP<sub>ref</sub>"),
    "objective": (r"J(c) = \overline{\mathrm{AP}}(c) - \kappa \cdot \frac{N_{\mathrm{retrain}}(c)}"
                  r"{\text{number of 4-week periods}}",
                  f"<i>J</i>(<i>c</i>) = mean AP(<i>c</i>) - {KAPPA} x "
                  f"<i>N</i><sub>retrain</sub>(<i>c</i>) / (number of 4-week periods)"),
    "safety": (r"\mathrm{AP}_{\mathrm{live}} < (1 - \tau)\,\max\big(\mathrm{AP}_{\mathrm{val}},\ "
               r"\mathrm{median}_{4}(\mathrm{AP}_{\mathrm{live}})\big)"
               r"\quad\text{or}\quad \mathrm{AP}_{\mathrm{live}} < \lambda\,\pi",
               f"AP<sub>live</sub> &lt; (1 - {TAU}) max(AP<sub>val</sub>, median<sub>4</sub>"
               f"(AP<sub>live</sub>))&nbsp;&nbsp; or &nbsp;&nbsp;AP<sub>live</sub> &lt; "
               f"{LAMBDA} {PI}"),
    "gate": (r"\mathrm{AP}(\text{challenger}) \ge \mathrm{AP}(\text{champion}) - \delta",
             f"AP(challenger) {GE} AP(champion) {MINUS} {DELTA}"),
}

TABLES = {
    "datasets": {
        "caption": "Datasets used in this study (after preparation)",
        "widths": [0.17, 0.28, 0.28, 0.27],
        "header": ["", "IEEE-CIS", "Sparkov", "BAF (Base)"],
        "rows": [
            ["Domain", "Real e-commerce card transactions (Vesta)",
             "Simulated card transactions (Sparkov generator)",
             "Bank account opening applications (privacy-preserving synthetic data)"],
            ["Records", "590,540", "1,801,969", "1,000,000"],
            ["Time span", "182 days", "720 days (to 21 Dec 2020)", "8 months"],
            ["Fraud rate", "3.50%", "0.53%", "1.10%"],
            ["Features used", "431 (transaction and identity tables, mostly anonymised)",
             "16 (amount, time, location, customer and card-history features)",
             "29 (applicant, request, device and velocity features)"],
            ["Time resolution", "Seconds from a reference point", "Exact timestamp",
             "Month only (time within the month assigned at random)"],
            ["Stream starts at", "60% of rows (day 101)", "50% of rows (about 1 Jan 2020)",
             "50% of rows (month 4)"],
            ["Role", "Main dataset", "Replication", "Second replication"],
        ],
    },
    "hyper": {
        "caption": "LightGBM settings used for every model in the study",
        "widths": [0.34, 0.2, 0.46],
        "header": ["Setting", "Value", "Purpose"],
        "rows": [
            ["Objective", "binary", "Fraud probability for each transaction"],
            ["Learning rate", "0.05", "Step size of each boosting round"],
            ["Number of leaves", "256", "Capacity of each tree"],
            ["Minimum samples per leaf", "100", "Prevents leaves fitted to a few transactions"],
            ["Row subsampling", "0.8, every round", "Randomness and regularisation"],
            ["Feature subsampling", "0.5 per tree", "Randomness and regularisation"],
            ["L2 regularisation", "1.0", "Shrinks leaf values"],
            ["Class weight of fraud", "legitimate / fraud count in the training data",
             "Compensates for class imbalance"],
            ["Boosting rounds", "up to 2,000, early stopping after 100 without improvement",
             "Stops on the validation part of the training window"],
            ["Validation part", "most recent 15% of the training window",
             "Early stopping, threshold choice and reference PR-AUC"],
        ],
    },
    "strategies": {
        "caption": "Retraining strategies compared in the simulation (default settings)",
        "widths": [0.2, 0.3, 0.25, 0.25],
        "header": ["Strategy", "When it retrains", "Training data", "Needs labels to decide?"],
        "rows": [
            ["Static", "Never", "History before the stream", "-"],
            ["Periodic expanding", "Every 14 days", "All labelled data so far", "No"],
            ["Periodic sliding", "Every 14 days", "Labelled data of the last 60 days", "No"],
            ["Drift: score PSI", "PSI of the model's scores over the last week, against its "
             "validation scores, exceeds 0.1 (at least 14 days apart)",
             "All labelled data so far", "No"],
            ["Drift: performance", "PR-AUC on newly matured labels falls more than 15% below "
             "the reference (Equation [@eq:perftrigger]; at least 14 days apart)",
             "All labelled data so far (or the last 60 days in tuned variants)",
             "Yes (delayed)"],
        ],
    },
    "grid": {
        "caption": "Settings searched by walk-forward tuning (190 configurations)",
        "widths": [0.24, 0.5, 0.26],
        "header": ["Family", "Settings searched", "Configurations"],
        "rows": [
            ["Static", "-", "1"],
            ["Schedule", "Expanding window, period 7, 14 or 28 days; sliding window, period "
             "7, 14 or 28 days and window 30 or 60 days", "9"],
            ["Drift: performance", "Tolerance 0.05, 0.1, 0.15, 0.2 or 0.3; monitoring window "
             "7, 14 or 28 days; minimum gap 7, 14 or 28 days; reference = validation PR-AUC "
             "or first live measurement; training data = all or last 60 days", "180"],
        ],
    },
    "framework": {
        "caption": "Settings of the framework (configs/default.toml)",
        "widths": [0.34, 0.16, 0.5],
        "header": ["Setting", "Value", "Meaning"],
        "rows": [
            ["Loop interval", "7 days", "How often the framework monitors, decides and serves"],
            ["Label delay d", "30 days", "Labels usable this long after the transaction"],
            ["Retraining schedule", "14 days", "Primary policy"],
            ["Training window", "60 days", "Sliding window of labelled data (0 = expanding)"],
            ["Monitoring window", "14 days", "Matured labels used for live PR-AUC"],
            ["Minimum frauds", "30", "Fewer frauds: live PR-AUC is not computed"],
            ["Safety-net tolerance " + TAU, "0.30", "Relative drop that triggers an early retrain"],
            ["History for the reference", "4", "Recent live measurements in the median"],
            ["No-skill factor " + LAMBDA, "3", "Live PR-AUC below 3 x fraud rate triggers"],
            ["Minimum gap between retrains", "7 days", "Prevents repeated early retrains"],
            ["Gate tolerance " + DELTA, "0.01", "Largest PR-AUC loss a challenger may show"],
            ["Gate minimum frauds", "30", "Fewer frauds: the challenger is promoted by default"],
            ["Drift features tracked", "10", "Most important numeric features (PSI logged)"],
        ],
    },
    "faults": {
        "caption": "Faults injected into retraining jobs to test the safeguards",
        "widths": [0.25, 0.4, 0.35],
        "header": ["Fault", "What goes wrong", "Real-world cause"],
        "rows": [
            ["Label shuffle", "Labels are permuted within the training window, so they no "
             "longer match their transactions", "A misaligned join in the training job"],
            ["Label loss", "80% of frauds in the training window are recorded as legitimate",
             "An outage of the chargeback feed"],
            ["Feature unit bug", "The five most important numeric features of the live model "
             "are multiplied by 100 in training only", "A unit change (e.g. cents and dollars) "
             "between training and serving"],
            ["Upstream label shuffle", "The label store itself is corrupted, so the promotion "
             "gate also reads wrong labels", "Corruption at the source; a known blind spot"],
        ],
    },
}

CH3 = {"title": "Methodology", "sections": [
    {"title": "Research Design", "blocks": [
        ("p", "This study is an empirical, simulation-based investigation followed by the design "
              "and evaluation of a software framework. Chapter 2 showed that the claims made for "
              "drift-triggered retraining rest mostly on artificial drift, single datasets and "
              "labels that arrive immediately [@shakil2025,@wong2025,@yelleti2025]. We therefore "
              "first measure, on public fraud data with natural drift, how a deployed model "
              "behaves and which retraining policy works best under label delay (research "
              "questions RQ1 to RQ3), and only then design the framework around the result and "
              "test it (RQ4). The study follows the workflow in Figure [@fig:method]."),
        ("p", "Three principles guide every experiment. First, <b>time is respected</b>: a model "
              "is only trained on transactions that happened, and whose labels were available, "
              "before the period it is evaluated on [@menezes2025]. Second, <b>the deployment is "
              "simulated, not imagined</b>: each dataset is replayed as a stream, week by week, "
              "and the label delay is applied explicitly, as described by Dal Pozzolo et al. "
              "[@dalpozzolo2018]. Third, <b>differences must survive uncertainty</b>: every "
              "comparison is reported with a confidence interval, the settings of competing "
              "strategies are tuned with the same procedure, and the key results are repeated "
              "with several training seeds and on a second dataset."),
        ("p", "The rest of this chapter describes the datasets and their preparation "
              "(Section 3.2), the model (Section 3.3), the simulated deployment and the "
              "retraining strategies (Sections 3.4 and 3.5), the drift and performance measures "
              "(Section 3.6), the statistical validation (Section 3.7), the framework "
              "(Section 3.8), the fault-injection tests (Section 3.9) and the implementation "
              "(Section 3.10)."),
    ]},
    {"title": "Datasets and Preparation", "blocks": [
        ("p", "Public fraud datasets with reliable timestamps are scarce. Many studies use the "
              "European credit-card dataset, which covers only two days and therefore cannot "
              "show drift [@dang2021,@mienye2023]. We use three datasets that cover months to "
              "years, summarised in Table [@tab:datasets]."),
        ("table", "datasets"),
        ("p", "<b>IEEE-CIS Fraud Detection</b> is the main dataset. It was released by the IEEE "
              "Computational Intelligence Society and Vesta Corporation and contains 590,540 "
              "real e-commerce transactions over 182 days, of which 3.5% are fraudulent. It has "
              "been used in recent studies of drift and fraud detection [@menezes2025,@uddin2026]. "
              "The transaction and identity tables are joined on the transaction identifier, "
              "giving 431 features, most of them anonymised by the provider. Only the time "
              "column (seconds from a reference point), the identifier and the label are "
              "excluded. Text columns are treated as categorical features, which LightGBM "
              "handles natively, and missing values are kept as missing rather than imputed. "
              "More than half of the columns are missing for over 50% of transactions."),
        ("p", "<b>Sparkov</b> is the first replication dataset. It contains card transactions "
              "for about 1,000 customers produced by the Sparkov transaction simulator between "
              "January 2019 and December 2020. After inspection we cut the data at 21 December "
              "2020, because the simulator produces almost no fraud after that date (17 frauds "
              "and then none, while the volume doubles), which would make the last weeks "
              "meaningless for evaluation. The remaining 1,801,969 transactions cover 720 days "
              "with a fraud rate of 0.53%. Names, street addresses, card numbers and "
              "transaction numbers are removed. We derive the transaction hour and weekday, the "
              "customer's age, and the distance between customer and merchant, and five "
              "per-card behaviour features: the time since the card's previous transaction, "
              "its number and total amount of transactions in the previous 24 hours, its "
              "number of transactions in the previous 30 days, and the amount relative to the "
              "card's running average. Each of these is computed only from the card's "
              "<i>earlier</i> transactions, so no information from the future leaks into a "
              "feature. A lifetime transaction count was deliberately not used: it only grows "
              "over time and would drift by construction."),
        ("p", "<b>Bank Account Fraud (BAF)</b> is the second replication dataset. It is a "
              "privacy-preserving synthetic dataset of one million bank account applications "
              "over eight months, generated from real data and designed to contain realistic "
              "drift between months. We use the Base variant. Values that the providers "
              "document as missing (negative values in six columns) are converted to missing "
              "values, a constant column is dropped, and the month itself is not used as a "
              "feature, so that the model cannot memorise the period. BAF records only the "
              "month of each application, so each record is given a random time within its "
              "month with a fixed seed. Drift therefore appears between months, while the weeks "
              "within a month are exchangeable. This is a limitation of the dataset, and the "
              "BAF results are interpreted accordingly."),
        ("p", "All three datasets are sorted by time and never shuffled. Each is split at a "
              "fixed fraction of its rows into a <i>history</i>, used to train the first model, "
              "and a <i>stream</i>, which is replayed as if it were arriving live (Figure "
              "[@fig:protocol]). The stream starts at 60% of the rows for IEEE-CIS (day 101), "
              "and at 50% for Sparkov and BAF, whose longer histories allow a full year or four "
              "months of training data before deployment."),
        ("fig", "protocol"),
    ]},
    {"title": "Model", "blocks": [
        ("p", "All experiments use LightGBM, a gradient boosted decision tree library, with the "
              "settings in Table [@tab:hyper]. Gradient boosted trees are strong and widely "
              "used models for tabular fraud data [@uddin2026,@amekoe2024], they handle missing "
              "values and categorical features natively, and they train fast enough to be "
              "retrained hundreds of times. Amekoe et al. also found that batch-trained boosted "
              "trees remain competitive with incremental learners when labels arrive late "
              "[@amekoe2024], which is exactly the setting of this study. The goal is not to "
              "find the best possible classifier, as many static studies do "
              "[@mienye2023,@abdelnaby2023,@sohony2018], but to keep one strong and realistic "
              "model fixed so that differences between retraining strategies can be attributed "
              "to the strategies alone. No resampling such as SMOTE is used, because it can "
              "distort the evaluation [@dang2021]; class imbalance is handled by weighting the "
              "fraud class instead."),
        ("table", "hyper"),
        ("p", "Every training run uses a time-ordered validation split: the most recent 15% of "
              "the training window is held out for early stopping, for choosing the decision "
              "threshold, and for recording the model's reference PR-AUC. The decision "
              "threshold is the one that maximises the F1-score on this validation part. It is "
              "used for the F1-score, precision and recall reported alongside PR-AUC and for "
              "the fraud flags returned by the scoring service; PR-AUC itself does not depend "
              "on a threshold."),
        ("p", "For the static baseline in RQ1, the first 50% of each dataset is used for "
              "training, the next 10% for validation, and the remaining 40% is divided into ten "
              "consecutive time bins, so that performance can be followed over the months after "
              "deployment."),
    ]},
    {"title": "Simulated Deployment under Label Delay", "blocks": [
        ("p", "The stream is replayed in steps of seven days, from the stream start to the end "
              "of the data. At each step, three things happen in order: the strategy checks "
              "which labels are available and decides whether to retrain; if it retrains, a new "
              "model is trained and immediately replaces the old one; and the current model "
              "scores the transactions of the next seven days. The scores of each week are kept, "
              "together with the version of the model that produced them."),
        ("p", "Label delay is modelled as a fixed number of days <i>d</i> between a transaction "
              "and the moment its label can be used. At time <i>t</i>, the labelled data "
              "available for monitoring and training is"),
        ("eq", "labelled"),
        ("p", "where <i>t<sub>j</sub></i> is the time of transaction <i>x<sub>j</sub></i>. The "
              "first model is trained at the stream start under the same rule, so it never sees "
              "labels it could not have had. We evaluate delays of 0, 15, 30 and 60 days. A "
              "delay of 0 corresponds to the assumption made by most retraining studies "
              "[@yelleti2025,@wong2025]; 30 to 60 days corresponds to the time a cardholder "
              "typically needs to notice a fraudulent charge and file a dispute "
              "[@dalpozzolo2018]. Investigator feedback on a few alerted transactions, which "
              "arrives faster in a real system [@dalpozzolo2018], is not modelled: every label "
              "arrives after the same delay. This makes the setting slightly pessimistic for "
              "performance-based triggers, a point we return to in the discussion."),
    ]},
    {"title": "Retraining Strategies", "blocks": [
        ("p", "Five strategies are compared, covering the options described in Section 2.1 "
              "[@pulicharla2019,@dalpozzolo2018]. They are summarised in Table "
              "[@tab:strategies]."),
        ("table", "strategies"),
        ("p", "The <b>static</b> model is the reference: it is trained once at the stream start "
              "and never updated. The two <b>periodic</b> strategies retrain on a fixed "
              "schedule and differ only in their training data: the expanding window uses all "
              "labelled data so far, while the sliding window keeps only the most recent 60 "
              "days and forgets older patterns."),
        ("p", "The <b>score-PSI</b> strategy is a typical label-free drift trigger "
              "[@shakil2025,@wong2025]. It compares the distribution of the scores the model "
              "produced over the last week with the scores it produced on its own validation "
              "data, using the PSI defined in Section 3.6, and retrains when the PSI exceeds "
              "0.1, the usual boundary between a stable and a shifted population. It can react "
              "immediately because it needs no labels."),
        ("p", "The <b>performance</b> strategy is an error-based trigger in the spirit of DDM "
              "and ADWIN [@yelleti2025,@aldaoud2025], adapted to delayed labels. At each step "
              "it computes the model's PR-AUC on transactions whose labels matured during the "
              "last 14 days and that the model was not trained on, provided they contain at "
              "least 30 frauds. It retrains when"),
        ("eq", "perftrigger"),
        ("p", "where AP<sub>live</sub>(<i>t</i>) is that live PR-AUC, " + TAU + " is the "
              "tolerance (0.15 by default), and AP<sub>ref</sub> is either the model's "
              "validation PR-AUC or its first live measurement. Both drift strategies wait at "
              "least 14 days between retrains, so that one drift episode does not trigger a "
              "series of retrains."),
        ("p", "Each strategy is run at each label delay, with exactly the same data, model "
              "settings and random seed, so that the only difference between runs is the "
              "retraining decision and the training window. Models trained on identical "
              "windows are cached and reused across strategies, which makes the large number "
              "of runs feasible and guarantees that identical decisions give identical models."),
    ]},
    {"title": "Drift and Performance Measures", "blocks": [
        ("p", "<b>Population Stability Index.</b> Drift in a feature or in the model's scores "
              "is measured with the PSI, which compares a current distribution with a "
              "reference distribution [@wong2025]:"),
        ("eq", "psi"),
        ("p", "where the reference values are divided into <i>B</i> = 10 bins at their deciles, "
              "<i>r<sub>i</sub></i> and <i>c<sub>i</sub></i> are the shares of reference and "
              "current values in bin <i>i</i>, missing values form a bin of their own, and "
              "shares are floored at 10<super>-6</super> to avoid division by zero. A PSI below "
              "0.1 is usually read as stable, 0.1 to 0.25 as a moderate shift and above 0.25 as "
              "a major shift. For the baseline analysis, the PSI is computed for the ten most "
              "important features of the static model (by total gain) and for its scores, in "
              "every time bin against the training period."),
        ("p", "<b>PR-AUC.</b> Fraud is rare and investigators can only review the "
              "highest-scored alerts, so the main measure is the area under the "
              "precision-recall curve, computed as average precision [@dalpozzolo2018]:"),
        ("eq", "ap"),
        ("p", "where <i>P<sub>n</sub></i> and <i>R<sub>n</sub></i> are the precision and recall "
              "at the <i>n</i>-th threshold of the ranked scores. Unlike accuracy, PR-AUC is not "
              "inflated by the large number of legitimate transactions, and unlike ROC-AUC it "
              "focuses on how clean the top of the alert list is [@abdelnaby2023]. A model with "
              "no skill has a PR-AUC equal to the fraud rate."),
        ("p", "<b>Mean weekly PR-AUC.</b> In the stream, different weeks are scored by different "
              "model versions, whose scores are not calibrated to each other. Computing a single "
              "PR-AUC over all weeks would mix these scales and penalise strategies that "
              "retrain often, even when every one of their models ranks transactions well. We "
              "therefore compute PR-AUC separately for each week <i>w</i> of the stream and "
              "report the mean over the <i>W</i> weeks:"),
        ("eq", "weekly"),
        ("p", "Weeks without any fraud are skipped. The lowest weekly PR-AUC, the number of "
              "retrains and the F1-score at each model's own threshold are reported as "
              "secondary measures."),
    ]},
    {"title": "Statistical Validation", "blocks": [
        ("p", "<b>Paired day-block bootstrap.</b> Weekly PR-AUC is noisy, especially when a week "
              "contains only a few hundred frauds, and consecutive transactions are correlated. "
              "Confidence intervals are therefore computed with a bootstrap that resamples "
              "whole days rather than single transactions. Within every week, the days of that "
              "week are drawn with replacement, and the mean weekly PR-AUC is recomputed. The "
              "same resampled days are used for every strategy, so differences between "
              "strategies are paired and much of the shared noise cancels out. We use 500 "
              "resamples and report 95% percentile intervals. A difference is considered "
              "reliable when its interval excludes zero."),
        ("p", "<b>Walk-forward tuning.</b> A fair comparison requires that every strategy has "
              "good settings, chosen without looking at the data it is evaluated on. All 190 "
              "configurations in Table [@tab:grid] are run over a tuning stream that starts at "
              "35% of the rows, earlier than the evaluated stream. The evaluated stream is "
              "divided into folds of four weeks. At the start of each fold, each family "
              "(static, schedule, drift performance) selects the configuration with the "
              "highest score"),
        ("eq", "objective"),
        ("p", "computed only on weeks whose labels have matured by the start of the fold, that "
              "is, on information that would really have been available. "
              + KAPPA + " = 0.002 is a small cost per retrain that breaks ties in favour of "
              "simpler configurations. The selected configuration's results on the fold are "
              "then recorded, and the next fold repeats the selection. The result is an honest "
              "estimate of how each family would perform if an operator tuned it with the "
              "information available at the time."),
        ("table", "grid"),
        ("p", "<b>Training seeds.</b> The bootstrap only covers the randomness of the "
              "evaluation sample. LightGBM also has randomness in training, through row and "
              "feature subsampling. The key configurations are therefore re-run with five "
              "random seeds (42, 1, 2, 3 and 4) at label delays of 0 and 30 days, and for each "
              "comparison we report the mean and spread of the difference and the number of "
              "seeds in which its sign holds."),
        ("p", "<b>Replication.</b> Finally, the whole sequence, from the static baseline through "
              "the delay sweep, walk-forward tuning, seeds and the framework replay, is repeated "
              "unchanged on Sparkov and BAF, with the same scripts and settings. A finding that "
              "holds on IEEE-CIS but not on the other datasets is reported as dataset-specific."),
    ]},
    {"title": "Proposed Framework", "blocks": [
        ("p", "The framework turns the experimental findings into a working MLOps system. Its "
              "components are shown in Figure [@fig:architecture] and its settings in Table "
              "[@tab:framework]. It runs the loop of Figure [@fig:loop] every seven days: "
              "monitor, decide, retrain, check, and serve."),
        ("fig", "architecture"),
        ("p", "<b>Monitoring on delayed labels.</b> The monitor measures the live model's "
              "PR-AUC on the labels that matured during the last 14 days, restricted to "
              "transactions the model was not trained on. It also computes the PSI of the "
              "model's scores and of its ten most important numeric features against a sample "
              "of 50,000 rows of its training window, and logs both to MLflow. In line with the "
              "experimental results, the PSI is logged for diagnosis only and never triggers "
              "retraining on its own."),
        ("p", "<b>Retraining policy.</b> The primary policy is a fixed schedule: a new model is "
              "trained every 14 days on the last 60 days of labelled data. The schedule is "
              "complemented by a <i>safety net</i> that retrains early, at most once every seven "
              "days, when performance collapses between scheduled retrains:"),
        ("eq", "safety"),
        ("p", "with " + TAU + " = 0.30, " + LAMBDA + " = 3 and " + PI + " the fraud rate among "
              "the matured labels. The first condition compares live PR-AUC with the better of "
              "the model's own validation PR-AUC and the median of the last four live "
              "measurements. Using recent live history as well as the validation score matters: "
              "a model trained on corrupted data can report a poor validation score, which "
              "would otherwise lower the bar it is measured against. The second condition is a "
              "no-skill check, since a model without skill has a PR-AUC close to the fraud "
              "rate."),
        ("p", "<b>Training.</b> The trainer fits a LightGBM model with the settings of Table "
              "[@tab:hyper] on the policy's training window, which ends one label delay before "
              "the present. The window length can be set to zero for an expanding window. Each "
              "new model is registered in MLflow as a <i>challenger</i>, together with its "
              "training window, validation PR-AUC and threshold."),
        ("p", "<b>Promotion gate.</b> A challenger replaces the live model (the <i>champion</i>) "
              "only if it passes the promotion gate, following the champion-challenger pattern "
              "of MLOps practice [@pulicharla2019,@kodakandla2024]. The gate compares both "
              "models on the challenger's validation data, restricted to transactions that the "
              "champion was not trained on, so that neither model is judged on data it has "
              "seen. The challenger is promoted if"),
        ("eq", "gate"),
        ("p", "with " + DELTA + " = 0.01. Both models are re-scored through the production "
              "feature path rather than with scores stored at training time. This detail is "
              "essential: a model trained on wrongly scaled features looks good on its own, "
              "equally wrong, validation data, but fails on the inputs it will actually "
              "receive in production. If the comparison data contains fewer than 30 frauds, "
              "the comparison is not reliable and the challenger is promoted by default."),
        ("table", "framework"),
        ("p", "<b>Registry, serving and dashboard.</b> MLflow stores every model version with its "
              "parameters, metrics and training window. The model in service carries the alias "
              "<i>champion</i> and its predecessor the alias <i>previous</i>, so a rollback is a "
              "single alias change. A FastAPI service loads the champion and offers three "
              "endpoints: <i>/health</i> reports the version being served, <i>/predict</i> "
              "returns a fraud probability and flag for each submitted transaction, and "
              "<i>/reload</i> switches to a newly promoted champion without restarting. A "
              "monitoring dashboard, generated from the MLflow logs, shows weekly performance, "
              "drift, every retraining decision with its reason, and the history of model "
              "versions."),
        ("p", "<b>Replay.</b> The framework is evaluated by running it over the evaluated stream "
              "of each dataset as if it were live, with a label delay of 30 days. The replay "
              "uses the same data store, MLflow registry and model code as a deployment would. "
              "Its mean weekly PR-AUC is compared with the offline simulation of the same "
              "policy, which checks that the framework reproduces the experimental results, "
              "and with the static model."),
    ]},
    {"title": "Fault-Injection Testing", "blocks": [
        ("p", "A retraining pipeline that runs automatically will eventually train on bad data. "
              "To test whether the framework's safeguards stop such models, we inject the "
              "faults in Table [@tab:faults] into selected retraining jobs of the replay "
              "(by default the second and fourth scheduled retrains). The faults imitate common "
              "pipeline failures: a misaligned join, an outage of the chargeback feed, and a "
              "unit change between training and serving. In the first three, only the data "
              "read by the training job is corrupted, while the production feature path and "
              "the label store stay correct. The fourth corrupts the label store itself, which "
              "the gate also reads, and represents a failure the gate cannot see by design; it "
              "tests whether the safety net limits the damage."),
        ("table", "faults"),
        ("p", "Each fault scenario is replayed twice, once with the promotion gate and once "
              "without it, and compared with a clean replay. A safeguard is considered "
              "effective if the run with the fault and the safeguard stays close to the clean "
              "run, while the run without the safeguard loses performance. Each scenario uses "
              "its own MLflow store, so the main registry is never affected."),
    ]},
    {"title": "Implementation and Reproducibility", "blocks": [
        ("p", "The study is implemented in Python with LightGBM, pandas, NumPy and "
              "scikit-learn for the experiments, MLflow with an SQLite backend for tracking "
              "and the model registry, and FastAPI for the scoring service. The framework is "
              "organised as a Python package with separate modules for configuration, data "
              "access, training, monitoring, the retraining policy, the promotion gate, the "
              "registry, fault injection, serving and the dashboard, and is configured with "
              "one TOML file per dataset. Twenty-one automated tests on small synthetic data "
              "check the components and the full loop."),
        ("p", "The long experiments, which take several hours each, save their results after "
              "every completed label delay or seed and can resume after an interruption such as "
              "a power cut. One runner script reproduces the whole replication on any of the "
              "datasets, and a Kaggle notebook runs it on Kaggle's CPU servers. The experiments "
              "were run on a desktop computer and on Kaggle notebooks with four CPU cores. All "
              "code, configurations and instructions are kept in a public Git repository, so "
              "that every number in this thesis can be regenerated from the public datasets."),
    ]},
]}
