"""Chapter 4 (Results): text, figures and tables. Merged into content.py.

Every number is taken from the experiment outputs (outputs/, outputs/sparkov/, outputs/baf/
and outputs/<dataset>/window_selection/); figures are drawn by results_figures.py.
PR-AUC means mean weekly PR-AUC unless stated otherwise; intervals are 95% paired day-block
bootstrap intervals (see the statistical validation) unless stated otherwise.
"""

FIGURES = {
    "decay": ("figures/results_decay.png",
              "Static model after deployment, in ten consecutive time bins of the test period: "
              "PR-AUC (top) and the largest PSI among the ten most important features (bottom). "
              "Each panel has its own scale"),
    "delaygain": ("figures/results_delay.png",
                  "Gain in mean weekly PR-AUC over the static model for scheduled retraining "
                  "every 14 days, by label delay, with 95% paired bootstrap intervals"),
    "window": ("figures/results_window.png",
               "Framework replays (30-day label delay): gain in mean weekly PR-AUC over the "
               "static model for each training-window choice, with 95% bootstrap intervals "
               "over weeks"),
    "gate": ("figures/results_gate.png",
             "Fault injection on IEEE-CIS: change in mean weekly PR-AUC against a clean replay "
             "when two retraining jobs are faulty, with the promotion gate on and off"),
}

TABLES = {
    "decay": {
        "caption": "Performance of the static model and drift of its inputs after deployment",
        "widths": [0.34, 0.22, 0.22, 0.22],
        "header": ["", "IEEE-CIS", "Sparkov", "BAF"],
        "rows": [
            ["PR-AUC over the whole test period", "0.510", "0.936", "0.164"],
            ["ROC-AUC over the whole test period", "0.888", "0.997", "0.871"],
            ["PR-AUC, first time bin", "0.581", "0.954", "0.164"],
            ["PR-AUC, lowest time bin", "0.388 (bin 6)", "0.893 (bin 10)", "0.149 (bin 6)"],
            ["PR-AUC, last time bin", "0.534", "0.893", "0.169"],
            ["Largest feature PSI (feature)", "0.17 (D10)", "0.53 (card_txns_24h)",
             "4.60 (velocity_4w)"],
            ["Score-PSI trigger (threshold 0.1)", "never fired; maximum score PSI 0.017",
             "fired once (1 of 4 label delays)", "never fired"],
        ],
    },
    "sweep": {
        "caption": "Gain in mean weekly PR-AUC over the static model by label delay (retraining "
                   "every 14 days; 95% intervals; bold: interval excludes zero)",
        "widths": [0.13, 0.1, 0.13, 0.22, 0.22, 0.2],
        "header": ["Dataset", "Delay", "Static", "All history", "Last 60 days",
                   "Drift: performance"],
        "rows": [
            ["IEEE-CIS", "0 d", "0.507", "<b>+0.033</b> [0.027, 0.040]", "<b>+0.078</b> [0.067, 0.088]", "+0.030 (2 retrains)"],
            ["", "15 d", "0.491", "<b>+0.025</b> [0.020, 0.029]", "<b>+0.062</b> [0.052, 0.072]", "+0.020 (2)"],
            ["", "30 d", "0.474", "<b>+0.028</b> [0.022, 0.033]", "<b>+0.056</b> [0.046, 0.064]", "+0.027 (3)"],
            ["", "60 d", "0.472", "<b>+0.008</b> [0.002, 0.014]", "<b>+0.010</b> [0.004, 0.016]", "+0.010 (4)"],
            ["Sparkov", "0 d", "0.931", "<b>+0.016</b> [0.013, 0.020]", "<b>-0.258</b> [-0.273, -0.242]", "0.000 (0)"],
            ["", "15 d", "0.928", "<b>+0.017</b> [0.013, 0.021]", "<b>-0.214</b> [-0.228, -0.199]", "0.000 (0)"],
            ["", "30 d", "0.924", "<b>+0.019</b> [0.016, 0.023]", "<b>-0.173</b> [-0.187, -0.160]", "0.000 (0)"],
            ["", "60 d", "0.923", "<b>+0.019</b> [0.015, 0.023]", "<b>-0.198</b> [-0.211, -0.183]", "0.000 (0)"],
            ["BAF", "0 d", "0.162", "<b>+0.011</b> [0.007, 0.015]", "-0.003 [-0.007, 0.002]", "0.000 (0)"],
            ["", "15 d", "0.161", "<b>+0.008</b> [0.004, 0.012]", "-0.005 [-0.011, 0.000]", "0.000 (0)"],
            ["", "30 d", "0.161", "<b>+0.009</b> [0.004, 0.014]", "<b>-0.007</b> [-0.013, -0.002]", "0.000 (0)"],
            ["", "60 d", "0.155", "<b>+0.010</b> [0.006, 0.015]", "+0.002 [-0.003, 0.007]", "+0.001 (1)"],
        ],
    },
    "walkforward": {
        "caption": "Walk-forward tuning: each family re-tuned every four weeks on matured labels "
                   "only (mean weekly PR-AUC, total retrains, and differences with 95% intervals)",
        "widths": [0.13, 0.09, 0.13, 0.17, 0.25, 0.23],
        "header": ["Dataset", "Delay", "Static", "Schedule (retrains)", "Schedule - static",
                   "Drift trigger - schedule"],
        "rows": [
            ["IEEE-CIS", "0 d", "0.470", "0.614 (12)", "<b>+0.144</b> [0.132, 0.158]", "<b>-0.069</b> [-0.082, -0.060]"],
            ["", "15 d", "0.468", "0.547 (12)", "<b>+0.079</b> [0.070, 0.091]", "<b>-0.034</b> [-0.044, -0.027]"],
            ["", "30 d", "0.458", "0.518 (6)", "<b>+0.059</b> [0.049, 0.069]", "<b>-0.032</b> [-0.038, -0.025]"],
            ["Sparkov", "0 d", "0.916", "0.945 (12)", "<b>+0.029</b> [0.024, 0.033]", "-0.001 [-0.003, 0.001]"],
            ["", "15 d", "0.913", "0.944 (12)", "<b>+0.031</b> [0.026, 0.036]", "<b>-0.007</b> [-0.010, -0.004]"],
            ["", "30 d", "0.917", "0.943 (12)", "<b>+0.027</b> [0.023, 0.031]", "<b>-0.007</b> [-0.009, -0.004]"],
            ["BAF", "0 d", "0.163", "0.174 (8)", "<b>+0.011</b> [0.006, 0.016]", "<b>-0.011</b> [-0.016, -0.006]"],
            ["", "15 d", "0.157", "0.168 (8)", "<b>+0.011</b> [0.007, 0.016]", "<b>-0.006</b> [-0.011, -0.002]"],
            ["", "30 d", "0.159", "0.162 (7)", "+0.003 [-0.003, 0.008]", "+0.001 [-0.003, 0.006]"],
        ],
    },
    "seeds": {
        "caption": "Key differences over five training seeds (mean, and the number of seeds in "
                   "which the difference is positive). Best schedule: last 60 days on IEEE-CIS, all "
                   "history on Sparkov and BAF",
        "widths": [0.13, 0.09, 0.26, 0.26, 0.26],
        "header": ["Dataset", "Delay", "All history - static", "Last 60 days - static",
                   "Default drift trigger - best schedule"],
        "rows": [
            ["IEEE-CIS", "0 d", "+0.033 (5/5)", "+0.080 (5/5)", "-0.055 (0/5)"],
            ["", "30 d", "+0.022 (5/5)", "+0.046 (5/5)", "-0.026 (0/5)"],
            ["Sparkov", "0 d", "+0.012 (5/5)", "-0.194 (0/5)", "-0.012 (0/5)"],
            ["", "30 d", "+0.016 (5/5)", "-0.171 (0/5)", "-0.016 (0/5)"],
            ["BAF", "0 d", "+0.006 (5/5)", "-0.006 (0/5)", "-0.006 (0/5)"],
            ["", "30 d", "+0.009 (5/5)", "-0.007 (0/5)", "-0.009 (0/5)"],
        ],
    },
    "replay": {
        "caption": "Framework replays at a 30-day label delay: mean weekly PR-AUC (retrains, "
                   "rejected by the gate) and the offline simulation of the same policy",
        "widths": [0.22, 0.26, 0.26, 0.26],
        "header": ["", "IEEE-CIS (12 weeks)", "Sparkov (52 weeks)", "BAF (19 weeks)"],
        "rows": [
            ["Static model", "0.478", "0.924", "0.161"],
            ["Last 60 days", "<b>0.529</b> (5, 0)", "0.752 (26, 11)", "0.153 (9, 0)"],
            ["All history", "0.504 (6, 0)", "<b>0.942</b> (25, 0)", "<b>0.169</b> (9, 1)"],
            ["Automatic window", "<b>0.529</b> (5, 0)", "<b>0.942</b> (25, 1)", "0.165 (9, 1)"],
            ["Window chosen", "60 days in 5 of 5", "all history in 25 of 25",
             "all history in 7 of 9"],
            ["Offline simulation: last 60 days / all history", "0.530 / 0.502", "0.751 / 0.943",
             "0.154 / 0.170"],
        ],
    },
    "gatefaults": {
        "caption": "Fault injection on IEEE-CIS (two faulty retraining jobs, 30-day label delay)",
        "widths": [0.24, 0.11, 0.15, 0.16, 0.34],
        "header": ["Fault", "Gate", "Mean PR-AUC", "Change vs clean", "Faulty models promoted"],
        "rows": [
            ["None (clean)", "on", "0.531", "-", "-"],
            ["Label shuffle", "on", "0.519", "-0.012", "0 of 2 (0.035 vs 0.475; 0.034 vs 0.620)"],
            ["", "off", "0.451", "-0.080", "2 of 2"],
            ["Label loss (80%)", "on", "0.519", "-0.012", "0 of 2 (0.275 vs 0.475; 0.445 vs 0.620)"],
            ["", "off", "0.492", "-0.039", "2 of 2"],
            ["Feature unit bug", "on", "0.519", "-0.012", "0 of 2 (0.464 vs 0.475; 0.597 vs 0.620)"],
            ["", "off", "0.512", "-0.019", "2 of 2"],
            ["Upstream label shuffle", "on", "0.451", "-0.080", "2 of 2; safety net replaced each "
             "after 7 days"],
            ["", "off", "0.451", "-0.080", "2 of 2"],
        ],
    },
    "answers": {
        "caption": "Answers to the research questions",
        "widths": [0.1, 0.9],
        "header": ["RQ", "Answer"],
        "rows": [
            ["RQ1", "Performance after deployment changed differently on each dataset: it fell by "
             "a third on IEEE-CIS but stayed stable on Sparkov and BAF. Label-free drift "
             "measures did not reflect this: input and score drift stayed low while IEEE-CIS "
             "decayed, and input drift was strong (PSI up to 4.6) while Sparkov and BAF did not."],
            ["RQ2", "Scheduled retraining improved performance on every dataset when its window "
             "suited the data, with gains of 0.008 to 0.078 PR-AUC. On IEEE-CIS the gain shrank "
             "from 0.078 to 0.010 as the label delay grew from 0 to 60 days."],
            ["RQ3", "No. Tuned fairly with walk-forward tuning, drift-triggered retraining was "
             "worse than scheduled retraining in 7 of 9 comparisons and never better beyond "
             "noise; it retrained less often. The training window mattered more than the "
             "trigger."],
            ["RQ4", "Yes. The framework reproduced the offline results within 0.002 PR-AUC, "
             "blocked every faulty model the gate could observe, limited the damage of the "
             "unobservable fault through the safety net, and, with automatic window selection, "
             "matched the best fixed window on two datasets and came within 0.005 on the third."],
        ],
    },
}

CH4 = {"title": "Results", "sections": [
    {"title": "Overview", "blocks": [
        ("p", "This chapter reports the results of the experiments described in the previous "
              "chapter, in the order of the research questions. Section [@sec:rq1] examines how the static model "
              "behaves after deployment and whether drift measures reflect it (RQ1). Section "
              "[@sec:rq2] measures the benefit of retraining under label delay (RQ2), and Section "
              "[@sec:rq3] compares drift-triggered with scheduled retraining under fair tuning (RQ3). "
              "Section [@sec:rq4] evaluates the framework: its replay, its safeguards under fault "
              "injection, and automatic window selection (RQ4). Section [@sec:summary] summarises "
              "the answers."),
        ("p", "All results are mean weekly PR-AUC (Equation [@eq:weekly]) unless stated "
              "otherwise. Intervals are 95% paired day-block bootstrap intervals over 500 "
              "resamples, and a difference is called reliable when its interval excludes zero. "
              "The three datasets differ strongly in difficulty: a model without skill would "
              "score 0.035 on IEEE-CIS, 0.005 on Sparkov and 0.011 on BAF, so PR-AUC values "
              "should be compared within a dataset, not across datasets."),
    ]},
    {"title": "Model Performance and Drift after Deployment (RQ1)", "key": "rq1", "blocks": [
        ("p", "Table [@tab:decay] and Figure [@fig:decay] show how the static model performed "
              "over the test period of each dataset, which follows the training and validation "
              "periods in time."),
        ("table", "decay"),
        ("fig", "decay"),
        ("p", "<b>IEEE-CIS decays, and the drift measures do not see it.</b> The PR-AUC of the "
              "static model fell from 0.581 in the first time bin to 0.388 in the sixth, a "
              "relative loss of a third, before partly recovering to 0.534. ROC-AUC fell from "
              "0.920 to 0.849 over the same bins. Over the same period, none of the ten most "
              "important features reached the 0.25 level of a major shift (the largest PSI was "
              "0.17), and the PSI of the model's scores never exceeded 0.017. A label-free "
              "monitor would therefore have reported a stable system during the largest loss of "
              "performance in the study. This agrees with the decay that Menezes and Filho "
              "observed for graph models on the same dataset [@menezes2025], and shows that the "
              "decay is not visible in the input distribution."),
        ("p", "<b>Sparkov and BAF drift, and the model does not decay.</b> On Sparkov the "
              "per-card activity features drifted strongly in the last bin (PSI 0.53), and on "
              "BAF the velocity features shifted far beyond any usual threshold in every bin "
              "(PSI up to 4.60), yet PR-AUC stayed within 0.89 to 0.98 on Sparkov and within "
              "0.15 to 0.19 on BAF, without a downward trend. A drift monitor on these "
              "features would have raised an alarm in almost every week. The score-PSI "
              "trigger, which looks at the model's outputs rather than its inputs, fired only "
              "once on Sparkov, at one of the four label delays, and never on BAF."),
        ("p", "<b>Answer to RQ1.</b> The performance of a deployed fraud model can decay "
              "substantially, but whether it does depends on the data, and label-free drift "
              "measures are not a reliable indicator of it in either direction: they missed real "
              "decay on IEEE-CIS and signalled drift that did not matter on Sparkov and BAF. "
              "This is the situation that Shakil et al. anticipated when they noted that "
              "statistical drift does not always mean lower performance [@shakil2025], now "
              "observed on natural drift in fraud data."),
    ]},
    {"title": "Benefit of Retraining under Label Delay (RQ2)", "key": "rq2", "blocks": [
        ("p", "Table [@tab:sweep] and Figure [@fig:delaygain] compare scheduled retraining every "
              "14 days with the static model at label delays of 0, 15, 30 and 60 days."),
        ("table", "sweep"),
        ("fig", "delaygain"),
        ("p", "<b>Retraining helps when its window suits the data.</b> On IEEE-CIS both windows "
              "improved on the static model at every delay, and the last 60 days was clearly "
              "better than all history (+0.078 against +0.033 at no delay). On Sparkov and BAF "
              "the order reversed: retraining on all history gained +0.016 to +0.019 and +0.008 "
              "to +0.011 at every delay, with every interval excluding zero, while the 60-day "
              "window was <i>worse</i> than not retraining at all, dramatically so on Sparkov "
              "(-0.17 to -0.26). Sparkov has only 0.53% fraud and a stable fraud pattern: 60 "
              "days contain too few frauds to learn it again, and the model gains nothing from "
              "forgetting older data. IEEE-CIS, in contrast, drifts, and older data teaches the "
              "model patterns that no longer hold."),
        ("p", "<b>Label delay erodes the benefit.</b> On IEEE-CIS the gain of the 60-day window "
              "fell from +0.078 with immediate labels to +0.062 at 15 days, +0.056 at 30 days "
              "and +0.010 at 60 days, while the static model itself declined only from 0.507 to "
              "0.472. With 60 days of delay, a retrained model learns from data that is already "
              "two months old, which on a drifting dataset is little better than the original "
              "training data. On Sparkov and BAF the gain of all-history retraining hardly "
              "changed with the delay, which is consistent with their lack of decay: when the "
              "pattern is stable, old labels are still useful."),
        ("p", "<b>Untuned drift triggers rarely act.</b> With default settings, the performance "
              "trigger retrained two to four times on IEEE-CIS and reached between a third and "
              "half of the gain "
              "of the 60-day schedule at delays up to 30 days. On Sparkov and BAF it almost "
              "never fired, because performance never dropped by 15%, and it behaved like the "
              "static model."),
        ("p", "<b>Answer to RQ2.</b> Retraining improved performance on every dataset, but only "
              "with a training window that suited the data, and on the drifting dataset its "
              "benefit shrank sharply as labels arrived later; at 60 days of delay most of the "
              "benefit was gone. This confirms, on public data, the importance of verification "
              "latency stressed by Dal Pozzolo et al. [@dalpozzolo2018]."),
    ]},
    {"title": "Drift-Triggered versus Scheduled Retraining (RQ3)", "key": "rq3", "blocks": [
        ("p", "The comparison in Section [@sec:rq2] used default settings. Table [@tab:walkforward] "
              "gives the fair comparison: every family was re-tuned every four weeks with "
              "walk-forward tuning over 190 configurations, using only labels that had matured "
              "at the time."),
        ("table", "walkforward"),
        ("p", "<b>The tuned schedule beat or matched the tuned trigger everywhere.</b> On "
              "IEEE-CIS the schedule was better than the drift trigger by 0.069, 0.034 and "
              "0.032 PR-AUC at delays of 0, 15 and 30 days, all reliable. On Sparkov the "
              "trigger came close with far fewer retrains (2 to 8 against 12): the gap was "
              "within noise with immediate labels and 0.007 at 15 and 30 days. On BAF the "
              "schedule was better at 0 and 15 days, and at 30 days neither retraining family "
              "was reliably better than the static model, because the gains on BAF are small. "
              "Of the nine comparisons, the trigger was reliably worse in seven and never "
              "reliably better."),
        ("p", "The reason is visible in how the tuned trigger behaves under label delay. It can "
              "only react to a drop in performance after the drop has been confirmed by labels "
              "that are at least one delay old, and a drop must exceed noise in weekly PR-AUC "
              "before it is acted upon. A schedule keeps the model fresh without waiting for "
              "evidence. Where the trigger came close, as on Sparkov, it did so by retraining "
              "rarely on a stable dataset where retraining matters little."),
        ("table", "seeds"),
        ("p", "<b>The results are not due to training randomness.</b> Table [@tab:seeds] repeats "
              "the key comparisons with five LightGBM seeds. Every difference in the table had "
              "the same sign in all five seeds on every dataset. The spread of a single configuration across seeds was "
              "small compared with the differences between strategies: the median standard "
              "deviation was 0.004 on IEEE-CIS, 0.010 on Sparkov and 0.003 on BAF."),
        ("p", "<b>Answer to RQ3.</b> No. When both are tuned fairly and labels arrive late, "
              "drift-triggered retraining did not outperform scheduled retraining on any of the "
              "three datasets; it was worse in most comparisons and at best equal, although it "
              "retrained less often. This contradicts the results of Wong and Perumal "
              "[@wong2025] and Yelleti [@yelleti2025], which were obtained with injected drift "
              "or immediate labels. A stronger effect than the choice of trigger was the choice "
              "of training window, which changed the result by up to 0.26 PR-AUC."),
    ]},
    {"title": "Evaluation of the Framework (RQ4)", "key": "rq4", "blocks": [
        ("p", "<b>Replay.</b> Table [@tab:replay] reports the framework replayed over the stream "
              "of each dataset at a 30-day label delay, with four window settings run on the "
              "same Kaggle machine. The framework reproduced the offline simulation of the same "
              "policy closely: 0.529 against 0.530 for the 60-day window on IEEE-CIS, 0.942 "
              "against 0.943 for all history on Sparkov, and 0.169 against 0.170 on BAF. An "
              "earlier replay of the default configuration on a desktop computer gave 0.531 on "
              "IEEE-CIS. The MLOps components, from the registry and promotion gate to the "
              "monitoring loop, therefore add their safeguards without changing the "
              "performance of the policy they run."),
        ("table", "replay"),
        ("p", "<b>Automatic window selection.</b> No fixed window suited all three datasets: the "
              "60-day window was best on IEEE-CIS (0.529) but lost 0.172 against the static "
              "model on Sparkov, while all history was best on Sparkov and BAF but only second "
              "on IEEE-CIS. With automatic selection, the framework chose the 60-day window at "
              "all five retrains on IEEE-CIS and all history at all 25 retrains on Sparkov, "
              "matching the best fixed window on both (difference 0.000 and -0.0002). On BAF it "
              "chose the 60-day window at the first two retrains, when the two candidates were "
              "close and the shorter window scored slightly higher on recent data, and all "
              "history at the remaining seven. It ended 0.0045 below the best fixed window on "
              "BAF, a small but reliable difference (95% interval over weeks -0.008 to -0.001), "
              "and 0.004 above the static model. Figure [@fig:window] shows that the automatic "
              "setting avoided the large loss of a wrongly chosen window on every dataset "
              "without knowing in advance which window suited the data."),
        ("fig", "window"),
        ("p", "<b>Safeguards under fault injection.</b> Table [@tab:gatefaults] and Figure "
              "[@fig:gate] show the IEEE-CIS replay with two faulty retraining jobs. With the "
              "promotion gate on, all six faulty models produced by training-job faults were "
              "rejected. Shuffled labels gave models with a PR-AUC of 0.035, no better than "
              "chance, and the label outage gave 0.275 and 0.445 against champions of 0.475 and "
              "0.620. The feature unit bug was the hardest case: its models scored 0.464 and "
              "0.597 against 0.475 and 0.620, and were rejected only because they were re-scored "
              "on production features and fell just beyond the tolerance of 0.01. With the gate "
              "on, each of these three faults cost 0.012 PR-AUC, the price of keeping an older champion for "
              "two weeks. Without the gate the faulty models were promoted and the cost rose to "
              "0.019, 0.039 and 0.080."),
        ("table", "gatefaults"),
        ("fig", "gate"),
        ("p", "The upstream fault, which corrupts the label store that the gate itself reads, "
              "passed the gate as expected, since both models looked equally poor on corrupted "
              "labels. Here the safety net took over: seven days after each faulty promotion, "
              "the live PR-AUC on matured labels had fallen to 0.045 and 0.036, below both 70% "
              "of the reference and three times the fraud rate, and the framework retrained "
              "early. The damage was limited to one week per fault (-0.080 in total). Before "
              "the reference of the safety net included recent live performance (Section [@sec:safety]), "
              "the same fault went undetected and cost -0.149."),
        ("p", "<b>Answer to RQ4.</b> Yes. Built on the findings of RQ1 to RQ3, the framework "
              "sustained the performance of the best offline policy in a full replay, prevented "
              "every faulty model it could observe from reaching production, recovered within "
              "a week from the fault it could not observe, and, with automatic window "
              "selection, performed at or near the best fixed window on all three datasets."),
    ]},
    {"title": "Summary of Results", "key": "summary", "blocks": [
        ("p", "Table [@tab:answers] summarises the answers to the research questions. Taken "
              "together, the results support the design of the proposed framework: performance "
              "is monitored on delayed labels rather than inferred from drift, retraining "
              "follows a schedule rather than a drift alarm, the training window is selected "
              "from the data, and every new model must pass a promotion gate. The next chapter "
              "discusses what these results mean, how they relate to earlier work, and where "
              "they are limited."),
        ("table", "answers"),
    ]},
]}
