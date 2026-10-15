"""Hand-written LaTeX for the passages around the equations, where the Google Docs export
mixes equation-editor fragments into the text. The wording is the document's own; only the
mathematics is written properly. Each entry replaces the blocks from the first block whose
text starts with `start` (on `page`) up to, but not including, the block starting with `end`.
"""

EQ = {
    "4.1": r"\mathcal{L}(t) = \{\, x_j : t_j \le t - d \,\}",
    "4.2": r"\mathrm{AP}_{\mathrm{live}}(t) < (1 - \tau)\, \mathrm{AP}_{\mathrm{ref}}",
    "4.3": r"\mathrm{PSI} = \sum_{i=1}^{B} (c_i - r_i)\,\ln\frac{c_i}{r_i}",
    "4.4": r"\mathrm{AP} = \sum_{n} (R_n - R_{n-1})\,P_n",
    "4.5": r"\overline{\mathrm{AP}} = \frac{1}{W} \sum_{w=1}^{W} \mathrm{AP}_w",
    "4.6": r"J(c) = \overline{\mathrm{AP}}(c) - \kappa \cdot "
           r"\frac{N_{\mathrm{retrain}}(c)}{\text{number of 4-week periods}}",
    "4.7": r"\mathrm{AP}_{\mathrm{live}} < (1 - \tau)\,\max\big(\mathrm{AP}_{\mathrm{val}},\ "
           r"\mathrm{median}_{4}(\mathrm{AP}_{\mathrm{live}})\big) \quad\text{or}\quad "
           r"\mathrm{AP}_{\mathrm{live}} < \lambda\,\pi",
    "4.8": r"w^{*} = \arg\max_{w \in \mathcal{W}} \mathrm{AP}_{E}(m_w), \qquad "
           r"\mathcal{W} = \{\text{60 days},\ \text{all history}\}",
    "4.9": r"\mathrm{AP}(\text{challenger}) \ge \mathrm{AP}(\text{champion}) - \delta",
}


def eq(n):
    return "\\begin{equation*}\n  " + EQ[n] + " \\tag{" + n + "}\n\\end{equation*}\n"


PASSAGES = [
    {"page": 17, "start": "We model the time delay", "end": "4.1.3",
     "tex": "We model the time delay between a transaction and the use of its label as a fixed "
            "delay, $d$ days. The labelled data accessible for monitoring and training is given "
            "by, at time $t$,\n" + eq("4.1") +
            "where $t_j$ is the time of transaction $x_j$. The first model is trained at the "
            "beginning of the stream and is taught by the same rule, so it does not see any "
            "label that it might not have had. We consider delays of 0, 15, 30 and 60 days. The "
            "value of delay equal to 0 is commonly assumed in majority of the retraining "
            "research [3]\n\n"
            "The amount of time a cardholder takes to notice a fraudulent charge and dispute it "
            "is 30 to 60 days [2]; [5]. Not all Investigator feedback for some alerted "
            "transactions is modelled: all labels arrive with the same delay, which is smaller in "
            "a real system [5]. This is a slightly bad background for the performance based "
            "triggers which are discussed.\n"},
    {"page": 19, "start": "The performance method", "end": "The identical data",
     "tex": "The performance method is a simple error based trigger in the sense of DDM and "
            "ADWIN [3] [8] adapted for delayed labels. It computes the PR-AUC of the model on "
            "transactions with labels from the past 14 days not included in the model training, "
            "if there are at least 30 frauds in these transactions at each phase. It retrains "
            "when\n" + eq("4.2") +
            "where $\\mathrm{AP}_{\\mathrm{live}}(t)$ is that live PR-AUC, $\\tau$ is the "
            "tolerance (0.15 by default), and $\\mathrm{AP}_{\\mathrm{ref}}$ is either a model's "
            "validation PR-AUC, or the initial live measurement. Both drift techniques include a "
            "minimum of 14 days between retrains, to avoid a series of retrains owing to one "
            "drift occurrence.\n"},
    {"page": 19, "start": "Population Stability Index", "end": "PR AUC.",
     "tex": "Population Stability Index The PSI measures drift in a feature or in the scores of "
            "the model by comparing a current distribution with a reference distribution [2]:\n"
            + eq("4.3") +
            "with reference values binned into $B = 10$ bins at the deciles of the values. $r_i$ "
            "and $c_i$ are shares of the reference and current values in bin $i$. Shares are "
            "floored at $10^{-6}$ to avoid division by zero. Missing values are binned in their "
            "own bin. A PSI less than 0.1 is usually regarded steady, 0.1 to 0.25 is considered "
            "a moderate change, and greater than 0.25 is considered a major change. The PSI is "
            "determined for the 10 most relevant features of the static model (by total gain) "
            "and for the scores, in each time bin compared to the training period.\n"},
    {"page": 19, "start": "PR AUC.", "end": "Average PR AUC per week",
     "tex": "PR AUC. Fraud is exceedingly rare and investigators can only investigate the "
            "highest scoring alerts the main criterion is the area under the accuracy recall "
            "curve as determined by average precision [5]:\n" + eq("4.4") +
            "where $P_n$ and $R_n$ are the precision and recall at the $n$-th threshold of the "
            "ranked scores. The high number of genuine transactions does not inflate the PR-AUC. "
            "It also does not have the ROC-AUC bias on the cleanliness of the top of the alert "
            "list [9]. Note that the PR-AUC of a no-skill model is equal to the fraud rate.\n"},
    {"page": 20, "start": "Average PR AUC per week", "end": "4.1.5",
     # corrected: the document said "over all 1000 runs of data", which matches nothing in the
     # experiments, and split one sentence into two paragraphs
     "tex": "Average PR AUC per week. Each week a different version of the model is used to "
            "score, and the scores of different versions are not normalised to each other. A "
            "single PR-AUC computed over all weeks would blend such measures and often penalise "
            "training tactics even when their own models ranked transactions well. Therefore, "
            "we compute PR-AUC for each "
            "week $w$ of the stream independently, then average PR-AUC over the $W$ weeks:\n"
            + eq("4.5") +
            "If no fraud is found in one week, that week is skipped. Secondary measurements are "
            "minimum weekly PR-AUC, minimum number of retrains and F1-score at each model's own "
            "threshold.\n"},
    {"page": 20, "start": "(4.6)", "end": "Table 4.2",
     "tex": eq("4.6") +
            "$\\kappa = 0.002$ is a low cost per retrain, which breaks ties in favour of simpler "
            "configurations (as only computed on weeks whose labels have matured by the start of "
            "the week of the fold; i.e., on information that would really have been available). "
            "The results of the selected configuration on the fold are then recorded down and "
            "the same configuration is picked for the following fold. The result is a reasonable "
            "approximation of the performance of each family if an operator had the knowledge "
            "available at the time to tune the family to that information.\n"},
    {"page": 26, "start": "The basic policy", "end": "4.2.5",
     "tex": "The basic policy is a set schedule ; retrain a new model every 14 days on the latest "
            "60 days of labelled data . The schedule is augmented with a safety net that retrains "
            "early, at most once every 7 days, when performance collapses between scheduled "
            "retrains:\n" + eq("4.7") +
            "with $\\tau = 0.30$, $\\lambda = 3$ and $\\pi$ the fraud rate among the matured "
            "labels. The first condition compares the live PR-AUC to the best of the model's own "
            "validation PR-AUC and the median of the last four live measurements. Recent live "
            "history is also important, as is using the validation score. A model trained on "
            "corrupted data can claim a poor validation score, which would otherwise lower the "
            "bar it is assessed against. The other condition is a no-skill check, as a "
            "skill-less model gives a PR-AUC close to the fraud rate.\n"},
    {"page": 26, "start": "Automatic window picking", "end": "4.2.6",
     "tex": "\\textbf{Automatic window picking}: The trials indicated that the ideal training "
            "window depended on the dataset, 60 days on IEEE-CIS but full labelled history on "
            "Sparkov and BAF. A fixed window chosen on one dataset may be incorrect for another. "
            "So the framework can select the window itself. At each retrain it trains one "
            "challenger $m_w$ per candidate window $w$, here the last 60 days and all labelled "
            "history, and scores all of them on the same evaluation rows $E$: the most recent "
            "labelled transactions that none of the candidates was trained on and the champion "
            "was not trained on either. It only keeps\n" + eq("4.8") +
            "and only this challenger proceeds to the promotion gate, the rest are noted as not "
            "picked. The option depends solely on data that would be accessible at the time, and "
            "has no need for knowing which window is appropriate for the dataset. The extra cost "
            "is one extra training run per retrain as the candidates are trained on the same "
            "data except for their start date.\n"},
    {"page": 27, "start": "In the champion-challenger", "end": "4.2.7",
     "tex": "In the champion-challenger pattern of MLOps practice [7], [20], a challenger "
            "replaces the live model (the champion) if it clears the promotion gate. The gate "
            "evaluates both models on the challenger's validation data, which is restricted to "
            "transactions not observed by the champion at training, thus neither model is "
            "evaluated on data it has observed. The challenger is promoted if\n" + eq("4.9") +
            "where $\\delta = 0.01$. Both models are re-scored via the production feature path, "
            "not with scores saved at training time. This is the key: a model trained on "
            "incorrectly scaled features appears excellent on its own, equally incorrect, "
            "validation data, but fails on the inputs it will actually receive in production. If "
            "the comparison data contains less than 30frauds the comparison is unreliable and the "
            "challenger is promoted by default.\n"},
]
