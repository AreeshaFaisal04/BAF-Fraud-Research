"""
Turns the CSV/JSON outputs into the LaTeX tables used in the paper (paper/tables/).
Run after 01-04. Numbers are never typed by hand.

    python scripts/05_paper_tables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import pandas as pd

from src.config import TAB_DIR, ROOT
from src.data import FEATURE_GROUPS

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


# ---------------- fold sizes ----------------
folds = pd.read_csv(TAB_DIR / "fold_sizes.csv")
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Forward-chaining folds on the development months (training sizes before sampling).}",
         r"\label{tab:folds}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{ccrrrr}", r"\hline",
         r"Fold & Train mon. & Train rows & Fraud & Val. mon. & Fraud \\", r"\hline"]
for _, r in folds.iterrows():
    tm = "0" if r.fold == 1 else f"0--{r.fold - 1}"
    lines.append(f"{r.fold} & {tm} & {int(r.n_train):,} & {r.train_fraud_rate*100:.2f}\\% & "
                 f"{int(r.val_month)} & {r.val_fraud_rate*100:.2f}\\% \\\\")
lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
(OUT / "fold_sizes.tex").write_text("\n".join(lines) + "\n")

# ---------------- tuning ranges + chosen values ----------------
RANGES = {
    "logreg": [("C", "log-unif. $[10^{-3}, 10^{2}]$", "C")],
    "rf": [("n\\_estimators", "integer $[100, 300]$", "n_estimators"),
           ("max\\_depth", "\\{8, 12, 16, 20, 24\\}", "max_depth"),
           ("min\\_samples\\_leaf", "\\{1, 5, 10, 25, 50, 100\\}", "min_samples_leaf"),
           ("max\\_features", "\\{sqrt, 0.2, 0.33\\}", "max_features"),
           ("max\\_samples", "uniform $[0.05, 0.30]$", "max_samples")],
}
NAMES = {"logreg": "Log. reg.", "rf": "Rand. forest"}


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}" if v < 1 else f"{v:.2f}"
    return esc(v)


lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Random-search ranges (20 trials per model) and selected values.}",
         r"\label{tab:tuning}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{llll}", r"\hline",
         r"Model & Hyperparameter & Search range & Selected \\", r"\hline"]
for m, rows in RANGES.items():
    f = TAB_DIR / f"best_params_{m}.json"
    best = json.loads(f.read_text()) if f.exists() else {}
    for i, (label, rng, key) in enumerate(rows):
        name = NAMES[m] if i == 0 else ""
        lines.append(f"{name} & {label} & {rng} & {fmt(best.get(key, '--'))} \\\\")
    lines.append(r"\hline")
lines.append(r"% Members 2 and 3: add LightGBM, CatBoost and MLP rows here (or regenerate with this script).")
lines += [r"\end{tabular}", r"\end{table}"]
(OUT / "tuning_table.tex").write_text("\n".join(lines) + "\n")

# ---------------- feature groups ----------------
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Feature groups used in the dataset description and the ablation study.}",
         r"\label{tab:groups}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.55cm}>{\raggedright\arraybackslash}p{6.3cm}}", r"\hline",
         r"Group & Features \\", r"\hline"]
pretty = {"applicant": "Applicant", "address": "Address history", "email_phone": "Email and phone",
          "velocity": "Application velocity", "credit_banking": "Credit and banking",
          "device_session": "Device and session"}
for g, cols in FEATURE_GROUPS.items():
    lines.append(f"{pretty[g]} ({len(cols)}) & " + ", ".join(r"\texttt{\footnotesize " + esc(c).replace("\\_", "\\_\\allowbreak ") + "}" for c in cols) + r" \\")
lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
(OUT / "feature_groups.tex").write_text("\n".join(lines) + "\n")

# ---------------- Member 1 results (CV + test) ----------------
order = [("majority", "Majority class"), ("logreg", "Logistic regression"), ("rf", "Random forest")]
lines = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Baselines and random forest: mean $\pm$ std over the five validation folds, and the "
         r"test months with the threshold fixed on month 5. (To be merged into the main results table.)}",
         r"\label{tab:member1-results}", r"\footnotesize",
         r"\begin{tabular}{lcccc|ccc}", r"\hline",
         r" & \multicolumn{4}{c|}{Validation folds (months 1--5)} & \multicolumn{3}{c}{Test} \\",
         r"Model & Recall@5\%FPR & ROC-AUC & PR-AUC & F1 & Recall M6 & Recall M7 & FPR M6--7 \\", r"\hline"]
for m, name in order:
    f = TAB_DIR / f"cv_folds_{m}.csv"
    if not f.exists():
        continue
    cv = pd.read_csv(f)
    cells = [f"{cv[c].mean():.3f} $\\pm$ {cv[c].std(ddof=1):.3f}" for c in ["recall_at_5fpr", "roc_auc", "pr_auc", "f1"]]
    t = pd.read_csv(TAB_DIR / f"test_{m}.csv").set_index("split")
    cells += [f"{t.loc['month 6', 'recall_at_5fpr']:.3f}", f"{t.loc['month 7', 'recall_at_5fpr']:.3f}",
              f"{t.loc['months 6-7', 'fpr']*100:.2f}\\%"]
    lines.append(name + " & " + " & ".join(cells) + r" \\")
lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
(OUT / "member1_results.tex").write_text("\n".join(lines) + "\n")

print("wrote:", ", ".join(p.name for p in sorted(OUT.glob("*.tex"))))
