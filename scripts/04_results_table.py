# puts every model that has results into one table (cv mean ± std + test months)
# python scripts/04_results_table.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import TAB_DIR

ORDER = ["majority", "logreg", "rf", "lightgbm", "catboost", "mlp"]
NAMES = {"majority": "Majority class", "logreg": "Logistic regression", "rf": "Random forest",
         "lightgbm": "LightGBM", "catboost": "CatBoost", "mlp": "MLP"}
METRICS = [("recall_at_5fpr", "Recall@5%FPR"), ("roc_auc", "ROC-AUC"),
           ("pr_auc", "PR-AUC"), ("f1", "F1")]

rows = []
for m in ORDER:
    f = TAB_DIR / f"cv_folds_{m}.csv"
    if not f.exists():
        continue
    folds = pd.read_csv(f)
    row = {"model": NAMES[m]}
    for col, label in METRICS:
        row[f"CV {label}"] = f"{folds[col].mean():.3f} ± {folds[col].std(ddof=1):.3f}"
    t = TAB_DIR / f"test_{m}.csv"
    if t.exists():
        test = pd.read_csv(t).set_index("split")
        row["Test M6 Recall@5%FPR"] = f"{test.loc['month 6', 'recall_at_5fpr']:.3f}"
        row["Test M7 Recall@5%FPR"] = f"{test.loc['month 7', 'recall_at_5fpr']:.3f}"
        row["Test M6-7 FPR"] = f"{test.loc['months 6-7', 'fpr']:.3f}"
        row["Test M6-7 ROC-AUC"] = f"{test.loc['months 6-7', 'roc_auc']:.3f}"
    rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(TAB_DIR / "results_table.csv", index=False)
with open(TAB_DIR / "results_table.md", "w") as f:
    f.write(table.to_markdown(index=False))
with open(TAB_DIR / "results_table.tex", "w") as f:
    f.write(table.to_latex(index=False).replace("±", r"$\pm$").replace("%", r"\%"))
print(table.to_markdown(index=False))
