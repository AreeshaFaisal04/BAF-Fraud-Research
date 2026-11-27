# final run with the best params: train on months 0-4, threshold on month 5, test on months 6 and 7
# python scripts/03_final_test.py --model logreg
# saves test_<model>.csv and test_scores_<model>.csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np

from src.config import TAB_DIR, SEED
from src.data import load_clean
from src.experiment import final_test
from src.models import MODELS

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=list(MODELS))
args = parser.parse_args()
np.random.seed(SEED)

with open(TAB_DIR / f"best_params_{args.model}.json") as f:
    params = json.load(f)

df = load_clean()
metrics, test_scores, _ = final_test(args.model, params, df)
metrics["model"] = args.model
metrics.to_csv(TAB_DIR / f"test_{args.model}.csv", index=False)
test_scores.to_csv(TAB_DIR / f"test_scores_{args.model}.csv", index=False)

cols = ["split", "recall_at_5fpr", "fpr", "precision", "f1", "roc_auc", "pr_auc", "threshold"]
print(metrics[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
