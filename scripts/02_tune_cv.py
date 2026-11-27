"""
Random search (N_TRIALS configs) on the 5 forward-chaining folds for one model.

    python scripts/02_tune_cv.py --model logreg
    python scripts/02_tune_cv.py --model rf
    python scripts/02_tune_cv.py --model majority

Outputs (in outputs/tables):
    cv_trials_<model>.csv   every trial with its mean ± std
    cv_folds_<model>.csv    fold-by-fold scores of the best trial (for the box plot)
    best_params_<model>.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import numpy as np

from src.config import TAB_DIR, SEED, N_TRIALS, NEG_SAMPLE_FRAC
from src.data import load_clean
from src.experiment import random_search
from src.models import MODELS

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=list(MODELS))
parser.add_argument("--trials", type=int, default=N_TRIALS)
args = parser.parse_args()

np.random.seed(SEED)
TAB_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()
df = load_clean()
print(f"loaded {len(df)} rows; negative sampling frac for training = {NEG_SAMPLE_FRAC}")

trials, best_params, best_folds = random_search(args.model, df, n_trials=args.trials)

trials.to_csv(TAB_DIR / f"cv_trials_{args.model}.csv", index=False)
best_folds["model"] = args.model
best_folds.to_csv(TAB_DIR / f"cv_folds_{args.model}.csv", index=False)


def to_py(v):  # numpy types -> plain python for json
    return v.item() if hasattr(v, "item") else v


with open(TAB_DIR / f"best_params_{args.model}.json", "w") as f:
    json.dump({k: to_py(v) for k, v in best_params.items()}, f, indent=2)

print("\nbest params:", best_params)
print(best_folds[["fold", "recall_at_5fpr", "roc_auc", "pr_auc", "f1"]].to_string(index=False))
print(f"total time {(time.time() - t0) / 60:.1f} min")
