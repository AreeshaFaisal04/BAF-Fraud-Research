"""
Shared settings for the whole group.
Members 2 and 3 should import from here instead of copying values,
so that every model uses the same seed, folds, budget and FPR target.
"""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

TARGET = "fraud_bool"
MONTH_COL = "month"

# time split (same as Jesus et al. 2022): months 0-5 for development, 6-7 for test
DEV_MONTHS = [0, 1, 2, 3, 4, 5]
TEST_MONTHS = [6, 7]

# final model: fit on months 0-4, choose the 5% FPR threshold on month 5
FINAL_FIT_MONTHS = [0, 1, 2, 3, 4]
THRESHOLD_MONTH = 5

FPR_TARGET = 0.05

# tuning budget agreed by the group (same for every model)
N_TRIALS = 20

# Keep every fraud row but only this fraction of non-fraud rows when TRAINING.
# Validation and test months are never sampled. Set to 1.0 to use everything
# (needs a lot more time for random forest / MLP on a laptop).
NEG_SAMPLE_FRAC = 0.25

# age group used in the fairness part (paper: >50 vs <=50)
AGE_COL = "customer_age"
AGE_CUTOFF = 50
