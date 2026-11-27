# settings for the project (seed, months, sampling etc.)
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

TARGET = "fraud_bool"
MONTH_COL = "month"

# months 0-5 are for training/tuning, months 6-7 are the test set (same as the BAF paper)
DEV_MONTHS = [0, 1, 2, 3, 4, 5]
TEST_MONTHS = [6, 7]

# final model is trained on months 0-4 and the 5% FPR threshold is picked on month 5
FINAL_FIT_MONTHS = [0, 1, 2, 3, 4]
THRESHOLD_MONTH = 5

FPR_TARGET = 0.05

# number of random search trials for each model
N_TRIALS = 20

# for training use all fraud rows but only 25% of the non-fraud rows (much faster)
# validation and test months are not sampled
NEG_SAMPLE_FRAC = 0.25

# fairness groups: age over 50 vs 50 and under
AGE_COL = "customer_age"
AGE_CUTOFF = 50
