# time based splits (months are never shuffled)
# fold 1: train on month 0 -> validate on month 1
# fold 2: train on months 0-1 -> validate on month 2
# ...
# fold 5: train on months 0-4 -> validate on month 5
# months 6 and 7 are only used at the end for testing
import numpy as np

from .config import DEV_MONTHS, TEST_MONTHS, FINAL_FIT_MONTHS, THRESHOLD_MONTH, MONTH_COL


def forward_chaining_folds(df):
    # gives (fold number, train rows, validation rows)
    months = df[MONTH_COL].to_numpy()
    for k, val_m in enumerate(DEV_MONTHS[1:], start=1):
        train_idx = np.where(months < val_m)[0]
        val_idx = np.where(months == val_m)[0]
        yield k, train_idx, val_idx


def final_split(df):
    # rows for the final run: training, threshold month and test months
    months = df[MONTH_COL].to_numpy()
    out = {
        "fit": np.where(np.isin(months, FINAL_FIT_MONTHS))[0],
        "threshold": np.where(months == THRESHOLD_MONTH)[0],
    }
    for m in TEST_MONTHS:
        out[f"test_m{m}"] = np.where(months == m)[0]
    out["test_m6_7"] = np.where(np.isin(months, TEST_MONTHS))[0]
    return out


def describe_folds(df, target="fraud_bool"):
    # size of each fold (for the paper)
    rows = []
    y = df[target].to_numpy()
    for k, tr, va in forward_chaining_folds(df):
        rows.append({
            "fold": k,
            "train_months": f"0-{k-1}" if k > 1 else "0",
            "val_month": k,
            "n_train": len(tr), "train_fraud_rate": y[tr].mean(),
            "n_val": len(va), "val_fraud_rate": y[va].mean(),
        })
    return rows
