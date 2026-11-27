# loading and cleaning the data + the feature groups
import os

import numpy as np
import pandas as pd

from .config import DATA_DIR, TARGET, MONTH_COL

# in these columns -1 means missing (datasheet)
NEG1_MISSING = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
# here any negative value means missing
NEGATIVE_MISSING = ["intended_balcon_amount"]

CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

# not used as features
DROP_ALWAYS = [MONTH_COL, "device_fraud_count"]  # month is only for the split, device_fraud_count is always 0

# each feature is in exactly one group (for the ablation)
FEATURE_GROUPS = {
    "applicant": ["income", "customer_age", "employment_status", "housing_status"],
    "address": ["prev_address_months_count", "current_address_months_count"],
    "email_phone": ["name_email_similarity", "email_is_free", "phone_home_valid", "phone_mobile_valid"],
    "velocity": ["velocity_6h", "velocity_24h", "velocity_4w", "zip_count_4w",
                 "bank_branch_count_8w", "date_of_birth_distinct_emails_4w", "days_since_request"],
    "credit_banking": ["credit_risk_score", "proposed_credit_limit", "intended_balcon_amount",
                       "payment_type", "bank_months_count", "has_other_cards"],
    "device_session": ["device_os", "source", "session_length_in_minutes", "keep_alive_session",
                       "device_distinct_emails_8w", "foreign_request"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def load_raw(path=None):
    # reads Base.csv or Base.parquet
    if path is None and os.environ.get("BAF_DATA"):
        path = os.environ["BAF_DATA"]   # path set in the colab notebook
    if path is None:
        pq = DATA_DIR / "Base.parquet"
        path = pq if pq.exists() else DATA_DIR / "Base.csv"
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def clean(df):
    # change the missing value codes to NaN (nothing is learned here so it is ok before splitting)
    df = df.copy()
    for c in NEG1_MISSING:
        df[c] = df[c].where(df[c] != -1)      # becomes float with NaN
    for c in NEGATIVE_MISSING:
        df[c] = df[c].where(df[c] >= 0)
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)
    return df


def load_clean(path=None):
    df = clean(load_raw(path))
    # check that every column is in a group
    leftover = set(df.columns) - set(ALL_FEATURES) - {TARGET} - set(DROP_ALWAYS)
    assert not leftover, f"columns not in any feature group: {leftover}"
    assert len(ALL_FEATURES) == len(set(ALL_FEATURES)), "a feature is in two groups"
    return df


def numeric_features(features):
    return [f for f in features if f not in CATEGORICAL]


def categorical_features(features):
    return [f for f in features if f in CATEGORICAL]


def features_without(groups_to_drop=(), extra_drop=()):
    # feature list without some groups (for the ablation)
    out = []
    for g, cols in FEATURE_GROUPS.items():
        if g in groups_to_drop:
            continue
        out += [c for c in cols if c not in extra_drop]
    return out
