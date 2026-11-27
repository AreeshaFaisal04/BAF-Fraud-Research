"""
Exploratory analysis of the BAF Base dataset.
Writes summary tables to outputs/tables and figures to outputs/figures.

    python scripts/01_eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIG_DIR, TAB_DIR, TARGET, AGE_COL, AGE_CUTOFF
from src.data import load_raw, clean, NEG1_MISSING, NEGATIVE_MISSING, CATEGORICAL, FEATURE_GROUPS
from src.folds import describe_folds
from src.plotstyle import use_paper_style, save, BLUE, ORANGE, GREY, TEXT2

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
use_paper_style()

raw = load_raw()
print("shape:", raw.shape)

# ---------- 1. basic checks (also used to confirm we have the right file) ----------
overview = {
    "rows": int(len(raw)),
    "columns": int(raw.shape[1]),
    "fraud_cases": int(raw[TARGET].sum()),
    "fraud_rate": float(raw[TARGET].mean()),
    "months": sorted(int(m) for m in raw["month"].unique()),
    "duplicate_rows": int(raw.duplicated().sum()),
    "true_nan_cells": int(raw.isna().sum().sum()),
    "device_fraud_count_unique": sorted(int(v) for v in raw["device_fraud_count"].unique()),
    "n_numeric": int(raw.shape[1] - len(CATEGORICAL) - 2),  # minus target and month
    "n_categorical": len(CATEGORICAL),
}
print(json.dumps(overview, indent=2))
with open(TAB_DIR / "eda_overview.json", "w") as f:
    json.dump(overview, f, indent=2)

# column ranges, to compare with the datasheet
num_cols = raw.select_dtypes("number").columns
ranges = pd.DataFrame({"min": raw[num_cols].min(), "max": raw[num_cols].max(),
                       "n_unique": raw[num_cols].nunique()})
ranges.to_csv(TAB_DIR / "eda_column_ranges.csv")

# negative values in columns the datasheet does NOT call missing
for c in ["velocity_6h", "credit_risk_score"]:
    print(f"{c}: {(raw[c] < 0).sum()} negative values ({(raw[c] < 0).mean():.4%})")

# ---------- 2. per month ----------
per_month = raw.groupby("month").agg(n=(TARGET, "size"), fraud=(TARGET, "sum"))
per_month["fraud_rate"] = per_month["fraud"] / per_month["n"]
per_month["split"] = ["dev"] * 6 + ["test"] * 2
per_month.to_csv(TAB_DIR / "eda_per_month.csv")
print(per_month)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
colors = [BLUE] * 6 + [ORANGE] * 2
axes[0].bar(per_month.index, per_month["n"] / 1000, color=colors, width=0.7)
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Applications (thousands)")
axes[0].set_title("(a) Applications per month")
axes[1].bar(per_month.index, per_month["fraud_rate"] * 100, color=colors, width=0.7)
axes[1].axhline(overview["fraud_rate"] * 100, color=GREY, lw=1, ls="--")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Fraud rate (%)")
axes[1].set_title("(b) Fraud rate per month")
for ax in axes:
    ax.set_xticks(range(8))
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=BLUE, label="Development months (0-5)"),
                    Patch(color=ORANGE, label="Test months (6-7)"),
                    Line2D([0], [0], color=GREY, ls="--", lw=1,
                           label=f"Overall fraud rate ({overview['fraud_rate']*100:.2f}%)")],
           loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.88))
save(fig, FIG_DIR / "eda_months")

# ---------- 3. missing values ----------
df = clean(raw)
miss_cols = NEG1_MISSING + NEGATIVE_MISSING
miss_rows = []
for c in miss_cols:
    is_miss = df[c].isna()
    miss_rows.append({
        "feature": c,
        "missing_rate": is_miss.mean(),
        "fraud_rate_if_missing": df.loc[is_miss, TARGET].mean(),
        "fraud_rate_if_present": df.loc[~is_miss, TARGET].mean(),
    })
miss = pd.DataFrame(miss_rows).sort_values("missing_rate", ascending=False)
miss.to_csv(TAB_DIR / "eda_missing.csv", index=False)
print(miss)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1, 1.25]})
y = np.arange(len(miss))
short = [c.replace("_count", "").replace("_in_minutes", "").replace("_months", "_mo")
         for c in miss["feature"]]
axes[0].barh(y, miss["missing_rate"] * 100, color=BLUE, height=0.6)
axes[0].set_yticks(y, short)
axes[0].invert_yaxis()
axes[0].set_xlabel("Rows with value missing (%)")
axes[0].set_title("(a) Share of missing values")
for yi, v in zip(y, miss["missing_rate"] * 100):
    axes[0].text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=6.5, color=TEXT2)
axes[0].set_xlim(0, 100)
h = 0.36
axes[1].barh(y - h / 2, miss["fraud_rate_if_present"] * 100, height=h, color=BLUE, label="Value present")
axes[1].barh(y + h / 2, miss["fraud_rate_if_missing"] * 100, height=h, color=ORANGE, label="Value missing")
axes[1].set_yticks(y, [""] * len(y))
axes[1].invert_yaxis()
axes[1].set_xlabel("Fraud rate (%)")
axes[1].set_title("(b) Fraud rate when missing vs. present")
fig.legend(handles=[Patch(color=BLUE, label="Value present"), Patch(color=ORANGE, label="Value missing")],
           loc="upper center", ncol=2, bbox_to_anchor=(0.72, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9))
save(fig, FIG_DIR / "eda_missing")

# ---------- 4. age groups (protected attribute) ----------
age = raw.groupby(AGE_COL).agg(n=(TARGET, "size"), fraud=(TARGET, "sum"))
age["fraud_rate"] = age["fraud"] / age["n"]
age["share"] = age["n"] / len(raw)
age.to_csv(TAB_DIR / "eda_age.csv")
older = raw[AGE_COL] > AGE_CUTOFF
age_groups = {
    "over_50_share": float(older.mean()),
    "over_50_fraud_rate": float(raw.loc[older, TARGET].mean()),
    "50_or_under_fraud_rate": float(raw.loc[~older, TARGET].mean()),
}
with open(TAB_DIR / "eda_age_groups.json", "w") as f:
    json.dump(age_groups, f, indent=2)
print(age, age_groups)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
col = [ORANGE if a > AGE_CUTOFF else BLUE for a in age.index]
axes[0].bar(age.index, age["share"] * 100, width=6, color=col)
axes[0].set_xlabel("customer_age (decade)")
axes[0].set_ylabel("Share of applications (%)")
axes[0].set_title("(a) Age distribution")
axes[1].bar(age.index, age["fraud_rate"] * 100, width=6, color=col)
axes[1].set_xlabel("customer_age (decade)")
axes[1].set_ylabel("Fraud rate (%)")
axes[1].set_title("(b) Fraud rate by age")
for ax in axes:
    ax.set_xticks(age.index)
fig.legend(handles=[Patch(color=BLUE, label=f"Age \u2264 {AGE_CUTOFF}"),
                    Patch(color=ORANGE, label=f"Age > {AGE_CUTOFF} (fairness group)")],
           loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.88))
save(fig, FIG_DIR / "eda_age")

# ---------- 5. a few features by class ----------
show = ["name_email_similarity", "credit_risk_score", "current_address_months_count",
        "velocity_6h"]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
for ax, c in zip(axes, show):
    vals = df[c].dropna()
    bins = np.linspace(vals.quantile(0.001), vals.quantile(0.999), 40)
    for lab, colr, name in [(0, BLUE, "Legitimate"), (1, ORANGE, "Fraud")]:
        ax.hist(df.loc[df[TARGET] == lab, c].dropna(), bins=bins, density=True,
                histtype="step", lw=1.4, color=colr, label=name)
    ax.set_title(c.replace("_months_count", "_months"), fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("value")
axes[0].set_ylabel("Density")
fig.legend(handles=[Patch(color=BLUE, label="Legitimate"), Patch(color=ORANGE, label="Fraud")],
           loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.86))
save(fig, FIG_DIR / "eda_features_by_class")

# ---------- 6. drift: how much do feature means move per month ----------
num_feats = [c for g in FEATURE_GROUPS.values() for c in g if c not in CATEGORICAL]
dev = df[df["month"] <= 5]
z = (df.groupby("month")[num_feats].mean() - dev[num_feats].mean()) / dev[num_feats].std()
z.T.to_csv(TAB_DIR / "eda_monthly_shift_z.csv")
top = z.loc[[6, 7]].abs().max().sort_values(ascending=False).head(5).index.tolist()
print("features whose monthly mean moves most (in dev std units):", top)

fig, ax = plt.subplots(figsize=(3.4, 2.9))
for i, c in enumerate(top):
    ax.plot(z.index, z[c], marker="o", ms=3, color=[BLUE, ORANGE, "#00b894", "#f2a600", "#6a4ce2"][i], label=c)
ax.axvspan(5.5, 7.5, color="#f0f0f0", zorder=0)
ax.text(6.5, ax.get_ylim()[1] * 0.92, "test", ha="center", fontsize=6.5, color=TEXT2)
ax.axhline(0, color=GREY, lw=0.8)
ax.set_xticks(range(8))
ax.set_xlabel("Month")
ax.set_ylabel("Mean shift (std. of months 0-5)")
ax.set_title("Monthly drift of the most-shifting features")
ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
fig.tight_layout()
save(fig, FIG_DIR / "eda_drift")

# ---------- 7. fold sizes ----------
folds = pd.DataFrame(describe_folds(df))
folds.to_csv(TAB_DIR / "fold_sizes.csv", index=False)
print(folds)
print("EDA done.")
