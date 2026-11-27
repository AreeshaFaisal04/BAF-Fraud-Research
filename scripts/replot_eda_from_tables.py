"""
Re-draws the EDA figures from the summary tables that 01_eda.py saved
(outputs/tables/eda_*.csv), so the figures can be remade without the 200 MB data file.

    python scripts/replot_eda_from_tables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.config import FIG_DIR, TAB_DIR, AGE_CUTOFF
from src.plotstyle import use_paper_style, save, BLUE, ORANGE, GREY, TEXT2

FIG_DIR.mkdir(parents=True, exist_ok=True)
use_paper_style()
overview = json.load(open(TAB_DIR / "eda_overview.json"))

# months
per_month = pd.read_csv(TAB_DIR / "eda_per_month.csv", index_col="month")
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
fig.legend(handles=[Patch(color=BLUE, label="Development months (0-5)"),
                    Patch(color=ORANGE, label="Test months (6-7)"),
                    Line2D([0], [0], color=GREY, ls="--", lw=1,
                           label=f"Overall fraud rate ({overview['fraud_rate']*100:.2f}%)")],
           loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.88))
save(fig, FIG_DIR / "eda_months")

# missing values
miss = pd.read_csv(TAB_DIR / "eda_missing.csv").sort_values("missing_rate", ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1, 1.25]})
y = np.arange(len(miss))
short = [c.replace("_count", "").replace("_in_minutes", "").replace("_months", "_mo") for c in miss["feature"]]
axes[0].barh(y, miss["missing_rate"] * 100, color=BLUE, height=0.6)
axes[0].set_yticks(y, short)
axes[0].invert_yaxis()
axes[0].set_xlabel("Rows with value missing (%)")
axes[0].set_title("(a) Share of missing values")
for yi, v in zip(y, miss["missing_rate"] * 100):
    axes[0].text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=6.5, color=TEXT2)
axes[0].set_xlim(0, 100)
h = 0.36
axes[1].barh(y - h / 2, miss["fraud_rate_if_present"] * 100, height=h, color=BLUE)
axes[1].barh(y + h / 2, miss["fraud_rate_if_missing"] * 100, height=h, color=ORANGE)
axes[1].set_yticks(y, [""] * len(y))
axes[1].invert_yaxis()
axes[1].set_xlabel("Fraud rate (%)")
axes[1].set_title("(b) Fraud rate when missing vs. present")
fig.legend(handles=[Patch(color=BLUE, label="Value present"), Patch(color=ORANGE, label="Value missing")],
           loc="upper center", ncol=2, bbox_to_anchor=(0.72, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9))
save(fig, FIG_DIR / "eda_missing")

# age
age = pd.read_csv(TAB_DIR / "eda_age.csv", index_col="customer_age")
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
fig.legend(handles=[Patch(color=BLUE, label=f"Age ≤ {AGE_CUTOFF}"),
                    Patch(color=ORANGE, label=f"Age > {AGE_CUTOFF} (fairness group)")],
           loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.88))
save(fig, FIG_DIR / "eda_age")

# drift
z = pd.read_csv(TAB_DIR / "eda_monthly_shift_z.csv", index_col=0).T
z.index = z.index.astype(int)
top = z.loc[[6, 7]].abs().max().sort_values(ascending=False).head(5).index.tolist()
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
print("figures redrawn from", TAB_DIR)
