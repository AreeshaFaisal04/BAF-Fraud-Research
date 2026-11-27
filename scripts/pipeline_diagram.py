# draws the pipeline diagram for the methodology section
# python scripts/pipeline_diagram.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.config import FIG_DIR
from src.plotstyle import use_paper_style, save, BLUE, ORANGE, TEXT, TEXT2

use_paper_style()
FIG_DIR.mkdir(parents=True, exist_ok=True)

W, H = 7.16, 3.05           # IEEE two-column width in inches
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

BW, BH = 1.02, 0.50          # box size
FILL_KEY = "#e8f1fc"
FILL_TEST = "#fdeee6"


def box(x, y, text, fill="white", w=BW, h=BH, edge=TEXT2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec=edge, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.3, color=TEXT,
            linespacing=1.15)
    return (x, y, w, h)


def arrow(p, q, style="-|>", ls="-"):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle=style, color=TEXT2, lw=0.7, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=7))


def right(b):  return (b[0] + b[2], b[1] + b[3] / 2)
def left(b):   return (b[0], b[1] + b[3] / 2)
def bottom(b): return (b[0] + b[2] / 2, b[1])
def top(b):    return (b[0] + b[2] / 2, b[1] + b[3])


gap = (W - 0.16 - 6 * BW) / 5          # horizontal gap between 6 boxes
xs = [0.08 + i * (BW + gap) for i in range(6)]

# ---- row 1: data preparation ----
y1 = 2.42
b_data = box(xs[0], y1, "BAF Base dataset\n1M applications\nmonths 0-7", FILL_KEY)
b_clean = box(xs[1], y1, "Cleaning\nmissing codes -> NaN\ndrop 2 non-features")
b_split = box(xs[2], y1, "Time split by month\n0-5: development\n6-7: test (locked)")
arrow(right(b_data), left(b_clean))
arrow(right(b_clean), left(b_split))

# ---- row 2: development loop ----
y2 = 1.30
ax.add_patch(FancyBboxPatch((0.03, y2 - 0.13), W - 0.085, BH + 0.40,
                            boxstyle="round,pad=0.0,rounding_size=0.08",
                            fc="none", ec=BLUE, lw=0.8, ls=(0, (4, 2))))
ax.text(xs[0] + BW / 2 + 0.08, y2 + BH + 0.14, "Development (months 0-5): repeated for every model and every tuning trial",
        fontsize=6.3, color=BLUE, va="center")
b_folds = box(xs[0], y2, "5 forward-chaining\nfolds: train 0..k-1,\nvalidate month k")
b_samp = box(xs[1], y2, "Training rows only:\nall fraud + 25%\nof non-fraud")
b_pre = box(xs[2], y2, "Preprocessing fit on\ntrain rows: impute,\nmissing flags, one-hot")
b_model = box(xs[3], y2, "Model: Majority, LR,\nRF, LightGBM,\nCatBoost, MLP", FILL_KEY)
b_score = box(xs[4], y2, "Score validation\nmonth: Recall@5%FPR,\nROC-AUC, PR-AUC, F1")
b_tune = box(xs[5], y2, "Random search\n20 trials; best mean\nRecall@5%FPR")
for a, b in [(b_folds, b_samp), (b_samp, b_pre), (b_pre, b_model), (b_model, b_score), (b_score, b_tune)]:
    arrow(right(a), left(b))

# split -> folds (elbow)
sx, sy = bottom(b_split)
fx, fy = top(b_folds)
mid = (y1 + y2 + BH) / 2 + 0.02
ax.plot([sx, sx, fx], [sy, mid, mid], color=TEXT2, lw=0.7)
arrow((fx, mid), (fx, fy + 0.0))
ax.text(sx + 0.05, mid + 0.08, "months 0-5", fontsize=5.8, color=TEXT2)

# ---- row 3: final run ----
y3 = 0.12
b_fit = box(xs[2], y3, "Refit best config\non months 0-4")
b_thr = box(xs[3], y3, "Fix threshold at\n5% FPR on month 5")
b_test = box(xs[4], y3, "Evaluate months\n6 and 7 with the\nfixed threshold", FILL_TEST, edge=ORANGE)
b_out = box(xs[5], y3, "Analysis: RQ1-RQ3,\nfairness, ablation,\nerror analysis", FILL_KEY)
arrow(right(b_fit), left(b_thr))
arrow(right(b_thr), left(b_test))
arrow(right(b_test), left(b_out))

# tune -> refit (elbow)
tx, ty = bottom(b_tune)
rx, ry = top(b_fit)
mid2 = (y2 - 0.13 + y3 + BH) / 2
ax.plot([tx, tx, rx], [ty, mid2, mid2], color=TEXT2, lw=0.7)
arrow((rx, mid2), (rx, ry))
ax.text(tx - 0.05, mid2 + 0.07, "best hyperparameters", fontsize=5.8, color=TEXT2, ha="right")

# split -> test (dashed, the test months are only touched at the end)
ex = b_split[0] + b_split[2] + 0.06
ax.plot([right(b_split)[0], ex + gap * 0.0 + 0.30], [right(b_split)[1]] * 2, color=ORANGE, lw=0.7, ls=(0, (3, 2)))
tx2 = b_test[0] + b_test[2] * 0.80
ax.plot([ex + 0.30, W - 0.02, W - 0.02, tx2], [right(b_split)[1], right(b_split)[1], y3 + BH + 0.10, y3 + BH + 0.10],
        color=ORANGE, lw=0.7, ls=(0, (3, 2)))
ax.annotate("", xy=(tx2, y3 + BH), xytext=(tx2, y3 + BH + 0.10),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.7, mutation_scale=7, shrinkA=0, shrinkB=0))
ax.text(xs[3] + 0.05, right(b_split)[1] + 0.06, "months 6-7 held out until the final run",
        fontsize=5.8, color=ORANGE)

save(fig, FIG_DIR / "pipeline")
print("saved", FIG_DIR / "pipeline.pdf")
