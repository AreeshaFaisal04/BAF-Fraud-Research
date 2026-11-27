# colors and settings for all the plots
import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE = "#257de0"     # legitimate / series 1
ORANGE = "#f47230"   # fraud / series 2
AQUA = "#00b894"     # series 3
GREY = "#9a9a9a"
TEXT = "#191919"
TEXT2 = "#666666"

SERIES = [BLUE, ORANGE, AQUA, "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]


def use_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
        "lines.linewidth": 1.6,
    })


def save(fig, path_no_ext):
    fig.savefig(f"{path_no_ext}.pdf")
    fig.savefig(f"{path_no_ext}.png")
    plt.close(fig)
