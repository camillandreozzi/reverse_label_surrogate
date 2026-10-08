"""Shared loading and styling for the result plots (MF blue, SF orange in every figure).

`--run NAME` (default independent) selects results/<run>/ for both inputs and figures.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_load import RESULTS, TARGETS  # noqa: E402

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--run", default="independent", help="results/<run>/ to plot (independent | pca)")
RUN = _p.parse_known_args()[0].run
RUN_NOTE = {"pca": "GP on the first 20 PCA scores, trees on all 195 bins"}.get(RUN)
IND = RESULTS / RUN
LOO = IND / "cv_loo"
FIG = IND / "figures"

VARIANTS = ["mf", "sf"]
LABEL = {"mf": "MF (HF + LF10k, AR(1) GP)", "sf": "SF (HF only)"}
COLOR = {"mf": "#2a78d6", "sf": "#eb6834"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Prior ranges of the parameters (from the LF10k design), used as reference bands
PRIOR = {"Kzz": (8, 11.5), "Rp": None, "Tint": (150, 450), "C/O": (0.1, 1.5), "[N/H]": (-1, 2), "[O/H]": (-1, 2),
         "[S/H]": (-1, 2), "logg": None, "f": (0.25, 0.67)}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "legend.frameon": False,
        "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def out_dir():
    FIG.mkdir(parents=True, exist_ok=True)
    return FIG


def load_loo():
    """LOO predictions (with error, sd and a valid-variance flag) and per-fold parameters."""
    p = pd.read_csv(LOO / "predictions.csv")
    q = pd.read_csv(LOO / "params.csv")
    p["err"] = p["mean"] - p["y"]
    p["valid_var"] = p["var"] > 0
    p["sd"] = np.sqrt(p["var"].where(p.valid_var))
    p["target"] = pd.Categorical(p["target"], TARGETS, ordered=True)
    q["target"] = pd.Categorical(q["target"], TARGETS, ordered=True)
    return p, q


def suptitle(fig, title, subtitle=None):
    if RUN_NOTE:
        title = f"{title}  [{RUN_NOTE}]"
    fig.suptitle(title, x=0.01, ha="left", y=1.0, fontsize=11)
    if subtitle:
        fig.text(0.01, 0.965, subtitle, ha="left", va="top", color=INK_2, fontsize=8.5)


def variant_legend(fig, y=0.93):
    handles = [plt.Line2D([], [], marker="s", ls="", color=COLOR[v], markersize=8, label=LABEL[v]) for v in VARIANTS]
    fig.legend(handles=handles, loc="upper left", ncol=2, bbox_to_anchor=(0.005, y))
