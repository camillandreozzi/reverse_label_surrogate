"""Shared helpers for the EDA scripts: data loading, output paths and plot styling."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "eda"

TARGETS = ["Kzz", "Rp", "Tint", "C/O", "[N/H]", "[O/H]", "[S/H]", "logg", "f"]

# Palette (validated reference instance, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua: all-pairs safe
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RED_ARM = ["#fbd5d4", "#f2a09f", "#e34948", "#b52a2a", "#7d1b1b"]
BLUE_ARM = ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQ)
CMAP_DIV = LinearSegmentedColormap.from_list("blue_red", BLUE_ARM[::-1] + ["#f0efec"] + RED_ARM)


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


def out_dir(name):
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_hf():
    """HF spectra (97 x 195, id column dropped), wavelengths (um) and targets."""
    x = pd.read_csv(DATA / "XHF_reverse.csv").drop(columns="wave")
    y = pd.read_csv(DATA / "YHF_reverse.csv")
    return x.to_numpy(float), x.columns.astype(float).to_numpy(), y


def load_lf_matched():
    """LF spectra row-matched to the 97 HF samples."""
    return pd.read_csv(DATA / "XLF_reverse.csv").to_numpy(float)


def load_lf10k():
    x = pd.read_csv(DATA / "XLF10k_reverse.csv")
    y = pd.read_csv(DATA / "YLF10k_reverse.csv")
    return x.to_numpy(float), x.columns.astype(float).to_numpy(), y


def log_depth(x):
    """log10 eclipse depth; depths span ~3 decades so correlations are computed in log space."""
    return np.log10(np.clip(x, 1e-12, None))


def log_wavelength_axis(ax):
    # Bins are dense below ~5 um and coarse above, so a log axis spreads them evenly.
    ax.set_xscale("log")
    ticks = [2.5, 3, 4, 5, 6, 8, 10, 12]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
