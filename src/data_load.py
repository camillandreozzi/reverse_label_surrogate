"""Load X/Y HF and LF data (drop the id column in XHF), add the is_hf flag and target names."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

TARGETS = ["Kzz", "Rp", "Tint", "C/O", "[N/H]", "[O/H]", "[S/H]", "logg", "f"]


def target_slug(target):
    """File-system safe target name: 'C/O' -> 'C_O', '[N/H]' -> 'N_H'."""
    return target.replace("/", "_").strip("[]").replace("[", "").replace("]", "")


def load_hf():
    """HF spectra (97 x 195, raw eclipse depth) and targets."""
    x = pd.read_csv(DATA / "XHF_reverse.csv").drop(columns="wave")
    y = pd.read_csv(DATA / "YHF_reverse.csv")[TARGETS]
    return x.to_numpy(float), y


def load_lf10k():
    """LF spectra (10000 x 195, raw eclipse depth) and targets."""
    x = pd.read_csv(DATA / "XLF10k_reverse.csv")
    y = pd.read_csv(DATA / "YLF10k_reverse.csv")[TARGETS]
    return x.to_numpy(float), y


def wavelengths():
    return pd.read_csv(DATA / "XLF10k_reverse.csv", nrows=0).columns.astype(float).to_numpy()


def load_mf(n_lf=None, seed=0):
    """HF97 + LF10k stacked, HF rows first.

    Returns X (n x 195 raw depth), y (DataFrame of the 9 targets) and is_hf (1.0 HF, 0.0 LF).
    `n_lf` subsamples the LF rows (for tests and smoke runs).
    """
    x_hf, y_hf = load_hf()
    x_lf, y_lf = load_lf10k()
    if n_lf is not None and n_lf < len(x_lf):
        idx = np.sort(np.random.default_rng(seed).choice(len(x_lf), n_lf, replace=False))
        x_lf, y_lf = x_lf[idx], y_lf.iloc[idx]
    X = np.vstack([x_hf, x_lf])
    y = pd.concat([y_hf, y_lf], ignore_index=True)
    is_hf = np.r_[np.ones(len(x_hf)), np.zeros(len(x_lf))]
    return X, y, is_hf
