"""Distributions of the 9 target parameters, HF vs LF.

Figures (results/eda/target_distributions/):
  targets.png              each target's own distribution: HF (97) vs LF10k (10000)
  spectra_all_bins.png     log10 depth pooled over all 195 bins: HF vs LF (97 matched) vs LF10k
  spectra_top_bins.png     per target, log10 depth pooled over only that target's top-k driver bins
  log_ratio_top_vs_all.png per target, log10(LF/HF) on its top-k bins vs on all bins: is LF
                           better or worse exactly where the target is decided?

Top bins come from eda/spectra_target_drivers.py output (--scores, --metric); if that CSV is
missing they are ranked by |Spearman ρ| on HF directly.

Usage (from repo root):
    python eda/target_distributions.py [--top-k 10] [--scores results/eda/spectra_drivers/scores_hf.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _eda_common import (MUTED, RESULTS, SERIES, TARGETS, load_hf, load_lf10k, load_lf_matched,
                         log_depth, out_dir, plt, set_style)

HF_C, LF_C, LF10K_C = SERIES  # colour follows the dataset in every figure
HIST = dict(bins=40, density=True, histtype="step", linewidth=1.6)


def top_bins(scores_path, metric, hf, y, k):
    """{target: array of bin indices}, best first."""
    if scores_path.exists():
        s = pd.read_csv(scores_path)
        print(f"Top bins from {scores_path.relative_to(RESULTS.parents[1])} ({metric})")
        return {t: s[s.target == t].nlargest(k, metric).bin.to_numpy() for t in TARGETS}
    print(f"{scores_path} not found; ranking bins by |Spearman ρ| on HF")
    rho = np.abs(spearmanr(hf, y[TARGETS].to_numpy()).statistic[:hf.shape[1], hf.shape[1]:])
    return {t: np.argsort(rho[:, i])[::-1][:k] for i, t in enumerate(TARGETS)}


def _legend(fig, handles, labels, title):
    fig.legend(handles, labels, loc="upper left", ncol=len(labels), bbox_to_anchor=(0.005, 0.965))
    fig.suptitle(title, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))


def plot_targets(y_hf, y_lf, path):
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for ax, t in zip(axes.flat, TARGETS):
        lo, hi = min(y_hf[t].min(), y_lf[t].min()), max(y_hf[t].max(), y_lf[t].max())
        bins = np.linspace(lo, hi, 30)
        ax.hist(y_lf[t], bins=bins, density=True, color=LF10K_C, alpha=0.25, label=f"LF10k (n={len(y_lf)})")
        ax.hist(y_hf[t], bins=bins, density=True, histtype="step", lw=1.8, color=HF_C, label=f"HF (n={len(y_hf)})")
        ax.set_title(t, loc="left")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    h, l = axes.flat[0].get_legend_handles_labels()
    _legend(fig, h, l, "Target distributions: HF vs LF10k (density)")
    fig.savefig(path)
    plt.close(fig)


def plot_spectra_all(hf, lf, lf10k, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for x, c, lab in [(hf, HF_C, "HF"), (lf, LF_C, "LF (97 matched)"), (lf10k, LF10K_C, "LF10k")]:
        ax.hist(x.ravel(), color=c, label=lab, **HIST)
    ax.set_xlabel("log10 eclipse depth (all 195 bins pooled)")
    ax.set_ylabel("density")
    ax.set_title("Spectral values aggregated over all wavelengths", loc="left")
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def plot_spectra_top(hf, lf, lf10k, wl, tops, k, path):
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5))
    for ax, t in zip(axes.flat, TARGETS):
        j = tops[t]
        for x, c, lab in [(hf, HF_C, "HF"), (lf, LF_C, "LF (97 matched)"), (lf10k, LF10K_C, "LF10k")]:
            ax.hist(x[:, j].ravel(), color=c, label=lab, **HIST)
        ax.set_title(f"{t}   bins {wl[j].min():.2f}–{wl[j].max():.2f} µm", loc="left")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    for ax in axes[-1]:
        ax.set_xlabel("log10 eclipse depth")
    h, l = axes.flat[0].get_legend_handles_labels()
    _legend(fig, h, l, f"Spectral values on each target's top-{k} driver bins")
    fig.savefig(path)
    plt.close(fig)


def plot_log_ratio(hf, lf, tops, k, path):
    r = lf - hf
    lim = np.quantile(np.abs(r), 0.995)
    bins = np.linspace(-lim, lim, 40)
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
    for ax, t in zip(axes.flat, TARGETS):
        top = r[:, tops[t]].ravel()
        ax.hist(r.ravel(), bins=bins, density=True, color=MUTED, alpha=0.3, label="all bins")
        ax.hist(top, bins=bins, density=True, histtype="step", lw=1.8, color=HF_C, label=f"top-{k} bins")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{t}   RMS top {np.sqrt((top ** 2).mean()):.3f} vs all {np.sqrt((r ** 2).mean()):.3f} dex",
                     loc="left")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    for ax in axes[-1]:
        ax.set_xlabel("log10(LF / HF)  (dex)")
    h, l = axes.flat[0].get_legend_handles_labels()
    _legend(fig, h, l, "LF − HF discrepancy (97 matched pairs): each target's top bins vs all bins")
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--scores", type=Path, default=RESULTS / "spectra_drivers" / "scores_hf.csv")
    p.add_argument("--metric", choices=["abs_spearman", "mi", "perm"], default="abs_spearman")
    a = p.parse_args()

    set_style()
    hf_lin, wl, y_hf = load_hf()
    lf10k_lin, _, y_lf = load_lf10k()
    hf, lf, lf10k = log_depth(hf_lin), log_depth(load_lf_matched()), log_depth(lf10k_lin)
    tops = top_bins(a.scores, a.metric, hf, y_hf, a.top_k)
    d = out_dir("target_distributions")

    plot_targets(y_hf, y_lf, d / "targets.png")
    plot_spectra_all(hf, lf, lf10k, d / "spectra_all_bins.png")
    plot_spectra_top(hf, lf, lf10k, wl, tops, a.top_k, d / f"spectra_top{a.top_k}_bins.png")
    plot_log_ratio(hf, lf, tops, a.top_k, d / f"log_ratio_top{a.top_k}_vs_all.png")
    pd.DataFrame([{"target": t, "rank": i, "bin": b, "wavelength": wl[b]}
                  for t in TARGETS for i, b in enumerate(tops[t])]).to_csv(d / "top_bins_used.csv", index=False)
    print(f"Saved to {d}")


if __name__ == "__main__":
    main()
