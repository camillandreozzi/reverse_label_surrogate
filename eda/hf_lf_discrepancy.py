"""HF vs LF spectra for matched samples.

How well LF tracks HF, as a function of wavelength and of the individual spectrum.
All statistics use log10 eclipse depth; XLF_reverse row i is the LF run of XHF_reverse row i.

Per wavelength bin (across the 97 samples):
  pearson / spearman  corr(HF, LF)
  ar1_slope           OLS slope of HF on LF: the AR(1) scale rho the MF model assumes
                      (bootstrap 90% CI), with the residual std of that fit
  rel_diff_*          (LF - HF) / HF in linear depth: median and 10-90% band
Per sample (across the 195 bins):
  rmse                RMS of log10(LF / HF)
  anomaly_corr        corr of HF and LF *anomalies* (sample minus the mean spectrum): does LF
                      reproduce how this spectrum differs from the average one?

Usage (from repo root):
    python eda/hf_lf_discrepancy.py [--n-boot 1000]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _eda_common import (CMAP_DIV, INK_2, MUTED, SERIES, TARGETS, load_hf, load_lf_matched,
                         log_depth, log_wavelength_axis, out_dir, plt, set_style)


def per_bin_stats(hf, lf, hf_lin, lf_lin, n_boot, seed):
    n, p = hf.shape
    pearson = np.array([np.corrcoef(hf[:, j], lf[:, j])[0, 1] for j in range(p)])
    spear = np.array([spearmanr(hf[:, j], lf[:, j]).statistic for j in range(p)])

    def slope(h, l):
        lc = l - l.mean(axis=-2, keepdims=True)
        hc = h - h.mean(axis=-2, keepdims=True)
        return (lc * hc).sum(axis=-2) / (lc ** 2).sum(axis=-2)

    b = slope(hf, lf)
    resid = hf - (hf.mean(0) + b * (lf - lf.mean(0)))
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    boot = slope(hf[idx], lf[idx])  # (n_boot, p)

    rel = (lf_lin - hf_lin) / hf_lin
    return pd.DataFrame({
        "pearson": pearson, "spearman": spear,
        "ar1_slope": b, "ar1_lo": np.quantile(boot, 0.05, 0), "ar1_hi": np.quantile(boot, 0.95, 0),
        "ar1_resid_std": resid.std(0, ddof=2),
        "rel_diff_median": np.median(rel, 0),
        "rel_diff_q10": np.quantile(rel, 0.1, 0), "rel_diff_q90": np.quantile(rel, 0.9, 0),
    })


def per_sample_stats(hf, lf, y):
    d = lf - hf
    ha, la = hf - hf.mean(0), lf - lf.mean(0)
    anomaly_corr = np.array([np.corrcoef(ha[i], la[i])[0, 1] for i in range(len(hf))])
    out = pd.DataFrame({"rmse": np.sqrt((d ** 2).mean(1)), "mean_log_ratio": d.mean(1),
                        "anomaly_corr": anomaly_corr})
    return pd.concat([out, y[TARGETS].reset_index(drop=True)], axis=1)


def plot_corr_vs_wavelength(b, wl, hf, path):
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.08})
    ax0.plot(wl, hf.mean(0), color=MUTED, lw=1.2)
    ax0.set_ylabel("mean HF\nlog10 depth")
    ax0.set_title("Correlation of HF and LF across the 97 matched samples, per wavelength bin", loc="left")
    ax.plot(wl, b.pearson, color=SERIES[0], label="Pearson")
    ax.plot(wl, b.spearman, color=SERIES[1], label="Spearman")
    ax.set_ylabel("corr(HF, LF)")
    ax.set_xlabel("Wavelength (µm)")
    ax.legend(loc="lower left", ncol=2)
    log_wavelength_axis(ax)
    fig.savefig(path)
    plt.close(fig)


def plot_ar1(b, wl, path):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})
    ax.fill_between(wl, b.ar1_lo, b.ar1_hi, color=SERIES[0], alpha=0.2, lw=0, label="bootstrap 90% CI")
    ax.plot(wl, b.ar1_slope, color=SERIES[0], label="OLS slope ρ(λ)")
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_ylabel("ρ(λ)")
    ax.set_title("AR(1) scale: HF ≈ ρ(λ)·LF + δ(λ)  (log10 depth). A constant ρ would be a flat line.", loc="left")
    ax.legend(loc="lower left", ncol=2)
    ax2.plot(wl, b.ar1_resid_std, color=SERIES[1])
    ax2.set_ylabel("residual std\n(dex)")
    ax2.set_xlabel("Wavelength (µm)")
    log_wavelength_axis(ax2)
    fig.savefig(path)
    plt.close(fig)


def plot_rel_diff(b, wl, path):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.fill_between(wl, 100 * b.rel_diff_q10, 100 * b.rel_diff_q90, color=SERIES[0], alpha=0.2, lw=0,
                    label="10–90% of samples")
    ax.plot(wl, 100 * b.rel_diff_median, color=SERIES[0], label="median")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("(LF − HF) / HF  (%)")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_title("Relative LF discrepancy per wavelength bin", loc="left")
    ax.legend(loc="upper left", ncol=2)
    log_wavelength_axis(ax)
    fig.savefig(path)
    plt.close(fig)


def plot_bin_scatter(b, wl, hf, lf, path):
    order = np.argsort(b.pearson.to_numpy())
    mid = len(order) // 2
    picks = [("best", order[-1]), ("best", order[-2]), ("median", order[mid]),
             ("median", order[mid + 1]), ("worst", order[1]), ("worst", order[0])]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, (lab, j) in zip(axes.flat, picks):
        lo, hi = min(hf[:, j].min(), lf[:, j].min()), max(hf[:, j].max(), lf[:, j].max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
        ax.scatter(lf[:, j], hf[:, j], s=16, color=SERIES[0], edgecolor="white", lw=0.5)
        ax.set_title(f"{lab}: {wl[j]:.3f} µm   r = {b.pearson[j]:.2f}", loc="left")
        ax.set_xlabel("LF log10 depth")
        ax.set_ylabel("HF log10 depth")
    fig.suptitle("HF vs LF at the best, median and worst correlated bins (dashed: y = x)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sample_vs_targets(s, metric, ylabel, path):
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharey=True)
    for ax, t in zip(axes.flat, TARGETS):
        ax.scatter(s[t], s[metric], s=16, color=SERIES[0], edgecolor="white", lw=0.5)
        rho = spearmanr(s[t], s[metric]).statistic
        ax.set_title(f"{t}   Spearman ρ = {rho:+.2f}", loc="left")
        ax.set_xlabel(t)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(f"Per-spectrum HF/LF agreement vs parameters: {ylabel}", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ratio_heatmap(hf, lf, wl, s, sort_by, path):
    order = np.argsort(s[sort_by].to_numpy())
    ratio = (lf - hf)[order]
    lim = np.quantile(np.abs(ratio), 0.99)
    fig, ax = plt.subplots(figsize=(10, 5))
    m = ax.pcolormesh(wl, np.arange(len(hf)), ratio, cmap=CMAP_DIV, vmin=-lim, vmax=lim, shading="nearest")
    ax.grid(False)
    ax.set_ylabel(f"sample, sorted by {sort_by} (low → high)")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_title(f"log10(LF / HF) per sample and bin  (red: LF deeper, blue: LF shallower)", loc="left")
    log_wavelength_axis(ax)
    fig.colorbar(m, ax=ax, label="log10(LF / HF)  (dex)", fraction=0.03, pad=0.01)
    fig.savefig(path)
    plt.close(fig)


def plot_extreme_samples(hf, lf, wl, s, path):
    order = s.rmse.sort_values().index.to_numpy()
    picks = [("best", i) for i in order[:3]] + [("worst", i) for i in order[::-1][:3]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for ax, (lab, i) in zip(axes.flat, picks):
        ax.plot(wl, hf[i], color=SERIES[0], label="HF")
        ax.plot(wl, lf[i], color=SERIES[1], label="LF")
        ax.set_title(f"{lab}: sample {i}   RMSE = {s.rmse[i]:.3f} dex", loc="left")
        log_wavelength_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("log10 depth")
    for ax in axes[-1]:
        ax.set_xlabel("Wavelength (µm)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, bbox_to_anchor=(0.005, 0.955))
    fig.suptitle("Spectra with the best and worst HF/LF agreement", x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    set_style()
    hf_lin, wl, y = load_hf()
    lf_lin = load_lf_matched()
    hf, lf = log_depth(hf_lin), log_depth(lf_lin)
    d = out_dir("hf_lf")

    b = per_bin_stats(hf, lf, hf_lin, lf_lin, a.n_boot, a.seed)
    b.insert(0, "wavelength", wl)
    s = per_sample_stats(hf, lf, y)
    b.to_csv(d / "per_bin.csv", index=False)
    s.to_csv(d / "per_sample.csv", index_label="sample")

    # Which parameter the discrepancy depends on most -> used to sort the heatmap.
    rho = {t: spearmanr(s[t], s.rmse).statistic for t in TARGETS}
    sort_by = max(rho, key=lambda t: abs(rho[t]))
    print(f"Per-bin Pearson corr: median {b.pearson.median():.2f}, min {b.pearson.min():.2f} "
          f"at {wl[b.pearson.idxmin()]:.3f} µm")
    print(f"AR(1) slope: median {b.ar1_slope.median():.2f}, range {b.ar1_slope.min():.2f}–{b.ar1_slope.max():.2f}")
    print("Spearman(per-sample RMSE, target): " + ", ".join(f"{t} {r:+.2f}" for t, r in rho.items()))

    plot_corr_vs_wavelength(b, wl, hf, d / "corr_vs_wavelength.png")
    plot_ar1(b, wl, d / "ar1_slope_vs_wavelength.png")
    plot_rel_diff(b, wl, d / "rel_diff_vs_wavelength.png")
    plot_bin_scatter(b, wl, hf, lf, d / "scatter_selected_bins.png")
    plot_sample_vs_targets(s, "rmse", "RMSE log10(LF/HF) (dex)", d / "sample_rmse_vs_targets.png")
    plot_sample_vs_targets(s, "anomaly_corr", "anomaly corr(HF, LF)", d / "sample_anomaly_corr_vs_targets.png")
    plot_ratio_heatmap(hf, lf, wl, s, sort_by, d / "log_ratio_heatmap.png")
    plot_extreme_samples(hf, lf, wl, s, d / "extreme_samples.png")
    print(f"Saved to {d}")


if __name__ == "__main__":
    main()
