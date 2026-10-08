"""Predicted vs true plots per target (nested LOO over the 97 HF samples).

Figures (results/<run>/figures/, --run independent | pca):
  loo_pred_vs_true_<variant>.png   3x3 grid, one panel per output: LOO prediction vs truth with 90%
                                   intervals; grey band = range of the output in the LF10k design.
                                   SF predictions whose variance came out negative are hollow, no interval.
  loo_calibration.png              empirical vs nominal coverage of the central intervals, per output
                                   (valid variances only); on the diagonal = well calibrated.
  loo_standardised_residuals.png   (y - mean) / sd per output and variant vs a standard normal

Usage (from repo root):
    python modelling/plot/plot_pred_vs_true.py
"""
import numpy as np
from scipy.stats import norm

from _plot_common import (COLOR, INK_2, LABEL, MUTED, TARGETS, VARIANTS, load_loo, out_dir, plt, set_style,
                          suptitle, variant_legend)
from src.data_load import load_lf10k

Z90 = norm.ppf(0.95)


def plot_pred_vs_true(p, design, variant, path):
    fig, axes = plt.subplots(3, 3, figsize=(11, 10.5))
    c = COLOR[variant]
    for ax, t in zip(axes.flat, TARGETS):
        g = p[(p.target == t) & (p.variant == variant)]
        lo, hi = design[t]
        ax.axhspan(lo, hi, color=MUTED, alpha=0.10, lw=0)
        ok, bad = g[g.valid_var], g[~g.valid_var]
        ax.errorbar(ok.y, ok["mean"], yerr=Z90 * ok.sd, fmt="none", ecolor=c, alpha=0.35, lw=0.9)
        ax.scatter(ok.y, ok["mean"], s=16, color=c, edgecolor="white", lw=0.5, zorder=3)
        if len(bad):
            ax.scatter(bad.y, bad["mean"], s=18, facecolor="none", edgecolor=c, lw=1.0, zorder=3)
        ylo, yhi = min(g.y.min(), g["mean"].min()), max(g.y.max(), g["mean"].max())
        pad = 0.05 * (yhi - ylo)
        ax.plot([ylo - pad, yhi + pad], [ylo - pad, yhi + pad], color=INK_2, lw=0.9, ls="--")
        r2 = 1 - (g.err ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum()
        extra = f", {len(bad)} invalid var" if len(bad) else ""
        ax.set_title(f"{t}   R² = {r2:.2f}{extra}", loc="left")
        ax.set_xlabel(f"true {t}")
        ax.set_ylabel(f"LOO prediction")
    suptitle(fig, f"LOO predictions vs truth, {LABEL[variant]}",
             "Bars: 90% predictive intervals. Dashed: y = x. Grey band: range of the output in the LF10k design."
             + ("  Hollow: variance came out negative (no interval)." if variant == "sf" else ""))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(p, path):
    levels = np.linspace(0.05, 0.99, 40)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, TARGETS):
        ax.plot([0, 1], [0, 1], color=INK_2, lw=0.9, ls="--")
        for v in VARIANTS:
            g = p[(p.target == t) & (p.variant == v) & p.valid_var]
            z = np.abs(g.err / g.sd).to_numpy()
            cov = [(z <= norm.ppf(0.5 + lv / 2)).mean() for lv in levels]
            ax.plot(levels, cov, color=COLOR[v], label=f"{v.upper()} (n={len(g)})")
        ax.set_title(t, loc="left")
        ax.legend(loc="upper left", fontsize=7.5)
        ax.set_aspect("equal")
    for ax in axes[-1]:
        ax.set_xlabel("nominal coverage")
    for ax in axes[:, 0]:
        ax.set_ylabel("empirical coverage")
    suptitle(fig, "Calibration of the LOO predictive intervals (valid variances only)",
             "Below the diagonal: intervals too narrow (overconfident); above: too wide.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_std_residuals(p, path):
    x = np.linspace(-4, 4, 200)
    bins = np.linspace(-4, 4, 25)
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True)
    for ax, t in zip(axes.flat, TARGETS):
        ax.plot(x, norm.pdf(x), color=INK_2, lw=1.0, ls="--")
        for v in VARIANTS:
            g = p[(p.target == t) & (p.variant == v) & p.valid_var]
            z = (g.err / g.sd).clip(-4, 4)
            ax.hist(z, bins=bins, density=True, histtype="step", lw=1.6, color=COLOR[v])
        ax.set_title(t, loc="left")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    for ax in axes[-1]:
        ax.set_xlabel("(prediction − truth) / predictive sd   (clipped at ±4)")
    suptitle(fig, "Standardised LOO residuals vs a standard normal (dashed), valid variances only",
             "Wider than the dashed curve: overconfident; narrower: underconfident; shifted: biased.")
    variant_legend(fig, y=0.945)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path)
    plt.close(fig)


def main():
    set_style()
    p, _ = load_loo()
    _, y_lf = load_lf10k()
    design = {t: (y_lf[t].min(), y_lf[t].max()) for t in TARGETS}
    d = out_dir()
    for v in VARIANTS:
        plot_pred_vs_true(p, design, v, d / f"loo_pred_vs_true_{v}.png")
    plot_calibration(p, d / "loo_calibration.png")
    plot_std_residuals(p, d / "loo_standardised_residuals.png")
    print(f"Saved to {d}")


if __name__ == "__main__":
    main()
