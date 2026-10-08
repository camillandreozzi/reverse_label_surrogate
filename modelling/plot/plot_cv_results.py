"""Plot CV metrics per target and variant (nested LOO over the 97 HF samples).

Figures (results/<run>/figures/, --run independent | pca):
  loo_metrics.png         R², NRMSE, 90% coverage and NLPD per output, MF vs SF
  loo_paired_errors.png   per-fold |error| of MF vs SF for each output (below the diagonal = MF better)
  loo_gp_parameters.png   per-fold GP estimates: MF rho and LF length scale, SF length scale, SF breakdowns
  loo_tuning.png          tuned boosting rounds and learning rate per output
Uncertainty metrics (coverage, NLPD) use only predictions with a valid (positive) variance; the number
of SF predictions whose variance came out negative is shown separately.

Usage (from repo root):
    python modelling/plot/plot_cv_results.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from _plot_common import (COLOR, INK_2, MUTED, TARGETS, VARIANTS, load_loo, out_dir, plt, set_style,
                          suptitle, variant_legend)


def metrics_table(p):
    rows = []
    for (t, v), g in p.groupby(["target", "variant"], observed=True):
        ok = g[g.valid_var]
        z = norm.ppf(0.95)
        rows.append({
            "target": t, "variant": v, "n": len(g), "n_valid_var": len(ok),
            "r2": 1 - (g.err ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum(),
            "nrmse": np.sqrt((g.err ** 2).mean()) / g.y.std(ddof=0),
            "coverage_90": (np.abs(ok.err) <= z * ok.sd).mean() if len(ok) else np.nan,
            "nlpd": (0.5 * np.log(2 * np.pi * ok["var"]) + 0.5 * ok.err ** 2 / ok["var"]).mean() if len(ok) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_metrics(m, path):
    panels = [("r2", "R²  (1 = perfect, 0 = predicting the mean)", 0.0),
              ("nrmse", "NRMSE  (RMSE / std of the output)", 1.0),
              ("coverage_90", "90% interval coverage  (valid variances only)", 0.9),
              ("nlpd", "NLPD  (lower is better; valid variances only)", None)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=True)
    y = np.arange(len(TARGETS))
    for ax, (col, title, ref) in zip(axes, panels):
        for off, v in zip([-0.19, 0.19], VARIANTS):
            vals = m[m.variant == v].set_index("target").reindex(TARGETS)[col]
            ax.barh(y + off, vals, height=0.36, color=COLOR[v])
        if ref is not None:
            ax.axvline(ref, color=MUTED, lw=0.9, ls="--" if ref else "-")
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, TARGETS)
    axes[0].invert_yaxis()
    axes[0].set_xlim(left=min(-1.5, m.r2.min() - 0.1))
    nneg = m[m.variant == "sf"].set_index("target").reindex(TARGETS)
    for i, t in enumerate(TARGETS):
        k = int(nneg.loc[t, "n"] - nneg.loc[t, "n_valid_var"])
        if k:
            axes[2].text(1.01, i + 0.19, f"SF: {k} invalid", va="center", fontsize=7, color=INK_2,
                         transform=axes[2].get_yaxis_transform())
    suptitle(fig, "Nested leave-one-out over the 97 HF samples: accuracy and calibration per output",
             "Each output modelled independently; 97 held-out predictions per output and variant.")
    variant_legend(fig, y=0.925)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path)
    plt.close(fig)


def plot_paired(p, path):
    w = p.pivot_table(index=["target", "sample"], columns="variant", values="err", observed=True).abs().reset_index()
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for ax, t in zip(axes.flat, TARGETS):
        g = w[w.target == t]
        lo = max(min(g.mf.min(), g.sf.min()), 1e-6)
        hi = max(g.mf.max(), g.sf.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls="--")
        better = g.mf < g.sf
        ax.scatter(g.sf[better], g.mf[better], s=16, color=COLOR["mf"], edgecolor="white", lw=0.5)
        ax.scatter(g.sf[~better], g.mf[~better], s=16, color=COLOR["sf"], edgecolor="white", lw=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{t}   MF closer in {better.mean():.0%} of folds", loc="left")
        ax.set_xlabel("SF |error|")
        ax.set_ylabel("MF |error|")
    suptitle(fig, "Per-fold absolute errors, MF vs SF (each point = one held-out HF sample)",
             "Blue: MF closer to the truth; orange: SF closer. Dashed: equal error. Log scales.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def _strip(ax, data, color, ref=None, log=False):
    y = np.arange(len(TARGETS))
    rng = np.random.default_rng(0)
    for i, t in enumerate(TARGETS):
        v = data.get(t, pd.Series(dtype=float)).dropna()
        v = v[v > 0] if log else v
        if not len(v):
            continue
        vv = np.log10(v) if log else v
        ax.scatter(vv, i + rng.uniform(-0.18, 0.18, len(vv)), s=7, color=color, alpha=0.45, lw=0)
        ax.plot([np.median(vv)] * 2, [i - 0.3, i + 0.3], color="black", lw=1.6)
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=0.9, ls="--")
    ax.set_yticks(y, TARGETS)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


def plot_gp_parameters(p, q, path):
    mf, sf = q[q.variant == "mf"], q[q.variant == "sf"]
    by = lambda df, col: {t: g[col] for t, g in df.groupby("target", observed=True)}  # noqa: E731
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=True)
    _strip(axes[0], by(mf, "rho"), COLOR["mf"], ref=1.0)
    lo, hi = -3, 3.5
    n_out = int(((mf.rho < lo) | (mf.rho > hi)).sum())
    axes[0].set_xlim(lo, hi)
    axes[0].set_title("MF: AR(1) scale ρ per fold\n(1 = LF transfers one-to-one to HF)"
                      + (f"; {n_out} fold{'s' if n_out > 1 else ''} off-axis" if n_out else ""), loc="left", fontsize=9)
    _strip(axes[1], by(mf, "range_L"), COLOR["mf"], ref=np.log10(17), log=True)
    axes[1].set_title("MF: LF GP length scale, log10\n(dashed: typical distance between spectra ≈ 17)", loc="left", fontsize=9)
    _strip(axes[2], by(sf, "range"), COLOR["sf"], ref=np.log10(17), log=True)
    axes[2].set_title("SF: GP length scale, log10\n(dashed: typical distance between spectra ≈ 17)", loc="left", fontsize=9)
    neg = p[(p.variant == "sf") & ~p.valid_var].groupby("target", observed=True).size().reindex(TARGETS, fill_value=0)
    axes[3].barh(np.arange(len(TARGETS)), neg.values, color=COLOR["sf"], height=0.5)
    for i, k in enumerate(neg.values):
        axes[3].text(k, i, f" {k}", va="center", fontsize=8, color=INK_2)
    axes[3].set_title("SF: predictions with negative variance\n(numerical breakdown, out of 97)", loc="left", fontsize=9)
    axes[3].grid(axis="y", visible=False)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    suptitle(fig, "GP parameters estimated in each LOO fold (black bar = median)",
             "Length scales far above the spread of the data mean the GP behaves like a global linear extrapolator.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def plot_tuning(q, path):
    mf = q[q.variant == "mf"]
    by = lambda col: {t: g[col] for t, g in mf.groupby("target", observed=True)}  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    _strip(axes[0], by("num_boost_round"), INK_2)
    axes[0].set_title("Boosting rounds (shared by MF and SF)", loc="left", fontsize=9)
    _strip(axes[1], by("tree_learning_rate"), INK_2)
    axes[1].set_title("Learning rate", loc="left", fontsize=9)
    _strip(axes[2], {t: v / 3600 for t, v in by("tune_seconds").items()}, INK_2)
    axes[2].set_title("Tuning time per fold (hours)", loc="left", fontsize=9)
    suptitle(fig, "Tuned tree hyperparameters per LOO fold (TPE, tuned with the MF model; black bar = median)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def main():
    set_style()
    p, q = load_loo()
    d = out_dir()
    m = metrics_table(p)
    m.to_csv(d / "loo_metrics_valid_variance.csv", index=False)
    plot_metrics(m, d / "loo_metrics.png")
    plot_paired(p, d / "loo_paired_errors.png")
    plot_gp_parameters(p, q, d / "loo_gp_parameters.png")
    plot_tuning(q, d / "loo_tuning.png")
    print(m.round(3).to_string(index=False))
    print(f"Saved to {d}")


if __name__ == "__main__":
    main()
