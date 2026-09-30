"""Which spectral bins drive each of the 9 targets, and with what strength.

Three complementary per-(target, wavelength) scores:
  spearman  signed monotone association (univariate)
  mi        mutual information, captures non-linear dependence (univariate)
  perm      random-forest permutation importance on a held-out split (multivariate:
            credit given to a bin *after* the other bins are available). Neighbouring bins
            are collinear and split credit, so this curve is smoothed.

Usage (from repo root):
    python eda/spectra_target_drivers.py                                 # HF, 97 samples
    python eda/spectra_target_drivers.py --normalise                     # shape only
    python eda/spectra_target_drivers.py --fidelity lf10k --n-sub 3000   # LF, more power
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from _eda_common import (CMAP_DIV, CMAP_SEQ, INK_2, MUTED, SERIES, TARGETS, load_hf,
                         load_lf10k, log_depth, log_wavelength_axis, out_dir, plt, set_style)

METRIC_LABELS = {"abs_spearman": "|Spearman ρ|", "mi": "Mutual information",
                 "perm": "RF permutation importance"}


def load_xy(fidelity, n_sub, normalise, seed):
    if fidelity == "hf":
        x, wl, y = load_hf()
    else:
        x, wl, y = load_lf10k()
        if n_sub and n_sub < len(x):
            idx = np.random.default_rng(seed).choice(len(x), n_sub, replace=False)
            x, y = x[idx], y.iloc[idx].reset_index(drop=True)
    x = log_depth(x)
    if normalise:
        # Remove each spectrum's overall level so only its shape remains.
        x = x - x.mean(axis=1, keepdims=True)
    return x, wl, y[TARGETS]


def compute_scores(x, y, seed, smooth=5):
    rows = []
    for t in TARGETS:
        yt = y[t].to_numpy()
        rho = spearmanr(x, yt[:, None]).statistic[-1, :-1]
        mi = mutual_info_regression(x, yt, random_state=seed)

        x_tr, x_te, y_tr, y_te = train_test_split(x, yt, test_size=0.25, random_state=seed)
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.3,
                                   n_jobs=-1, random_state=seed).fit(x_tr, y_tr)
        perm = permutation_importance(rf, x_te, y_te, n_repeats=10, n_jobs=-1,
                                      random_state=seed).importances_mean
        perm = pd.Series(np.clip(perm, 0, None)).rolling(smooth, center=True, min_periods=1).mean()
        r2 = rf.score(x_te, y_te)

        rows.append(pd.DataFrame({"target": t, "bin": np.arange(x.shape[1]), "spearman": rho,
                                  "abs_spearman": np.abs(rho), "mi": mi, "perm": perm.to_numpy(),
                                  "rf_test_r2": r2}))
        print(f"  {t:6s}  max|ρ|={np.abs(rho).max():.2f}  maxMI={mi.max():.2f}  RF test R²={r2:.2f}")
    return pd.concat(rows, ignore_index=True)


def _edges(wl):
    mid = (wl[1:] + wl[:-1]) / 2
    return np.concatenate([[wl[0] - (mid[0] - wl[0])], mid, [wl[-1] + (wl[-1] - mid[-1])]])


def plot_heatmap(scores, wl, x, col, cmap, vmin, vmax, cbar_label, title, path):
    mat = scores.pivot(index="target", columns="bin", values=col).loc[TARGETS].to_numpy()
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 3.2], "hspace": 0.06})
    ax0.plot(wl, x.mean(0), color=MUTED, lw=1.2)
    ax0.set_ylabel("mean log10\ndepth")
    ax0.set_title(title, loc="left")
    m = ax.pcolormesh(_edges(wl), np.arange(len(TARGETS) + 1), mat, cmap=cmap,
                      vmin=vmin, vmax=vmax, shading="flat")
    ax.set_yticks(np.arange(len(TARGETS)) + 0.5, TARGETS)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_xlabel("Wavelength (µm)")
    log_wavelength_axis(ax)
    fig.colorbar(m, ax=[ax0, ax], label=cbar_label, fraction=0.025, pad=0.01)
    fig.savefig(path)
    plt.close(fig)


def plot_profiles(scores, wl, title, path):
    # Each metric is scaled by its max over *all* targets, so weakly determined targets stay low.
    keys = ["abs_spearman", "mi", "perm"]
    gmax = {k: scores[k].max() or 1.0 for k in keys}
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, TARGETS):
        s = scores[scores.target == t]
        for c, key in zip(SERIES, keys):
            ax.plot(wl, s[key] / gmax[key], color=c, label=METRIC_LABELS[key])
        log_wavelength_axis(ax)
        ax.set_title(f"{t}   (RF test R² = {s.rf_test_r2.iloc[0]:.2f})", loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("Wavelength (µm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("score / global max")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=3, bbox_to_anchor=(0.005, 0.965))
    fig.suptitle(title, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def plot_strength_summary(scores, title, path):
    s = scores.groupby("target").agg(abs_spearman=("abs_spearman", "max"), mi=("mi", "max"),
                                     r2=("rf_test_r2", "first")).loc[TARGETS]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, col, lab in zip(axes, ["abs_spearman", "mi", "r2"],
                            ["max |Spearman ρ| over bins", "max MI over bins", "RF held-out R² (all bins)"]):
        ax.barh(TARGETS, s[col], color=SERIES[0], height=0.6)
        for i, v in enumerate(s[col]):
            ax.text(v, i, f" {v:.2f} " if v >= 0 else f"{v:.2f} ", va="center",
                    ha="left" if v >= 0 else "right", color=INK_2, fontsize=8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def top_bins(scores, wl, k=10):
    out = []
    for metric in ["abs_spearman", "mi", "perm"]:
        top = scores.sort_values(metric, ascending=False).groupby("target").head(k).copy()
        top["metric"], top["value"] = metric, top[metric]
        out.append(top[["target", "metric", "bin", "value"]])
    out = pd.concat(out)
    out["wavelength"] = wl[out.bin]
    return out.sort_values(["target", "metric", "value"], ascending=[True, True, False])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fidelity", choices=["hf", "lf10k"], default="hf")
    p.add_argument("--n-sub", type=int, default=3000, help="subsample size for lf10k")
    p.add_argument("--normalise", action="store_true", help="subtract each spectrum's mean log depth")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    set_style()
    x, wl, y = load_xy(a.fidelity, a.n_sub, a.normalise, a.seed)
    tag = f"{a.fidelity}{'_norm' if a.normalise else ''}"
    desc = f"{a.fidelity.upper()}, n={len(x)}{', shape-normalised' if a.normalise else ''}"
    d = out_dir("spectra_drivers")
    print(f"Scoring {desc} ...")

    scores = compute_scores(x, y, a.seed)
    scores["wavelength"] = wl[scores.bin]
    scores.to_csv(d / f"scores_{tag}.csv", index=False)
    top_bins(scores, wl).to_csv(d / f"top_bins_{tag}.csv", index=False)

    lim = np.nanmax(np.abs(scores.spearman))
    plot_heatmap(scores, wl, x, "spearman", CMAP_DIV, -lim, lim, "Spearman ρ",
                 f"Signed Spearman ρ between each bin and each target  ({desc})", d / f"heatmap_spearman_{tag}.png")
    plot_heatmap(scores, wl, x, "mi", CMAP_SEQ, 0, scores.mi.max(), "MI (nats)",
                 f"Mutual information between each bin and each target  ({desc})", d / f"heatmap_mi_{tag}.png")
    plot_profiles(scores, wl, f"Driver profiles per target, each score scaled to its max over all targets  ({desc})",
                  d / f"profiles_{tag}.png")
    plot_strength_summary(scores, f"How strongly the spectrum determines each target  ({desc})",
                          d / f"strength_{tag}.png")
    print(f"Saved to {d}")


if __name__ == "__main__":
    main()
