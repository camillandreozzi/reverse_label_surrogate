"""Cross-target correlation (motivates coregionalization).

The 9 parameters are sampled (near-)independently, so their *marginal* correlation says little
about what coregionalization can exploit. What matters is dependence *given the spectrum*:
if one spectrum is compatible with several parameter combinations (degeneracies), the errors
of independent per-target models are correlated, and a shared output covariance can use that.

For HF (97) and LF10k (subsample), three 9x9 matrices:
  targets      corr(y_i, y_j)                                   design / prior correlation
  predictions  corr(ŷ_i, ŷ_j)   out-of-fold RF predictions from the spectrum
  residuals    corr(y_i − ŷ_i, y_j − ŷ_j)                        conditional dependence -> coreg
Cells whose correlation is not significant (p > 0.05) are shown with faded text.

Usage (from repo root):
    python eda/target_correlation.py [--method spearman] [--n-sub 3000]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

from _eda_common import (CMAP_DIV, INK, MUTED, SERIES, TARGETS, load_hf, load_lf10k, log_depth,
                         out_dir, plt, set_style)


def corr_matrix(df, method):
    f = spearmanr if method == "spearman" else pearsonr
    k = len(TARGETS)
    r, p = np.eye(k), np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            res = f(df.iloc[:, i], df.iloc[:, j])
            r[i, j] = r[j, i] = res.statistic
            p[i, j] = p[j, i] = res.pvalue
    return pd.DataFrame(r, TARGETS, TARGETS), pd.DataFrame(p, TARGETS, TARGETS)


def oof_predictions(x, y, seed):
    cv = KFold(5, shuffle=True, random_state=seed)
    pred = {}
    for t in TARGETS:
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.3,
                                   n_jobs=-1, random_state=seed)
        pred[t] = cross_val_predict(rf, x, y[t], cv=cv)
    return pd.DataFrame(pred)


def draw_matrix(ax, r, p, title):
    k = len(TARGETS)
    m = ax.imshow(r.to_numpy(), cmap=CMAP_DIV, vmin=-1, vmax=1)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            sig = p.iat[i, j] <= 0.05
            v = r.iat[i, j]
            ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center",
                    fontsize=7, color=("white" if abs(v) > 0.6 else INK) if sig else MUTED,
                    alpha=1 if sig else 0.6)
    ax.set_xticks(range(k), TARGETS, rotation=45, ha="right")
    ax.set_yticks(range(k), TARGETS)
    ax.grid(False)
    ax.set_title(title, loc="left")
    return m


def top_pairs(mats, n=10):
    rows = []
    for (data, kind), (r, p) in mats.items():
        iu = np.triu_indices(len(TARGETS), 1)
        for i, j in zip(*iu):
            rows.append({"data": data, "matrix": kind, "pair": f"{TARGETS[i]} – {TARGETS[j]}",
                         "r": r.iat[i, j], "p": p.iat[i, j]})
    df = pd.DataFrame(rows)
    df["abs_r"] = df.r.abs()
    return df.sort_values("abs_r", ascending=False)


def plot_residual_pairs(pairs, path, n=12):
    res = pairs[pairs.matrix == "residuals"]
    order = (res.groupby("pair").abs_r.max().sort_values(ascending=False).index[:n])
    fig, ax = plt.subplots(figsize=(8, 0.35 * n + 1.2))
    y = np.arange(len(order))
    for off, (data, c) in zip([-0.18, 0.18], [("HF", SERIES[0]), ("LF10k", SERIES[2])]):
        v = res[res.data == data].set_index("pair").r.reindex(order)
        ax.barh(y + off, v, height=0.34, color=c, label=data)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("residual correlation")
    ax.set_title(f"Strongest conditional (residual) target pairs", loc="left")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["pearson", "spearman"], default="spearman")
    p.add_argument("--n-sub", type=int, default=3000, help="LF10k subsample for the RF residuals")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    set_style()
    x_hf, _, y_hf = load_hf()
    x_lf, _, y_lf = load_lf10k()
    idx = np.random.default_rng(a.seed).choice(len(x_lf), min(a.n_sub, len(x_lf)), replace=False)
    x_lf, y_lf = x_lf[idx], y_lf.iloc[idx].reset_index(drop=True)
    d = out_dir("target_correlation")

    mats = {}
    for name, x, y in [("HF", x_hf, y_hf[TARGETS]), ("LF10k", x_lf, y_lf[TARGETS])]:
        print(f"{name}: out-of-fold RF predictions (n={len(y)}) ...")
        pred = oof_predictions(log_depth(x), y, a.seed)
        resid = y.reset_index(drop=True) - pred
        for kind, df in [("targets", y), ("predictions", pred), ("residuals", resid)]:
            mats[(name, kind)] = corr_matrix(df, a.method)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))
    for row, name in enumerate(["HF", "LF10k"]):
        n = len(y_hf) if name == "HF" else len(y_lf)
        for col, kind in enumerate(["targets", "predictions", "residuals"]):
            m = draw_matrix(axes[row, col], *mats[(name, kind)], f"{name} (n={n}): {kind}")
    fig.colorbar(m, ax=axes, label=f"{a.method.capitalize()} correlation", fraction=0.02, pad=0.01)
    fig.suptitle("Between-output correlation: marginal, of spectrum-based predictions, and of residuals"
                 "  (faded = p > 0.05)", x=0.01, ha="left", y=0.995)
    fig.savefig(d / f"corr_matrices_{a.method}.png")
    plt.close(fig)

    pairs = top_pairs(mats)
    pairs.to_csv(d / f"pairs_{a.method}.csv", index=False)
    for (name, kind), (r, _) in mats.items():
        r.to_csv(d / f"{name}_{kind}_{a.method}.csv")
    plot_residual_pairs(pairs, d / f"residual_pairs_{a.method}.png")

    print("\nStrongest residual pairs:")
    print(pairs[pairs.matrix == "residuals"].head(8)[["data", "pair", "r", "p"]].to_string(index=False))
    print(f"Saved to {d}")


if __name__ == "__main__":
    main()
