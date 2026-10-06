"""Diagnostic: refit LOO folds with some covariance parameters held fixed, to tame runaway GP fits.

In the nested LOO, the LF GP of the mf model (and the sf GP) often ran to sigma2 ~ 1e6-1e9 and
range ~ 1e4 on the standardised scale, i.e. far beyond the data (HF spectra are ~17 apart, ~3-4 from
their nearest neighbour). For [S/H] this gave predictions far outside the prior range. This script
refits chosen folds with candidate fixes, reusing each fold's tuned tree parameters and round count
from the LOO (results/independent/cv_loo/folds/*_params.csv), so only the GP part changes.

One task = one (fold, mf config); the cheap sf configs run inside the task of the first mf config.
    results/independent/diagnostics/fixed_cov/<target>/fold_<i>_<config>.csv
    --summarise  ->  summary.csv comparing configs on the folds that finished

Usage (from repo root):
    python modelling/independent/diag_fixed_cov.py --target "[S/H]" --task-id 0
    python modelling/independent/diag_fixed_cov.py --target "[S/H]" --summarise
"""
import argparse
import time

import numpy as np
import pandas as pd

from _cli import out_dir
from src.cv import fold_rows, hf_loo_splits
from src.data_load import load_mf, target_slug
from src.metrics import summarise
from src.model import ModelConfig, fit_predict_safe

MF_CONFIGS = {"mf_rangeL5": {"range_L": 5.0}, "mf_rangeL15": {"range_L": 15.0}, "mf_rho1": {"rho": 1.0}}
SF_CONFIGS = {"sf_range5": {"range": 5.0}, "sf_range15": {"range": 15.0}}
# 6 folds with the worst mf errors in the LOO + 6 others (fixed seed), per target
N_WORST, N_OTHER = 6, 6


def chosen_folds(target):
    d = out_dir("cv_loo", "folds")
    p = pd.read_csv(d.parent / "predictions.csv") if (d.parent / "predictions.csv").exists() else \
        pd.concat(map(pd.read_csv, d.glob(f"fold_*_{target_slug(target)}_predictions.csv")))
    m = p[(p.target == target) & (p.variant == "mf")].copy()
    m["err"] = (m["mean"] - m.y).abs()
    worst = m.nlargest(N_WORST, "err")["fold"].tolist()
    rest = sorted(set(m["fold"]) - set(worst))
    other = np.random.default_rng(0).choice(rest, N_OTHER, replace=False).tolist()
    return sorted(int(f) for f in worst + other)


def tuned(target, fold):
    q = pd.read_csv(out_dir("cv_loo", "folds") / f"fold_{fold:02d}_{target_slug(target)}_params.csv")
    row = q[q.variant == "mf"].iloc[0]
    params = {k[5:]: row[k] for k in q.columns if k.startswith("tree_")}
    for k in ("max_depth", "verbose", "num_leaves", "min_data_in_leaf", "max_bin"):
        if k in params:
            params[k] = int(params[k])
    if "line_search_step_length" in params:
        params["line_search_step_length"] = str(params["line_search_step_length"]) == "True"
    return params, int(row["num_boost_round"])


def run_task(target, task_id, X, Y, is_hf):
    folds = chosen_folds(target)
    fold = folds[task_id // len(MF_CONFIGS)]
    mf_name = list(MF_CONFIGS)[task_id % len(MF_CONFIGS)]
    configs = {mf_name: ("mf", MF_CONFIGS[mf_name])}
    if task_id % len(MF_CONFIGS) == 0:
        configs.update({n: ("sf", c) for n, c in SF_CONFIGS.items()})
    params, n_rounds = tuned(target, fold)
    train_hf, test_hf = hf_loo_splits(int(is_hf.sum()))[fold]
    tr, te = fold_rows(is_hf, train_hf, test_hf)
    y = Y[target].to_numpy()
    d = out_dir("diagnostics", "fixed_cov", target_slug(target))
    for name, (variant, fixed) in configs.items():
        t0 = time.time()
        cfg = ModelConfig(variant=variant, boost_params=params, fixed_cov_pars=fixed)
        mean, var, cp, opt = fit_predict_safe(cfg, X[tr], y[tr], is_hf[tr], n_rounds, X[te])
        row = {"fold": fold, "target": target, "config": name, "variant": variant, "y": y[te][0],
               "mean": mean[0], "var": var[0], "gp_optimizer": opt, "seconds": time.time() - t0, **cp}
        pd.DataFrame([row]).to_csv(d / f"fold_{fold:02d}_{name}.csv", index=False)
        print(f"fold {fold} {name}: y={y[te][0]:.3f} mean={mean[0]:.3f} var={var[0]:.3g} "
              f"({time.time() - t0:.0f}s) " + " ".join(f"{k}={v:.3g}" for k, v in cp.items()), flush=True)


def summarise_target(target):
    d = out_dir("diagnostics", "fixed_cov", target_slug(target))
    new = pd.concat(map(pd.read_csv, d.glob("fold_*.csv")), ignore_index=True)
    base = pd.read_csv(out_dir("cv_loo") / "predictions.csv") if (out_dir("cv_loo") / "predictions.csv").exists() \
        else pd.concat(map(pd.read_csv, out_dir("cv_loo", "folds").glob(f"fold_*_{target_slug(target)}_predictions.csv")))
    base = base[(base.target == target) & base["fold"].isin(new["fold"])]
    base = base.assign(config=base.variant + "_original")
    allr = pd.concat([base[["fold", "config", "y", "mean", "var"]], new[["fold", "config", "y", "mean", "var"]]])
    rows = []
    for c, g in allr.groupby("config"):
        ok = g[g["var"] > 0]
        m = summarise(g.y, g["mean"], g["var"].clip(lower=1e-12))
        rows.append({"config": c, "n": len(g), "rmse": m["rmse"], "r2": m["r2"],
                     "outside_prior": int(((g["mean"] < -1) | (g["mean"] > 2)).sum()),
                     "neg_var": len(g) - len(ok),
                     "coverage_90_valid": summarise(ok.y, ok["mean"], ok["var"])["coverage_90"] if len(ok) > 1 else np.nan})
    out = pd.DataFrame(rows).sort_values("config")
    out.to_csv(d / "summary.csv", index=False)
    print(out.to_string(index=False, float_format="%.3f"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="[S/H]")
    p.add_argument("--task-id", type=int, help=f"0 .. {(N_WORST + N_OTHER) * len(MF_CONFIGS) - 1}")
    p.add_argument("--summarise", action="store_true")
    a = p.parse_args()
    if a.summarise:
        summarise_target(a.target)
        return
    X, Y, is_hf = load_mf()
    run_task(a.target, a.task_id, X, Y, is_hf)


if __name__ == "__main__":
    main()
