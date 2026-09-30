"""Nested leave-one-out CV of the independent model over the 97 HF samples, sf and mf.

For each held-out HF sample i and each of the 9 outputs (independently):
  1. tune with the mf model: Optuna TPE (10 trials) over tree parameters + rounds, each trial scored
     by 3-fold early stopping (MSE) on the other 96 HF rows, with a fixed 1000-row LF subsample always
     in training.
     Sample i is never seen, so the LOO estimate is fully nested.
  2. fit mf (all LF10k rows + the other 96 HF rows) and sf (the other 96 HF rows) with the tuned
     parameters and round count
  3. predict sample i with both
Each (fold, output) pair writes its own files, so Euler runs one array task per pair (97 x 9 = 873),
and an interrupted run can be resubmitted: finished pairs are skipped unless `--overwrite`.
`--merge` combines everything and computes the metrics.

    results/independent/cv_loo/
        folds/fold_<i>_<target>_predictions.csv   variant, y, mean, var, fit_seconds
        folds/fold_<i>_<target>_params.csv        variant, tuned tree params + rounds, covariance parameters
        folds/fold_<i>_<target>_trials.csv        every TPE trial (params, inner-CV MSE, rounds)
        folds/fold_<i>_models/<variant>/<target>/ full boosters (only with --save-models; ~40 MB each for mf)
        predictions.csv   params.csv   metrics.csv (per output and variant: RMSE, NRMSE, R², NLPD, 90% coverage)

Usage (from repo root):
    python modelling/independent/cv_loo.py --folds 0:1 --targets f      # one (fold, output) pair
    python modelling/independent/cv_loo.py --task-id 17                  # pair #17 = fold 17 // 9, output 17 % 9
    python modelling/independent/cv_loo.py --merge
"""
import argparse
import time

import numpy as np
import pandas as pd

from _cli import add_model_args, config_for, out_dir, run_tuning
from src.cv import fold_rows, hf_loo_splits
from src.data_load import TARGETS, load_mf, target_slug
from src.metrics import summarise
from src.model import IndependentGPBoost

N_HF = 97


def run_pair(i, target, train_hf, test_hf, X, Y, is_hf, a):
    fold_dir = out_dir("cv_loo", "folds")
    stem = f"fold_{i:02d}_{target_slug(target)}"
    pred_file, par_file = fold_dir / f"{stem}_predictions.csv", fold_dir / f"{stem}_params.csv"
    if pred_file.exists() and par_file.exists() and not a.overwrite:
        print(f"fold {i} {target}: done, skipping", flush=True)
        return
    tr, te = fold_rows(is_hf, train_hf, test_hf)
    y = Y[target].to_numpy()

    tuning, trials = run_tuning(X[tr], y[tr], is_hf[tr], a)
    n_rounds = tuning["num_boost_round"]
    trials.to_csv(fold_dir / f"{stem}_trials.csv", index=False)
    print(f"fold {i} {target}: tuned (mf, 96 HF + {tuning['n_tune_rows']['lf']} LF) -> {n_rounds} rounds, "
          f"{tuning['best_params']} ({tuning['seconds']:.0f}s)", flush=True)

    preds, params = [], []
    for variant in a.variants:
        t0 = time.time()
        model = IndependentGPBoost(config_for(variant, a, tuning)).fit(X[tr], y[tr], is_hf[tr], n_rounds)
        mean, var = model.predict(X[te])
        secs = time.time() - t0
        preds.append({"fold": i, "sample": int(test_hf[0]), "target": target, "variant": variant,
                      "y": y[te][0], "mean": mean[0], "var": var[0], "fit_seconds": secs})
        params.append({"fold": i, "target": target, "variant": variant, "num_boost_round": n_rounds,
                       **{f"tree_{k}": v for k, v in tuning["best_params"].items()},
                       "inner_best_iterations": " ".join(map(str, tuning["fold_best_iterations"])),
                       "inner_cv_mse": tuning["best_score"], "tune_seconds": tuning["seconds"],
                       **model.cov_pars().to_dict()})
        if a.save_models:
            model.save(fold_dir / f"fold_{i:02d}_models" / variant / target_slug(target))
        print(f"fold {i} {target} [{variant}]: y={y[te][0]:.3g} mean={mean[0]:.3g} "
              f"sd={np.sqrt(var[0]):.3g} ({secs:.0f}s)", flush=True)
    pd.DataFrame(params).to_csv(par_file, index=False)
    pd.DataFrame(preds).to_csv(pred_file, index=False)  # written last: marks the pair as done


def merge():
    d = out_dir("cv_loo")
    pred_files = sorted((d / "folds").glob("fold_*_predictions.csv"))
    if not pred_files:
        print("no finished (fold, output) pairs")
        return
    preds = pd.concat(map(pd.read_csv, pred_files), ignore_index=True)
    params = pd.concat(map(pd.read_csv, sorted((d / "folds").glob("fold_*_params.csv"))), ignore_index=True)
    preds.to_csv(d / "predictions.csv", index=False)
    params.to_csv(d / "params.csv", index=False)

    done = set(zip(preds["sample"], preds["target"]))
    missing = [(i, t) for i in range(N_HF) for t in TARGETS if (i, t) not in done]
    if missing:
        print(f"{len(missing)} of {N_HF * len(TARGETS)} (fold, output) pairs missing, e.g. {missing[:5]}")
    metrics = [{"target": t, "variant": v, "n": len(g), **summarise(g.y, g["mean"], g["var"])}
               for (t, v), g in preds.groupby(["target", "variant"], sort=False)]
    out = pd.DataFrame(metrics)
    out["target"] = pd.Categorical(out.target, TARGETS, ordered=True)
    out = out.sort_values(["target", "variant"])
    out.to_csv(d / "metrics.csv", index=False)
    print(out.to_string(index=False, float_format="%.3f"))


def parse_folds(s):
    a, b = s.split(":")
    return slice(int(a) if a else None, int(b) if b else None)


def main():
    p = add_model_args(argparse.ArgumentParser(description=__doc__,
                                               formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--folds", type=parse_folds, default=slice(None), help="held-out HF samples a:b (default all)")
    p.add_argument("--task-id", type=int, default=None,
                   help="run one (fold, output) pair: fold = id // 9, output = TARGETS[id mod 9]")
    p.add_argument("--merge", action="store_true", help="only combine fold files and compute metrics")
    p.add_argument("--overwrite", action="store_true", help="refit pairs that are already done")
    p.add_argument("--save-models", action="store_true", help="also save every fold's boosters (large)")
    a = p.parse_args()
    if a.merge:
        merge()
        return
    if a.task_id is not None:
        a.folds = slice(a.task_id // len(TARGETS), a.task_id // len(TARGETS) + 1)
        a.targets = [TARGETS[a.task_id % len(TARGETS)]]
    X, Y, is_hf = load_mf(n_lf=a.n_lf, seed=a.seed)
    print(f"{int(is_hf.sum())} HF + {int((is_hf == 0).sum())} LF rows; variants {a.variants}; "
          f"outputs {a.targets}", flush=True)
    for i, (train_hf, test_hf) in list(enumerate(hf_loo_splits(int(is_hf.sum()))))[a.folds]:
        for t in a.targets:
            run_pair(i, t, train_hf, test_hf, X, Y, is_hf, a)


if __name__ == "__main__":
    main()
