"""Fit the independent model on all data, per target, sf and mf.

For each target: one shared tuning step (TPE, 10 trials, over tree parameters + rounds with the mf model
on all HF rows + a 1000-row LF subsample, scored by 3-fold early stopping on HF rows), then both variants are
fitted with the tuned parameters and round count:
  mf  tree mean + AR(1) Matern GP on HF97 + LF10k
  sf  tree mean + Matern GP on HF97 only
Every target writes its own files so Euler array tasks can run targets in parallel:

    results/independent/
        tuning/<target>.json, _trials.csv    shared tuned parameters + rounds (tuned with mf), TPE trials
        <variant>/models/<target>/           booster.json (trees + GP) and meta.pkl (scalers, config)
        <variant>/cov_pars/<target>.csv      covariance parameters (standardised-target scale; rho for mf)
        <variant>/insample/<target>.csv      in-sample HF predictions
    --merge  ->  tuning.json, <variant>/cov_pars.csv (all targets)

Usage (from repo root):
    python modelling/independent/full_fit_model.py [--variants mf sf] [--targets f "C/O"]
    python modelling/independent/full_fit_model.py --merge
"""
import argparse
import json
import time

import pandas as pd

from _cli import add_model_args, config_for, out_dir, shared_tuning
from src.data_load import TARGETS, load_mf, target_slug
from src.model import fit_predict_safe


def fit_target(target, X, y, is_hf, a):
    slug = target_slug(target)
    tuning = shared_tuning(target, X, y, is_hf, a)
    n_rounds = tuning["num_boost_round"]
    for variant in a.variants:
        t0 = time.time()
        hf = is_hf == 1
        mean, var, cov_pars, gp_opt = fit_predict_safe(config_for(variant, a, tuning), X, y, is_hf, n_rounds,
                                                        X[hf], out_dir(variant, "models", slug))
        cp = pd.Series({**cov_pars, "gp_optimizer": gp_opt})
        cp.to_frame(target).T.rename_axis("target").to_csv(out_dir(variant, "cov_pars") / f"{slug}.csv")
        pd.DataFrame({"sample": range(hf.sum()), "y": y[hf], "mean": mean, "var": var}).to_csv(
            out_dir(variant, "insample") / f"{slug}.csv", index=False)
        print(f"{target} [{variant}]: rounds={n_rounds} fit {time.time() - t0:.0f}s  "
              + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in cp.items()), flush=True)


def merge(variants):
    tuned = {}
    for t in TARGETS:
        f = out_dir("tuning") / f"{target_slug(t)}.json"
        if f.exists():
            info = json.loads(f.read_text())
            tuned[t] = {"num_boost_round": info["num_boost_round"], **info.get("best_params", {})}
    (out_dir() / "tuning.json").write_text(json.dumps(tuned, indent=2))
    print(pd.DataFrame(tuned).T.to_string())
    for variant in variants:
        cps = [pd.read_csv(f) for t in TARGETS
               if (f := out_dir(variant, "cov_pars") / f"{target_slug(t)}.csv").exists()]
        missing = len(TARGETS) - len(cps)
        if cps:
            df = pd.concat(cps)
            df.to_csv(out_dir(variant) / "cov_pars.csv", index=False)
            print(f"\n[{variant}]" + (f" ({missing} targets missing)" if missing else ""))
            print(df.to_string(index=False))


def main():
    p = add_model_args(argparse.ArgumentParser(description=__doc__,
                                               formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--merge", action="store_true", help="only combine per-target outputs")
    a = p.parse_args()
    if a.merge:
        merge(a.variants)
        return
    X, y, is_hf = load_mf(n_lf=a.n_lf, seed=a.seed)
    print(f"{int(is_hf.sum())} HF + {int((is_hf == 0).sum())} LF rows; variants {a.variants}", flush=True)
    for t in a.targets:
        fit_target(t, X, y[t].to_numpy(), is_hf, a)


if __name__ == "__main__":
    main()
