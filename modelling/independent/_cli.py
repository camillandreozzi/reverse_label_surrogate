"""Shared CLI options, paths and the shared tuning step for the independent-model scripts."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_load import RESULTS, TARGETS, target_slug  # noqa: E402
from src.model import ModelConfig, apply_tuning, tune_hyperparameters  # noqa: E402

VARIANTS = ["mf", "sf"]


def add_model_args(p):
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    p.add_argument("--n-lf", type=int, default=None, help="subsample LF10k rows (tests / smoke runs only)")
    p.add_argument("--shape", type=float, default=1.5, help="Matern smoothness nu")
    p.add_argument("--num-neighbors", type=int, default=30, help="Vecchia neighbours (mf)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tune-n-lf", type=int, default=1000, help="LF rows subsampled for tuning")
    p.add_argument("--tune-trials", type=int, default=10, help="TPE trials")
    p.add_argument("--retune", action="store_true", help="ignore a cached tuning result")
    return p


def config_for(variant, a, tuning=None):
    c = ModelConfig(variant=variant, cov_fct_shape=a.shape, num_neighbors=a.num_neighbors,
                    tune_n_lf=a.tune_n_lf, tune_n_trials=a.tune_trials)
    return apply_tuning(c, tuning) if tuning else c


def run_settings(a):
    """Settings the shared tuning result depends on (it is always computed with the mf model)."""
    return {"n_lf": a.n_lf, "shape": a.shape, "num_neighbors": a.num_neighbors, "seed": a.seed,
            "tune_n_lf": a.tune_n_lf, "tune_trials": a.tune_trials}


def run_tuning(X, y, is_hf, a):
    """TPE tuning with the mf model; returns (json-able summary, per-trial table)."""
    t0 = time.time()
    res = tune_hyperparameters(X, y, is_hf, config_for("mf", a), seed=a.seed)
    trials = res.pop("trials")
    res.update({"tuned_with": "mf", "settings": run_settings(a), "seconds": time.time() - t0})
    return res, trials


def out_dir(*parts):
    d = RESULTS / "independent"
    for part in parts:
        d = d / part
    d.mkdir(parents=True, exist_ok=True)
    return d


def shared_tuning(target, X, y, is_hf, a):
    """Tuning result for `target` on all HF rows (full fit), shared by sf and mf.

    Cached in results/independent/tuning/<target>.json (+ _trials.csv); reused when the settings match.
    """
    d = out_dir("tuning")
    f = d / f"{target_slug(target)}.json"
    if f.exists() and not a.retune:
        info = json.loads(f.read_text())
        if info.get("settings") == run_settings(a) and "best_params" in info:
            return info
        print(f"{target}: cached tuning is from different settings; retuning", flush=True)
    res, trials = run_tuning(X, y, is_hf, a)
    f.write_text(json.dumps({"target": target, **res}, indent=2))
    trials.to_csv(d / f"{target_slug(target)}_trials.csv", index=False)
    print(f"{target}: tuned -> {res['num_boost_round']} rounds, {res['best_params']} ({res['seconds']:.0f}s)",
          flush=True)
    return res
