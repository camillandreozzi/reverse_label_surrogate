"""Independent per-target model: boosted-tree mean + GP, jointly estimated with GPBoost (sf and mf variants).

mf: AR(1) multi-fidelity Matern GP (`ar1_mf_matern`, f_H = rho * f_L + delta) on HF + LF rows,
    Vecchia approximation with Euclidean neighbours. The fidelity flag is the last `gp_coords` column;
    GPBoost also appends it as a boosting feature (fidelity_specific_mean=True), so a single tree
    ensemble is fitted on all rows.
sf: HF rows only, plain Matern GP, exact (97 rows).
Mean and covariance are estimated jointly by `gpb.train(..., gp_model=...)`: the covariance parameters are
re-estimated at every boosting iteration (train_gp_model_cov_pars=True).
Tuning (`tune_hyperparameters`) is shared by both variants: an Optuna TPE search over the tree
parameters and the number of rounds, scored by k-fold early stopping on HF rows only, using the mf
model (AR(1) GP in the loop) on the HF rows + a fixed subsample of LF rows (default 1000) to keep it
affordable (10 trials, 3 inner folds, early stopping after 10 rounds). The final fits use all rows.
"""
import os
import pickle
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import gpboost as gpb
import numpy as np
import optuna
import pandas as pd

from src.cv import fold_rows, hf_kfold_splits
from src.preprocessing import InputScaler, TargetScaler

# GPBoost's covariance-parameter order per variant, with the readable names used in results
COV_PAR_ORDER = {"mf": ["nugget", "sigma2_L", "range_L", "sigma2_delta", "range_delta", "rho"],
                 "sf": ["nugget", "sigma2", "range"]}
# Starting values used when some parameters are held fixed (GPBoost then needs a full initial vector);
# variances on the standardised-target scale, ranges in standardised-spectrum distance units.
COV_PAR_INIT = {"nugget": 0.5, "sigma2_L": 0.5, "range_L": 10.0, "sigma2_delta": 0.1, "range_delta": 3.0,
                "rho": 1.0, "sigma2": 0.5, "range": 10.0}

COV_PAR_NAMES = {"Error_var": "nugget", "low_GP_var": "sigma2_L", "low_GP_range": "range_L",
                 "discrepancy_GP_var": "sigma2_delta", "discrepancy_GP_range": "range_delta",
                 "rho": "rho", "GP_var": "sigma2", "GP_range": "range"}


def _n_threads():
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(n) if n else None


@dataclass
class ModelConfig:
    variant: str = "mf"  # "mf" (AR(1) HF+LF) or "sf" (HF only)
    cov_fct_shape: float = 1.5  # Matern smoothness nu
    num_neighbors: int = 30  # Vecchia neighbours (mf only)
    max_boost_round: int = 1000  # upper bound for early stopping while tuning
    early_stopping_rounds: int = 10  # while tuning
    tune_n_lf: int = 1000  # LF rows subsampled for tuning (None = all)
    tune_n_trials: int = 10  # TPE trials
    tune_k: int = 3  # inner folds over the HF rows
    gp_optim_params: dict = None  # passed to GPModel.set_optim_params (None = GPBoost defaults, lbfgs)
    fixed_cov_pars: dict = None  # covariance parameters held fixed by name, e.g. {"range_L": 15}; see COV_PAR_ORDER
    boost_params: dict = field(default_factory=lambda: {
        "learning_rate": 0.05, "max_depth": 3, "min_data_in_leaf": 20,
        "feature_fraction": 0.5, "lambda_l2": 1.0, "verbose": -1})


class IndependentGPBoost:
    """One target: tree-ensemble mean + (AR(1)) Matern GP."""

    def __init__(self, config=None):
        self.config = config or ModelConfig()

    # ---------------------------------------------------------------- internals
    def _gp_model(self, Xs, fid):
        c = self.config
        if c.variant == "mf":
            gp = gpb.GPModel(gp_coords=np.column_stack([Xs, fid]), cov_function="ar1_mf_matern",
                             cov_fct_shape=c.cov_fct_shape, gp_approx="vecchia_euclidean",
                             num_neighbors=c.num_neighbors, likelihood="gaussian",
                             num_parallel_threads=_n_threads())
        else:
            gp = gpb.GPModel(gp_coords=Xs, cov_function="matern", cov_fct_shape=c.cov_fct_shape,
                             likelihood="gaussian", num_parallel_threads=_n_threads())
        optim = dict(c.gp_optim_params or {})
        if c.fixed_cov_pars:
            order = COV_PAR_ORDER[c.variant]
            unknown = set(c.fixed_cov_pars) - set(order)
            if unknown:
                raise ValueError(f"unknown covariance parameters for {c.variant}: {unknown}")
            optim["init_cov_pars"] = np.array([c.fixed_cov_pars.get(n, COV_PAR_INIT[n]) for n in order], float)
            optim["estimate_cov_par_index"] = np.array([int(n not in c.fixed_cov_pars) for n in order], np.int32)
        if optim:
            gp.set_optim_params(params=optim)
        return gp

    def _params(self):
        p = dict(self.config.boost_params)
        if _n_threads():
            p["num_threads"] = _n_threads()
        return p

    def _pred_coords(self, Xs):
        # Predictions are always for HF (fidelity = 1).
        return np.column_stack([Xs, np.ones(len(Xs))]) if self.config.variant == "mf" else Xs

    def _select_rows(self, X, y, is_hf):
        if self.config.variant == "sf":
            keep = is_hf == 1
            return X[keep], np.asarray(y)[keep], is_hf[keep]
        return X, np.asarray(y), is_hf

    # ---------------------------------------------------------------- API
    def fit(self, X, y, is_hf, num_boost_round):
        X, y, is_hf = self._select_rows(X, y, is_hf)
        self.x_scaler = InputScaler().fit(X, is_hf)
        self.y_scaler = TargetScaler().fit(y)
        Xs = self.x_scaler.transform(X)
        self.gp_model = self._gp_model(Xs, is_hf)
        self.booster = gpb.train(self._params(), gpb.Dataset(Xs, self.y_scaler.transform(y)),
                                 gp_model=self.gp_model, num_boost_round=int(num_boost_round),
                                 train_gp_model_cov_pars=True)
        self.num_boost_round = int(num_boost_round)
        return self

    def predict(self, X):
        """HF predictive mean and variance (including the nugget) on the original target scale."""
        Xs = self.x_scaler.transform(X)
        p = self.booster.predict(Xs, gp_coords_pred=self._pred_coords(Xs), predict_var=True)
        return self.y_scaler.inverse_mean(p["response_mean"]), self.y_scaler.inverse_var(p["response_var"])

    def cov_pars(self):
        """Covariance parameters on the standardised-target scale, with readable names."""
        cp = self.booster.gp_model.get_cov_pars() if hasattr(self.booster, "gp_model") else self.gp_model.get_cov_pars()
        s = cp.iloc[0] if isinstance(cp, pd.DataFrame) else pd.Series(cp)
        return s.rename(index=lambda k: COV_PAR_NAMES.get(k, k))

    def inner_cv(self, X, y, is_hf, k=5, seed=0):
        """k folds over the HF rows (LF rows always in training), early stopping on the held-out HF rows.

        Returns the per-fold best validation MSE (standardised target) and best iteration.
        """
        c = self.config
        scores, best = [], []
        for train_hf, test_hf in hf_kfold_splits(int(is_hf.sum()), k, seed):
            tr, te = fold_rows(is_hf, train_hf, test_hf)
            Xtr, ytr, ftr = self._select_rows(X[tr], np.asarray(y)[tr], is_hf[tr])
            xs, ys = InputScaler().fit(Xtr, ftr), TargetScaler().fit(ytr)
            Xs_tr, Xs_te = xs.transform(Xtr), xs.transform(X[te])
            gp = self._gp_model(Xs_tr, ftr)
            gp.set_prediction_data(gp_coords_pred=self._pred_coords(Xs_te))
            dtr = gpb.Dataset(Xs_tr, ys.transform(ytr))
            dva = gpb.Dataset(Xs_te, ys.transform(np.asarray(y)[te]), reference=dtr)
            b = gpb.train({**self._params(), "metric": "mse"}, dtr, gp_model=gp, num_boost_round=c.max_boost_round,
                          valid_sets=[dva], early_stopping_rounds=c.early_stopping_rounds,
                          use_gp_model_for_validation=True, train_gp_model_cov_pars=True, verbose_eval=False)
            best.append(max(b.best_iteration, 1))
            scores.append(float(next(iter(b.best_score["valid_0"].values()))))
        return scores, best

    def select_num_rounds(self, X, y, is_hf, k=5, seed=0):
        """Mean early-stopping iteration over k folds of the HF rows (LF rows always in training)."""
        _, best = self.inner_cv(X, y, is_hf, k, seed)
        return int(round(np.mean(best))), best

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(d / "booster.json"))
        with open(d / "meta.pkl", "wb") as f:
            pickle.dump({"config": asdict(self.config), "x_scaler": self.x_scaler,
                         "y_scaler": self.y_scaler, "num_boost_round": self.num_boost_round}, f)

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        with open(d / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        m = cls(ModelConfig(**meta["config"]))
        m.x_scaler, m.y_scaler, m.num_boost_round = meta["x_scaler"], meta["y_scaler"], meta["num_boost_round"]
        m.booster = gpb.Booster(model_file=str(d / "booster.json"))
        m.gp_model = m.booster.gp_model if hasattr(m.booster, "gp_model") else None
        return m


# Model 3 search space (Exoplanets_MF_GPs), bounded by the tuning-set size. learning_rate is limited
# to [0.01, 0.3]: below 0.01 a trial needs thousands of GP-refitting rounds, and trials near 1 were
# clearly worst in a timing probe (each trial costs ~15-20 min single-threaded).
def default_search_space(n_train):
    return {"learning_rate": [0.01, 0.3], "min_data_in_leaf": [1, max(1, min(1000, n_train))],
            "num_leaves": [2, 1024], "lambda_l2": [0, 100], "max_bin": [63, max(63, min(10000, n_train))],
            "feature_fraction": [0.5, 1], "line_search_step_length": [True, False]}


def _suggest(trial, name, bounds):
    lo, hi = bounds
    if name == "line_search_step_length":
        return trial.suggest_categorical(name, [True, False])
    if name in ("num_leaves", "min_data_in_leaf", "max_bin"):
        return trial.suggest_int(name, lo, hi, log=True)
    return trial.suggest_float(name, lo, hi, log=name == "learning_rate")


def tune_hyperparameters(X, y, is_hf, config, seed=0, search_space=None):
    """Shared tuning step for sf and mf: TPE over tree parameters + rounds, with the mf model.

    Uses all HF rows passed in plus `config.tune_n_lf` LF rows (fixed subsample); each trial is scored
    by `config.tune_k`-fold early stopping on the HF rows only. Returns a dict with `best_params`,
    `num_boost_round`, `best_score` and a per-trial table.
    """
    y = np.asarray(y)
    hf_rows, lf_rows = np.flatnonzero(is_hf == 1), np.flatnonzero(is_hf == 0)
    if config.tune_n_lf is not None and config.tune_n_lf < len(lf_rows):
        lf_rows = np.sort(np.random.default_rng(seed).choice(lf_rows, config.tune_n_lf, replace=False))
    rows = np.r_[hf_rows, lf_rows]
    Xt, yt, ft = X[rows], y[rows], is_hf[rows]
    space = search_space or default_search_space(len(rows))
    base = {"max_depth": -1, "verbose": -1}

    def objective(trial):
        params = {**base, **{name: _suggest(trial, name, b) for name, b in space.items()}}
        model = IndependentGPBoost(replace(config, variant="mf", boost_params=params))
        try:
            scores, best = model.inner_cv(Xt, yt, ft, k=config.tune_k, seed=seed)
        except Exception as e:  # e.g. a diverging GP fit for an extreme trial
            trial.set_user_attr("error", str(e)[:200])
            return float("inf")
        trial.set_user_attr("num_boost_round", int(round(np.mean(best))))
        trial.set_user_attr("fold_best_iterations", best)
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=config.tune_n_trials)
    best = study.best_trial
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    return {"best_params": {**base, **best.params}, "num_boost_round": best.user_attrs["num_boost_round"],
            "fold_best_iterations": best.user_attrs["fold_best_iterations"], "best_score": best.value,
            "n_tune_rows": {"hf": int(len(hf_rows)), "lf": int(len(lf_rows))}, "trials": trials}


def apply_tuning(config, tuning):
    """Config with the tuned tree parameters (the round count is passed to fit separately)."""
    return replace(config, boost_params=dict(tuning["best_params"]))


# GPBoost's default covariance optimiser (lbfgs) can abort the whole process with an Eigen assertion on
# rare data sets (seen: sf, Rp, LOO fold 41). Such a fit is retried with this optimiser instead.
FALLBACK_GP_OPTIM = {"optimizer_cov": "gradient_descent"}


def _fit_child(conn, config, X, y, is_hf, num_boost_round, X_pred, save_dir):
    model = IndependentGPBoost(config).fit(X, y, is_hf, num_boost_round)
    if save_dir is not None:
        model.save(save_dir)
    mean, var = model.predict(X_pred)
    conn.send((mean, var, model.cov_pars().to_dict()))
    conn.close()


def fit_predict_safe(config, X, y, is_hf, num_boost_round, X_pred, save_dir=None):
    """Fit + predict in a child process; if GPBoost aborts, retry once with FALLBACK_GP_OPTIM.

    Returns (mean, var, cov_pars dict, gp_optimizer used). The model is saved to `save_dir` if given.
    """
    import multiprocessing as mp
    # "spawn", not "fork": forking after GPBoost/LightGBM have started their OpenMP threads (e.g. during
    # tuning) deadlocks the child in GNU OpenMP. A spawned child is a fresh interpreter (callers need the
    # usual `if __name__ == "__main__":` guard).
    ctx = mp.get_context("spawn")
    for cfg, label in [(config, "default"), (replace(config, gp_optim_params=FALLBACK_GP_OPTIM), "gradient_descent")]:
        recv, send = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_fit_child, args=(send, cfg, X, y, is_hf, num_boost_round, X_pred, save_dir))
        proc.start()
        send.close()
        try:
            result = recv.recv()  # blocks until the child sends or dies (EOFError)
        except EOFError:
            result = None
        proc.join()
        if result is not None and proc.exitcode == 0:
            return (*result, label)
        print(f"fit with {label} GP optimiser failed (exit code {proc.exitcode}); retrying", flush=True)
    raise RuntimeError("fit failed with both the default and the fallback GP optimiser")
