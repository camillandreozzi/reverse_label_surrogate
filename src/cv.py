"""Cross-validation splits (LOO, k-fold) on HF rows; LF rows always stay in training."""
import numpy as np
from sklearn.model_selection import KFold


def hf_loo_splits(n_hf):
    """[(train_hf_idx, test_hf_idx)] leaving out one HF sample at a time."""
    idx = np.arange(n_hf)
    return [(np.delete(idx, i), np.array([i])) for i in idx]


def hf_kfold_splits(n_hf, k=5, seed=0):
    return list(KFold(k, shuffle=True, random_state=seed).split(np.arange(n_hf)))


def fold_rows(is_hf, train_hf_idx, test_hf_idx):
    """Row indices into the stacked (HF-first) data: all LF rows + train HF rows, and test HF rows."""
    hf_rows = np.flatnonzero(is_hf == 1)
    lf_rows = np.flatnonzero(is_hf == 0)
    return np.r_[hf_rows[train_hf_idx], lf_rows], hf_rows[test_hf_idx]


def cross_validate(model_factory, X, y, is_hf, splits, num_boost_round):
    """Out-of-fold HF predictions for one target.

    model_factory() returns an unfitted IndependentGPBoost; returns (test_hf_idx, mean, var) per fold.
    """
    out = []
    for train_hf, test_hf in splits:
        tr, te = fold_rows(is_hf, train_hf, test_hf)
        model = model_factory().fit(X[tr], y[tr], is_hf[tr], num_boost_round)
        mean, var = model.predict(X[te])
        out.append((test_hf, mean, var))
    return out
