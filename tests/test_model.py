"""Tests for src.model: the ar1_mf / vecchia_euclidean wiring on a tiny subset."""
import numpy as np
import pytest

from src.cv import fold_rows, hf_loo_splits
from src.data_load import load_mf
from src.model import IndependentGPBoost, ModelConfig


@pytest.fixture(scope="module")
def data():
    X, y, is_hf = load_mf(n_lf=100, seed=0)
    keep = np.r_[np.arange(20), np.arange(97, 197)]  # 20 HF + 100 LF
    return X[keep], y["f"].to_numpy()[keep], is_hf[keep]


@pytest.mark.parametrize("variant", ["mf", "sf"])
def test_fit_predict_save_load(tmp_path, data, variant):
    X, y, is_hf = data
    m = IndependentGPBoost(ModelConfig(variant=variant, num_neighbors=10)).fit(X, y, is_hf, 5)
    mean, var = m.predict(X[:3])
    assert mean.shape == var.shape == (3,) and np.all(var > 0)
    cp = m.cov_pars()
    assert ("rho" in cp.index) == (variant == "mf")

    m.save(tmp_path)
    m2 = IndependentGPBoost.load(tmp_path)
    np.testing.assert_allclose(m2.predict(X[:3])[0], mean)


def test_loo_rows_keep_all_lf(data):
    _, _, is_hf = data
    train_hf, test_hf = hf_loo_splits(int(is_hf.sum()))[3]
    tr, te = fold_rows(is_hf, train_hf, test_hf)
    assert list(te) == [3]
    assert set(np.flatnonzero(is_hf == 0)) <= set(tr) and 3 not in tr
