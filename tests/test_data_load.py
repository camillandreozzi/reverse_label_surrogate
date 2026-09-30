"""Tests for src.data_load."""
import numpy as np

from src.data_load import TARGETS, load_hf, load_lf10k, load_mf, target_slug, wavelengths


def test_shapes():
    x_hf, y_hf = load_hf()
    x_lf, y_lf = load_lf10k()
    assert x_hf.shape == (97, 195) and y_hf.shape == (97, 9)
    assert x_lf.shape == (10000, 195) and y_lf.shape == (10000, 9)
    assert list(y_hf.columns) == TARGETS
    assert len(wavelengths()) == 195


def test_load_mf_hf_first_and_flags():
    X, y, is_hf = load_mf()
    assert X.shape == (10097, 195) and len(y) == 10097
    assert is_hf[:97].all() and not is_hf[97:].any()
    np.testing.assert_array_equal(X[:97], load_hf()[0])


def test_load_mf_subsample():
    X, y, is_hf = load_mf(n_lf=50, seed=1)
    assert X.shape == (147, 195) and int(is_hf.sum()) == 97


def test_target_slug_is_filename_safe():
    slugs = [target_slug(t) for t in TARGETS]
    assert len(set(slugs)) == 9
    assert not any(c in s for s in slugs for c in "/[]")
