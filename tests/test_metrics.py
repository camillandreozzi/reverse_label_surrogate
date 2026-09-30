"""Tests for src.metrics."""
import numpy as np
import pytest

from src.metrics import coverage, nlpd, nrmse, r2, rmse


def test_rmse_nrmse_r2():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = y + np.array([1.0, -1.0, 1.0, -1.0])
    assert rmse(y, m) == pytest.approx(1.0)
    assert nrmse(y, m) == pytest.approx(1.0 / np.std(y))
    assert r2(y, y) == pytest.approx(1.0)
    assert r2(y, np.full(4, y.mean())) == pytest.approx(0.0)


def test_nlpd_standard_normal():
    assert nlpd([0.0], [0.0], [1.0]) == pytest.approx(0.5 * np.log(2 * np.pi))


def test_coverage():
    y = np.array([0.0, 1.0, 2.0, 10.0])
    # 90% half-width for unit variance is 1.645
    assert coverage(y, np.zeros(4), np.ones(4), 0.9) == pytest.approx(0.5)
