"""Per-target metrics: RMSE, NRMSE, R², Gaussian NLPD and predictive-interval coverage."""
import numpy as np
from scipy.stats import norm


def rmse(y, mean):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(mean)) ** 2)))


def nrmse(y, mean):
    """RMSE divided by the standard deviation of y (1 = no better than predicting the mean); NaN if y is constant."""
    sd = float(np.std(y))
    return rmse(y, mean) / sd if sd > 0 else float("nan")


def r2(y, mean):
    y, mean = np.asarray(y), np.asarray(mean)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - mean) ** 2).sum() / ss) if ss > 0 else float("nan")


def nlpd(y, mean, var):
    """Mean negative log predictive density under a Gaussian predictive distribution."""
    y, mean, var = map(np.asarray, (y, mean, var))
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mean) ** 2 / var))


def coverage(y, mean, var, level=0.9):
    """Fraction of y inside the central `level` Gaussian predictive interval."""
    y, mean, var = map(np.asarray, (y, mean, var))
    half = norm.ppf(0.5 + level / 2) * np.sqrt(var)
    return float(np.mean(np.abs(y - mean) <= half))


def summarise(y, mean, var, level=0.9):
    return {"rmse": rmse(y, mean), "nrmse": nrmse(y, mean), "r2": r2(y, mean),
            "nlpd": nlpd(y, mean, var), f"coverage_{int(level * 100)}": coverage(y, mean, var, level)}
