"""Input scaling with HF statistics and target standardisation."""
import numpy as np


class InputScaler:
    """Per-bin standardisation of the raw depths, using mean/std of the HF training rows."""

    def fit(self, X, is_hf):
        hf = X[is_hf == 1]
        self.mean_ = hf.mean(0)
        self.std_ = hf.std(0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


class TargetScaler:
    """Standardise one target on the training rows; inverse maps mean and variance back."""

    def fit(self, y):
        self.mean_ = float(np.mean(y))
        self.std_ = float(np.std(y)) or 1.0
        return self

    def transform(self, y):
        return (np.asarray(y) - self.mean_) / self.std_

    def inverse_mean(self, m):
        return np.asarray(m) * self.std_ + self.mean_

    def inverse_var(self, v):
        return np.asarray(v) * self.std_ ** 2
