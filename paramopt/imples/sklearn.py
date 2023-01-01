from typing import Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from .base import BaseImple


class SklearnImple(BaseImple):

    def __init__(self, regressor: 'GaussianProcessRegressor') -> None:
        self.regressor = regressor

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.regressor.fit(X, y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
        mean, std = self.regressor.predict(X, return_std=True)
        return (mean, std)
