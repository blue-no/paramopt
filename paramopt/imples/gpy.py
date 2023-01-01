from typing import Tuple

import numpy as np
from GPy.models import GPRegression

from .base import BaseImple


class GpyImple(BaseImple):

    def __init__(self, regressor: 'GPRegression') -> None:
        self.regressor = regressor

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.regressor.set_XY(X=X, Y=y)
        self.regressor.optimize()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
        mean, var = self.regressor.predict(Xnew=X)
        return (mean, np.sqrt(var))
