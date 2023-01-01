from typing import Any, Tuple

import numpy as np


class BaseImple:

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError
