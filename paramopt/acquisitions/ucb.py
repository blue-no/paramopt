from typing import Any

import numpy as np

from .base import BaseAcquisition


class UCB(BaseAcquisition):
    """Upper Confidence Bounds.

    Parameters
    ----------
    c: float
        So-called 'C value'. This parameter determines the weight of the
        trade-off between exploration and exploitation. The larger the value,
        the wider the exploration area.
    """
    def __init__(self, c: float) -> None:
        self.c = c

    def __call__(
        self,
        mean: 'np.ndarray',
        std: 'np.ndarray',
        *args: Any,
        **kwargs: Any
    ) -> 'np.ndarray':
        return mean + self.c * std
