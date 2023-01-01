from typing import Any

import numpy as np


class BaseAcquisition:

    def __call__(
        self,
        mean: 'np.ndarray',
        std: 'np.ndarray',
        *args: Any,
        **kwargs: Any
    ) -> 'np.ndarray':
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__
