import numpy as np


class UnduplicatedSuggestor:

    def __init__(self) -> None:
        self.__i_history = []

    def argmax(self, a, *args, **kwargs):
        order = np.argsort(a, axis=0)[::-1]
        return self._select(order=order)

    def argmin(self, a, *args, **kwargs):
        order = np.argsort(a, axis=0)
        return self._select(order=order)

    def _select(self, order):
        for i in order:
            if i not in self.__i_history:
                self.__i_history.append(i)
                break
        return i
