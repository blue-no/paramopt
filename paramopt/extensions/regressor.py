import itertools
import warnings
from copy import copy
from pathlib import Path
from typing import Any, Callable, Dict, Union

import pandas as pd
from sklearn.gaussian_process.kernels import ConvergenceWarning


class AutoHyperparameter:

    def __init__(self, params):
        self.__names = list(params.keys())
        self.__combos = list(itertools.product(*params.values()))
        self.__cur_combo = self.__combos[0]

    @property
    def names(self):
        return self.__names

    def selected_params(self):
        return copy(self.__cur_combo)

    def iterate(self):
        for combo in self.__combos:
            self.__cur_combo = combo
            yield

    def select(self, item):
        return copy(self.__cur_combo[self.__names.index(item)])


class AutoHyperparameterRegressor:

    def __init__(
        self,
        hyperparams: Dict[str, list],
        regressor_factory: Callable[['AutoHyperparameter'], Any]
    ) -> None:
        self.autohp = AutoHyperparameter(params=hyperparams)
        self.regressor_factory = regressor_factory
        self.history = []
        self.regressor = regressor_factory(self.autohp)

    def fit(self, X, y, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            for _ in self.autohp.iterate():
                regressor = self.regressor_factory(self.autohp)
                try:
                    regressor.fit(X, y, *args, **kwargs)
                    failed = False
                    break
                except ConvergenceWarning:
                    pass
            else:
                failed = True

        if failed:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                regressor.fit(X, y, *args, **kwargs)
            warnings.warn('Auto adjustment not worked well', UserWarning)

        self.regressor = regressor
        self.history.append(self.autohp.selected_params())

    def predict(self, X, *args, **kwargs):
        return self.regressor.predict(X, *args, **kwargs)

    def dump_hp_history(self, fp: Union[Path, str]) -> None:
        fp_ = Path(fp)
        fp_.parent.mkdir(exist_ok=True)

        df = pd.DataFrame(self.history, columns=self.autohp.names)
        df.to_csv(fp_, mode='w', header=True, index=False)
