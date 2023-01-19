from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .acquisitions import BaseAcquisition
from .graphs.distribution import plot_distribution_1d, plot_distribution_2d
from .graphs.transition import plot_transition
from .parameter import ExplorationSpace

LABEL_NAME = 'label'


class BayesianOptimizer:

    def __init__(
        self,
        regressor: Any,
        exp_space: Union['ExplorationSpace', List[Tuple]],
        eval_name: Union[str, Tuple[str, str]],
        acq_func: 'BaseAcquisition',
        obj_func: Optional[Callable] = None,
        sug_func: Union[Literal['max', 'min'], Callable[['np.ndarray'], int]] = 'max',
        normalize_X: bool = True,
        use: Optional[Literal['sklearn', 'gpy']] = 'sklearn'
    ) -> None:

        if isinstance(eval_name, (tuple, list)):
            self.eval_name = str(eval_name[0])
            self.eval_name_with_unit = f'{str(eval_name[0])} [{str(eval_name[1])}]'
        else:
            self.eval_name = self.eval_name_with_unit = str(eval_name)

        if use.lower() == 'sklearn':
            from .imples.sklearn import SklearnImple
            self._imple = SklearnImple(regressor=regressor)
        elif use.lower() == 'gpy':
            from .imples.gpy import GpyImple
            self._imple = GpyImple(regressor=regressor)
        else:
            raise NotImplementedError(f'{use} not supported')

        if sug_func == 'max':
            self.sug_func = np.argmax
        elif sug_func == 'min':
            self.sug_func = np.argmin
        elif isinstance(sug_func, Callable):
            self.sug_func = sug_func
        else:
            raise ValueError('sug_func must be "min", "max" or callable')

        if isinstance(exp_space, ExplorationSpace):
            self.exp_space = exp_space
        elif isinstance(exp_space, (tuple, list)):
            self.exp_space = ExplorationSpace(exp_space)

        self.acq_func = acq_func
        self.obj_func = obj_func

        self.__X = np.empty((0, len(self.exp_space.axis_names)))
        self.__y = np.empty((0, 1))
        self.__labels = []
        self.__X_next = None

        self.normalize_X = normalize_X
        if self.normalize_X:
            self.__scalers = [MinMaxScaler().fit(np.atleast_2d(values).T) \
                              for values in self.exp_space.axis_values()]

    @property
    def labels(self) -> List[str]:
        return self.__labels.copy()

    @property
    def X(self) -> np.ndarray:
        return self.__X.copy()

    @property
    def y(self) -> np.ndarray:
        return self.__y.copy()

    def load_history(self, io: Union['pd.DataFrame', Path, str]) -> None:
        if isinstance(io, pd.DataFrame):
            df = io
        else:
            io_ = Path(io)
            if io_.suffix == '.xlsx':
                df = pd.read_excel(io_)
            else:
                df = pd.read_csv(io_)

        self.__X = np.atleast_2d(df[self.exp_space.axis_names].values)
        self.__y = np.atleast_2d(df[self.eval_name].values).T
        self.__labels = df[LABEL_NAME].fillna('').astype(str).to_list()

        self._fit(X=self.__X, y=self.__y)

    def save_history(self, fp: Union[Path, str]) -> None:
        df_X = pd.DataFrame(self.__X, columns=self.exp_space.axis_names)
        df_y = pd.DataFrame(self.__y, columns=[self.eval_name])
        df_label = pd.DataFrame(self.__labels, columns=[LABEL_NAME])
        df = pd.concat([df_X, df_y, df_label], axis=1)

        fp_ = Path(fp)
        df.to_csv(fp_, header=True, index=False, mode='w')

    def update(self, X: Any, y: Any, label: Optional[Any] = None) -> None:
        X_ = np.atleast_2d(X)
        y_ = np.atleast_2d(y)
        label_ = str(label) if label is not None else ''

        n_X, n_y = X_.shape[0], y_.shape[0]
        if n_X != n_y:
            raise Exception(f'Data length mismatch: {n_X}(X) != {n_y}(y)')

        self.__X = np.vstack((self.__X, X_))
        self.__y = np.vstack((self.__y, y_))
        self.__labels.append(label_)
        self.__X_next = None

        self._fit(X=self.__X, y=self.__y)

    def suggest(self) -> Union[float, Tuple[float, ...]]:
        X = self.exp_space.points()
        mean, std = self._predict(X=X)
        mean, std = mean.reshape(-1, 1), std.reshape(-1, 1)
        acq = self.acq_func(mean=mean, std=std, X=self.__X, y=self.__y)
        X_next = X[self.sug_func(acq)]
        if len(X_next) == 1:
            self.__X_next = X_next[0]
        else:
            self.__X_next = X_next
        return self.__X_next

    def plot_distribution(
        self,
        fig: Optional['plt.Figure'] = None,
        save_as: Optional[Union[Path, str]] = None
    ) -> 'plt.Figure':
        if fig is None:
            fig = plt.figure()

        space = self.exp_space
        X = space.grid_points()
        mean, std = self._predict(X=X)
        mean, std = mean.reshape(-1, 1), std.reshape(-1, 1)
        acq = self.acq_func(mean=mean, std=std, X=self.__X, y=self.__y)

        if space.ndim == 1:
            fig = plot_distribution_1d(
                fig=fig,
                X=self.__X,
                y=self.__y,
                axis_values=space.grid_axis_values()[0],
                mean=mean,
                std=std,
                acq=acq,
                X_next=self.__X_next,
                obj_func=self.obj_func,
                x_label=space.axis_names_with_unit[0],
                y_label=self.eval_name_with_unit,
                acq_label=self.acq_func.name
            )
        elif space.ndim == 2:
            fig = plot_distribution_2d(
                fig=fig,
                X=self.__X,
                y=self.__y,
                axis_values=space.grid_axis_values(),
                mean=mean,
                acq=acq,
                X_next=self.__X_next,
                obj_func=self.obj_func,
                x_label=space.axis_names_with_unit[0],
                y_label=space.axis_names_with_unit[1],
                z_label=self.eval_name_with_unit,
                acq_label=self.acq_func.name
            )
        else:
            raise NotImplementedError(f'{space.ndim}D-plot is not supported')

        if save_as is not None:
            fp = Path(save_as)
            fp.parent.mkdir(exist_ok=True)
            fig.savefig(fp.as_posix())
        return fig

    def plot_transition(
        self,
        fig: Optional['plt.Figure'] = None,
        save_as: Optional[Union[Path, str]] = None
    ) -> 'plt.Figure':
        if fig is None:
            fig = plt.figure()

        space = self.exp_space
        fig = plot_transition(
            fig=fig,
            X=self.__X,
            y=self.__y,
            axis_values=space.axis_values(),
            x_names=space.axis_names,
            y_names=[self.eval_name]
        )

        if save_as is not None:
            fp = Path(save_as)
            fp.parent.mkdir(exist_ok=True)
            fig.savefig(fp.as_posix())
        return fig

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.normalize_X:
            self._imple.fit(X=self._normalize(X), y=y)
        else:
            self._imple.fit(X=X, y=y)

    def _predict(self, X: np.ndarray) -> Any:
        if self.normalize_X:
            return self._imple.predict(X=self._normalize(X))
        return self._imple.predict(X=X)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        X_ = np.hstack([scaler.transform(np.atleast_2d(x).T) for x, scaler \
                       in zip(X.T, self.__scalers)])
        return X_
