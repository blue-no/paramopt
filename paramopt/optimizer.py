from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .graphs.distribution import plot_distribution_1d, plot_distribution_2d
from .graphs.transition import plot_transition
from .parameter import ExplorationSpace
from .acquisitions import BaseAcquisition
from .imples import BaseImple

LABEL_NAME = 'label'
DIST_PREFIX = 'dist-'
TRANS_PREFIX = 'trans-'


class BayesianOptimizer:

    def __init__(
        self,
        regressor: 'BaseImple',
        exp_space: 'ExplorationSpace',
        eval_name: Union[str, List[str]],
        acq_func: 'BaseAcquisition',
        obj_func: Optional[Callable] = None,
        suggest_func: Union[Literal['max', 'min'], Callable] = 'max',
        working_dir: Union[Path, str] = None,
        normalize_X: bool = True,
        use: Optional[Literal['sklearn', 'gpy']] = 'sklearn'
    ) -> None:

        if isinstance(eval_name, list):
            self.eval_name = eval_name
        else:
            self.eval_name = [eval_name]

        if working_dir is not None:
            self.working_dir = Path(working_dir)
            self.working_dir.mkdir(exist_ok=True)
        else:
            self.working_dir = Path.cwd()

        if use.lower() == 'sklearn':
            from .imples.sklearn import SklearnImple
            self.imple = SklearnImple(regressor=regressor)
        elif use.lower() == 'gpy':
            from .imples.gpy import GpyImple
            self.imple = GpyImple(regressor=regressor)
        else:
            raise NotImplementedError(f'{use} not supported')

        if suggest_func == 'max':
            self.suggest_func = np.argmax
        elif suggest_func == 'min':
            self.suggest_func = np.argmin
        elif isinstance(suggest_func, Callable):
            self.suggest_func = suggest_func
        else:
            raise ValueError('suggest_func must be "min", "max" or callable')

        self.exp_space = exp_space
        self.acq_func = acq_func
        self.obj_func = obj_func

        self.__X = np.empty((0, len(self.exp_space.axis_names)))
        self.__y = np.empty((0, len(self.eval_name)))
        self.__labels = []
        self.__X_next = None

        self.normalize_X = normalize_X
        if self.normalize_X:
            self.__scaler = MinMaxScaler().fit(
                np.atleast_2d(self.exp_space.axis_values()).T)

    def load_history(self, io: Union['pd.DataFrame', Path, str]) -> None:
        if isinstance(io, pd.DataFrame):
            df = io
        else:
            path_ = Path(io)
            if path_.suffix == '.xlsx':
                df = pd.read_excel(path_)
            else:
                df = pd.read_csv(path_)

        df = df.dropna()
        self.__X = np.atleast_2d(df[self.exp_space.axis_names].values)
        self.__y = np.atleast_2d(df[self.eval_name].values)
        self.__labels = df[LABEL_NAME].fillna('').to_list()

        self._fit(X=self.__X, y=self.__y)

    def save_history(self, path: Union[Path, str]) -> None:
        df_X = pd.DataFrame(self.__X, columns=self.exp_space.axis_names)
        df_y = pd.DataFrame(self.__y, columns=self.eval_name)
        df_label = pd.DataFrame(self.__labels, columns=[LABEL_NAME])
        df = pd.concat([df_X, df_y, df_label], axis=1)

        path_ = Path(path)
        df.to_csv(path_, header=True, index=False, mode='w')

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

        self._fit(X=self.__X, y=self.__y)

    def suggest(self) -> Union[float, Tuple[float, ...]]:
        X = self.exp_space.points()
        mean, std = self._predict(X=X)
        mean, std = mean.reshape(-1, 1), std.reshape(-1, 1)
        acq = self.acq_func(mean=mean, std=std, X=self.__X, y=self.__y)
        X_next = X[self.suggest_func(acq)]
        if len(X_next) == 1:
            self.__X_next = X_next[0]
        else:
            self.__X_next = X_next
        return self.__X_next

    def plot_distribution(self, fig: Optional['plt.Figure'] = None) -> 'plt.Figure':
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
                y_label=self.eval_name,
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
                z_label=self.eval_name[0],
                acq_label=self.acq_func.name
            )
        else:
            raise NotImplementedError(f'{space.ndim}D-plot is not supported')
        fig.savefig(self.working_dir/f'{DIST_PREFIX}{self.__labels[-1]}.png')
        return fig

    def plot_transition(self, fig: Optional['plt.Figure'] = None) -> 'plt.Figure':
        if fig is None:
            fig = plt.figure()

        space = self.exp_space
        fig = plot_transition(
            fig=fig,
            X=self.__X,
            y=self.__y,
            axis_values=space.axis_values(),
            x_names=space.axis_names_with_unit,
            y_names=self.eval_name
        )
        fig.savefig(self.working_dir/f'{TRANS_PREFIX}{self.__labels[-1]}.png')
        return fig

    def _fit(self, X, y):
        if self.normalize_X:
            X = self.__scaler.transform(np.atleast_2d(X))
        self.imple.fit(X=X, y=y)

    def _predict(self, X):
        if self.normalize_X:
            X = self.__scaler.transform(np.atleast_2d(X))
        return self.imple.predict(X=X)
