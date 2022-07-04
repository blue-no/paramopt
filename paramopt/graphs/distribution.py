from typing import Any, Callable, List, Optional, Union
from matplotlib import cm, gridspec, pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from paramopt.structures.dataset import Dataset
from .base import BaseGraph

from ..structures.parameter import ExplorationSpace


class Distribution(BaseGraph):

    PNG_PREFIX = "dist-"

    def plot(
        self,
        exploration_space: "ExplorationSpace",
        dataset: "Dataset",
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        acquisition: Optional[np.ndarray] = None,
        objective_fn: Optional[Callable] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        if exploration_space.dimension != dataset.dimension_X:
            raise ValueError(
                "exploration dimension does not match dataset dimension")

        ndim = exploration_space.dimension
        if ndim == 1:
            self.fig = _plot_process_1d(
                fig=plt.figure(*args, **kwargs),
                X=dataset.X,
                Y=dataset.Y,
                X_grid=exploration_space.grid_spaces[0],
                mean=mean,
                std=std,
                acquisition=acquisition,
                objective_fn=objective_fn,
                X_name=exploration_space.names[0],
                Y_name=dataset.Y_names[0])
        elif ndim == 2:
            self.fig = _plot_process_2d(
                fig=plt.figure(*args, **kwargs),
                X=dataset.X,
                Y=dataset.Y,
                X_grids=exploration_space.grid_spaces,
                mean=mean,
                acquisition=acquisition,
                objective_fn=objective_fn,
                X_names=exploration_space.names,
                Y_name=dataset.Y_names[0])
        else:
            raise NotImplementedError(f"{ndim}D plot is not supported")
        self.fig.tight_layout()


def _plot_process_1d(
    fig: Figure,
    X: np.ndarray,
    Y: np.ndarray,
    X_grid: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    acquisition: Optional[np.ndarray] = None,
    objective_fn: Optional[Callable] = None,
    X_name: str = "x",
    Y_name: str = "y"
) -> Figure:
    if acquisition is not None:
        spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3, 1])
        ax_upper = fig.add_subplot(spec[0])
        ax_lower = fig.add_subplot(spec[1], sharex=ax_upper)
        ax_lower.plot(X_grid, acquisition, 'r-')
        ax_lower.set_xlabel(X_name)
        ax_lower.set_ylabel("Acquisition")
        ax_upper.tick_params(bottom=False, labelbottom=False)
    else:
        ax_upper = fig.add_subplot()
        ax_upper.set_xlabel(X_name)

    if objective_fn is not None:
        ax_upper.plot(
            X_grid, objective_fn(X_grid), 'k:', alpha=.5, label="Objective fn")
    if mean is not None:
        ax_upper.plot(X_grid, mean, 'b-', label="Prediction")
    if std is not None:
        ax_upper.fill(
            np.concatenate([X_grid, X_grid[::-1]]),
            np.concatenate([mean -1.96*std, (mean + 1.96*std)[::-1]]),
            "p-", alpha=.5, label="95% CI")

    ax_upper.plot(X[:-1], Y[:-1], 'ko', label="Observations")
    ax_upper.plot(X[-1], Y[-1], "ro")
    ax_upper.set_ylabel(Y_name)
    ax_upper.legend(
        loc='lower center', bbox_to_anchor=(.5, 0.97), ncol=3, frameon=False)

    return fig


def _plot_process_2d(
    fig: Figure,
    X: np.ndarray,
    Y: np.ndarray,
    X_grids: List[np.ndarray],
    mean: Optional[np.ndarray] = None,
    acquisition: Optional[np.ndarray] = None,
    objective_fn: Optional[Callable] = None,
    X_names: List[str] = ["x1", "x2"],
    Y_name: str = "y"
) -> Figure:
    Xmeshes = np.meshgrid(X_grids[0], X_grids[1])
    ax = fig.add_subplot(projection="3d")

    if objective_fn is not None:
        ax.plot_wireframe(
            Xmeshes[0], Xmeshes[1], objective_fn(Xmeshes[0], Xmeshes[1]),
            color="black", alpha=0.5, linewidth=0.5, label="Objective fn")
    if mean is not None:
        mean = mean.reshape(X_grids[0].shape[0], X_grids[1].shape[0])
        ax.plot_wireframe(
            Xmeshes[0], Xmeshes[1], mean.T, color="blue", alpha=0.6,
            linewidth=0.5, label="Prediction")
    ax.scatter(X[:-1, 0], X[:-1, 1], Y[:-1], c="black", label="Observations")
    ax.scatter(X[-1, 0], X[-1, 1], Y[-1], c="red")
    if acquisition is not None:
        acquisition = acquisition.reshape(
            X_grids[0].shape[0], X_grids[1].shape[0])
        contf = ax.contourf(
            Xmeshes[0], Xmeshes[1], acquisition.T, zdir="z",
            offset=ax.get_zlim()[0], cmap=cm.jet, levels=100)
        fig.colorbar(contf, pad=0.1, shrink=0.6, label="Acquisition")

    ax.set_xlabel(X_names[0])
    ax.set_ylabel(X_names[1])
    ax.set_zlabel(Y_name)
    ax.grid(False)
    ax.legend(
        loc='lower center', bbox_to_anchor=(.5, 0.97), ncol=3, frameon=False)

    return fig