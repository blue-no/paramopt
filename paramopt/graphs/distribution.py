import warnings
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from matplotlib import gridspec
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter


def plot_distribution_1d(
    fig: 'Figure',
    X: 'np.ndarray',
    y: 'np.ndarray',
    axis_values: 'np.ndarray',
    mean: Optional['np.ndarray'] = None,
    std: Optional['np.ndarray'] = None,
    acq: Optional['np.ndarray'] = None,
    X_next: Optional[Any] = None,
    obj_func: Optional[Callable] = None,
    x_label: str = "x",
    y_label: str = "y",
    acq_label: str = "y",
    *args: Any,
    **kwargs: Any
) -> 'Figure':

    # Axes generation and initial settings
    if acq is not None:
        gs = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[5, 2])
        gs.update(hspace=0.25)
        ax_upper = fig.add_subplot(gs[0])
        ax_lower = fig.add_subplot(gs[1], sharex=ax_upper)
        ax_lower.set_ymargin(0.4)
        ax_lower.set_xlim(np.min(axis_values), np.max(axis_values))
        ax_lower.set_xlabel(x_label)
        ax_lower.set_ylabel(acq_label)
        ax_lower.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_lower.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

    else:
        ax_upper = fig.add_subplot()
        ax_lower = None
        ax_upper.set_xlabel(x_label)
    ax_upper.set_ymargin(0.4)
    ax_upper.set_xlim(axis_values[0], axis_values[-1])
    ax_upper.set_ylabel(y_label)
    ax_upper.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_upper.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

    # Observation
    ax_upper.plot(X[:-1], y[:-1], 'ko', label="Observation", zorder=3)
    ax_upper.plot(X[-1], y[-1], 'ro', zorder=3)

    # Posterior mean
    if mean is not None:
        ax_upper.plot(axis_values, mean, 'b-', label="Mean")

    # Posterior uncertainty
    if std is not None:
        ax_upper.fill(
            np.concatenate([axis_values, axis_values[::-1]]),
            np.concatenate([mean -1.96*std, (mean + 1.96*std)[::-1]]),
            "p-", alpha=.5, label="95% confidence interval")

    # Objective function
    if obj_func is not None:
        ax_upper.plot(
            axis_values, obj_func(axis_values), 'k:', alpha=.5,
            label="Objective function")

    # Acquisition function
    if acq is not None:
        ax_lower.plot(axis_values, acq, 'g-', label="Acquisition function")

    # Next location
    if X_next is not None:
        ax_upper.axvline(X_next, color="red", linewidth=0.8)
        if ax_lower is not None:
            ax_lower.axvline(X_next, color="red", linewidth=0.8,
            label="Acquisition max")

    # Additional axes settings
    ax_upper.legend(loc='lower left')
    if ax_lower is not None:
        ax_lower.legend(loc='lower left')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        fig.tight_layout()
    return fig


def plot_distribution_2d(
    fig: 'Figure',
    X: 'np.ndarray',
    y: 'np.ndarray',
    axis_values: List['np.ndarray'],
    mean: Optional['np.ndarray'] = None,
    std: Optional['np.ndarray'] = None,
    acq: Optional['np.ndarray'] = None,
    X_next: Optional[Tuple[Any]] = None,
    obj_func: Optional[Callable] = None,
    x_label: str = "x1",
    y_label: str = "x2",
    z_label: str = "y",
    acq_label: Optional[str] = None,
    *args: Any,
    **kwargs: Any
) -> 'Figure':

    # Axes generation and initial settings
    Xmeshes = np.meshgrid(axis_values[0], axis_values[1])
    ax = fig.add_subplot(projection="3d")
    ax.set_zmargin(0.4)
    ax.set_xlim(np.min(axis_values[0]), np.max(axis_values[0]))
    ax.set_ylim(np.min(axis_values[1]), np.max(axis_values[1]))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.ticklabel_format(style="sci", axis="z", scilimits=(-2, 2), useMathText=True)

    # Observation plot
    ax.scatter(X[:-1, 0], X[:-1, 1], y[:-1], color="black", label="Observation")
    ax.scatter(X[-1, 0], X[-1, 1], y[-1], color="red")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=np.VisibleDeprecationWarning)
        # Objective function plot
        if obj_func is not None:
            obj = obj_func(Xmeshes).reshape(len(axis_values[0]), len(axis_values[1]))
            ax.plot_wireframe(
                Xmeshes[0], Xmeshes[1], obj,
                color="black", alpha=0.5, linewidth=0.5, label="Objective function")

        # Posterior mean plot
        if mean is not None:
            mean = mean.reshape(len(axis_values[0]), len(axis_values[1])).T
            ax.plot_wireframe(
                Xmeshes[0], Xmeshes[1], mean, color="blue", alpha=0.5,
                linewidth=0.5, label="Mean")

        z_from, z_to = ax.get_zlim()
        for i in range(X.shape[0]-1):
            ax.plot(
                [X[i, 0]]*2, [X[i, 1]]*2, [z_from, y[i]], color='black',
                linewidth=0.8, linestyle='--', zorder=100)
        ax.plot(
            [X[-1, 0]]*2, [X[-1, 1]]*2, [z_from, y[-1]], color='red', linestyle='--',
            linewidth=0.8, zorder=100)

        # Acquisition function plot
        if acq is not None:
            acq = acq.reshape(
                len(axis_values[0]), len(axis_values[1]))
            contf = ax.contourf(
                Xmeshes[0], Xmeshes[1], acq.T, zdir="z",
                offset=z_from, levels=100, alpha=1.0)
            cb = fig.colorbar(
                contf, pad=0.11, shrink=0.7,
                label="Acquisition function"
                    + (f" ({acq_label})" if acq_label is not None else ""))
            cb.formatter.set_powerlimits((0, 0))
            cb.formatter.set_useMathText(True)
            ax.set_zlim(z_from, z_to)

        # Next location plot
        if X_next is not None:
            ax.plot(
                [X_next[0]]*2, [X_next[1]]*2, [z_from, z_to], color='red', linestyle='-',
                linewidth=0.8, zorder=100, label="Acquisition max")

        # Additional axes settings
        leg = ax.legend(loc='upper left')
        for line in leg.get_lines():
            line.set_linewidth(1.5)
            line.set_alpha(1.0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        fig.tight_layout()
    return fig
