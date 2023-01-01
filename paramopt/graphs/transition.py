from typing import List, Optional, Tuple

import numpy as np
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import MinMaxScaler

COLORS = list(TABLEAU_COLORS.keys())
MARKERS = ["o", "s", "^", "D", "v", "*"]


def plot_transition(
    fig: 'Figure',
    X: 'np.ndarray',
    y: 'np.ndarray',
    axis_values: List['np.ndarray'],
    x_names: List[str],
    y_names: List[str],
    y_bounds: Optional[Tuple[int]] = None
) -> 'Figure':

    # Axes generation and initial settings
    ax_left = fig.add_subplot()
    ax_right = ax_left.twinx()
    if y_bounds is not None:
        ax_left.set_ylim(y_bounds)
    ax_left.set_xlabel("Iteration")
    ax_left.set_ylabel("Objective Score")
    ax_right.set_ylabel("Parameter Value (Normalized)", labelpad=10)
    ax_left.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_left.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

    X = np.atleast_2d(X)
    y = np.atleast_2d(y)

    # Evaluation score and parameter value plot
    for i in range(y.shape[1]):
        mi, ci = int(i%len(MARKERS)), int(i%len(COLORS))
        ax_left.plot(
            y[:, i], f'-{MARKERS[mi]}', label=y_names[i], color=COLORS[ci])
    for j in range(X.shape[1]):
        X_vec = X[:, j:j+1]
        scaler = MinMaxScaler().fit(np.atleast_2d(axis_values[j]).T)
        scaled_X_vec = scaler.transform(X_vec)
        mj, cj = int((i+j+1)%len(MARKERS)), int((i+j+1)%len(COLORS))
        ax_right.plot(
            scaled_X_vec, MARKERS[mj], label=x_names[j], color=COLORS[cj])

    # Additional settings
    hl, ll = ax_left.get_legend_handles_labels()
    hr, lr = ax_right.get_legend_handles_labels()
    ax_left.legend(
        hl+hr, ll+lr, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=3)

    fig.tight_layout()
    return fig
