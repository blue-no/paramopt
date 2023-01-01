from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paramopt.graphs import (plot_distribution_1d, plot_distribution_2d,
                             plot_transition)

N_OBS = 5
N_AXVALS = 10


folder = Path('tests', 'result', 'graph')
folder.mkdir(exist_ok=True)


def _create_dist(ndim):
    X = np.random.random(size=(N_OBS, ndim))
    Y = np.random.random(size=(N_OBS, 1))
    if ndim == 1:
        axis_values = np.linspace(0, 1, N_AXVALS)
    else:
        axis_values = [np.linspace(0, 1, N_AXVALS) for _ in range(ndim)]
    mean = np.random.random(size=(N_AXVALS**ndim, 1))
    std = np.random.random(size=(N_AXVALS**ndim, 1))
    acq = np.random.random(size=(N_AXVALS**ndim, 1))
    X_next = np.random.rand(ndim)
    obj_func = lambda *args: np.random.random(size=(N_AXVALS**ndim, 1))
    return (X, Y, axis_values, mean, std, acq, X_next, obj_func)


def _create_trans(ndim):
    X = np.random.random(size=(N_OBS, ndim))
    Y = np.random.random(size=(N_OBS, 1))
    if ndim == 1:
        axis_values = np.linspace(0, 1, N_AXVALS)
    else:
        axis_values = [np.linspace(0, 1, N_AXVALS) for _ in range(ndim)]
    X_names = [f'x{i}' for i in range(ndim)]
    Y_names = ['y']
    return (X, Y, axis_values, X_names, Y_names)


def test_dist_1d():
    fig = plot_distribution_1d(
        plt.figure(),
        *_create_dist(ndim=1)
    )
    fig.savefig(folder.joinpath('dist_1d.png').as_posix())
    plt.close('all')


def test_dist_2d():
    fig = plot_distribution_2d(
        plt.figure(),
        *_create_dist(ndim=2)
    )
    fig.savefig(folder.joinpath('dist_2d.png').as_posix())
    plt.close('all')


def test_trans_3d():
    fig = plot_transition(
        plt.figure(),
        *_create_trans(ndim=3)
    )
    fig.savefig(folder.joinpath('trans_3d.png').as_posix())
    plt.close('all')
