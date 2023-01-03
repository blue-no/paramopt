import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import ConvergenceWarning

from paramopt import UCB, BayesianOptimizer, ExplorationSpace

folder = Path('tests', 'result', 'demo_2d')
folder.mkdir(exist_ok=True)


def obj_func(X):
    return -(X[0]-5)**2-(X[1]-50)**2/50


def test_demo():
    optimizer = BayesianOptimizer(
        regressor=GaussianProcessRegressor(
            kernel=C()*RBF(length_scale_bounds='fixed'),
            normalize_y=True),
        exp_space=[('Distance', 'mm', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                   ('Temperature', '°C', [0, 20, 40, 50, 60, 80, 100])],
        eval_name='Quality',
        acq_func=UCB(c=2.0),
        obj_func=obj_func,
        normalize_X=True)

    next_params = optimizer.suggest()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for i in range(10):
            y = obj_func(next_params)
            optimizer.update(X=next_params, y=y, label=i)
            optimizer.save_history(folder.joinpath('history.csv'))
            next_params = optimizer.suggest()
            optimizer.plot_distribution(save_as=folder.joinpath(f'dist-{i}.png'))
            optimizer.plot_transition(save_as=folder.joinpath(f'trans-{i}.png'))
            plt.close('all')
