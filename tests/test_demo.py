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
            kernel=C()*RBF(length_scale_bounds='fixed'), normalize_y=True),
        exp_space=ExplorationSpace([
            ('distance', 'mm', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            ('temperature', '°C', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])]),
        eval_name='quality',
        acq_func=UCB(c=2.0),
        obj_func=obj_func,
        normalize_X=True)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for i in range(10):
            next_params = optimizer.suggest()
            y = obj_func(next_params)
            optimizer.update(X=next_params, y=y, label=i)
            optimizer.plot_distribution(save_as=folder.joinpath(f'dist-{i}.png'))
            optimizer.plot_transition(save_as=folder.joinpath(f'trans-{i}.png'))
            optimizer.save_history(folder.joinpath('history.csv'))
            plt.close('all')
