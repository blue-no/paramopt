from GPy.kern import *

from paramopt.optimizers.gpyopt import BayesianOptimizer
from utils import f1, f2, search_1d, search_2d


def gpybo1d_ucb():
    bo = BayesianOptimizer(
        savedir='test_gpybo1d-ucb',
        kernel=RBF(input_dim=1, variance=1, lengthscale=0.5) * Bias(input_dim=1, variance=1),
        acqfunc='UCB',
        acquisition_weight=2,
        random_seed=71)
    search_1d(bo, f1)


def gpybo1d_ei():
    bo = BayesianOptimizer(
        savedir='test_gpybo1d-ei',
        kernel=RBF(input_dim=1, variance=1, lengthscale=0.5) * Bias(input_dim=1, variance=1),
        acqfunc='EI',
        acquisition_jitter=0.0,
        acquisition_maximize=True,
        random_seed=71)
    search_1d(bo, f1)


def gpybo2d_ucb():
    bo = BayesianOptimizer(
        savedir='test_gpybo2d-ucb',
        kernel=RBF(input_dim=2, variance=1, lengthscale=5.0) * Bias(input_dim=2, variance=1),
        acqfunc='UCB',
        acquisition_weight=2,
        random_seed=71)
    search_2d(bo, f2)


def gpybo2d_ei():
    bo = BayesianOptimizer(
        savedir='test_gpybo2d-ei',
        kernel=RBF(input_dim=2, variance=1, lengthscale=5.0) * Bias(input_dim=2, variance=1),
        acqfunc='EI',
        acquisition_jitter=0.0,
        acquisition_maximize=True,
        random_seed=71)
    search_2d(bo, f2)


if __name__ == '__main__':
    gpybo1d_ucb()
    gpybo1d_ei()
    gpybo2d_ucb()
    gpybo2d_ei()
