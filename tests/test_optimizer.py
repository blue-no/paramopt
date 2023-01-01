import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import ConvergenceWarning

from paramopt import (UCB, AutoHyperparameterRegressor, BayesianOptimizer,
                      ExplorationSpace, UnduplicatedSuggestor)

N_TRIAL = 10
INIT_PARAMS = [1, 10]
X1_VALS = list(range(0, 11, 1))
X2_VALS = list(range(0, 101, 10))
EXP_SPACE = [('x1', 'unit1', X1_VALS),
             ('x2', 'unit2', X2_VALS)]
EVAL_NAME = ('eval', 'unit3')
C_VAL = 2.0

folder = Path('tests', 'result', 'optimizer')
folder.mkdir(exist_ok=True)


def obj_func(X):
    return -(X[0]-5)**2-(X[1]-50)**2/50


def test_io():
    optimizer = BayesianOptimizer(
        regressor=GaussianProcessRegressor(),
        exp_space=ExplorationSpace(EXP_SPACE),
        eval_name=EVAL_NAME,
        acq_func=None,
    )
    df = pd.DataFrame(
        [[0, 1, 2, '3'], [0, 10, 20, '30']],
        columns=['x1', 'x2', 'eval', 'label'])
    optimizer.load_history(df)

    assert np.array_equal(
        df.iloc[:, 0:2].to_numpy(), optimizer._BayesianOptimizer__X)
    assert np.array_equal(
        df.iloc[:, 2:3].to_numpy(), optimizer._BayesianOptimizer__y)
    assert np.array_equal(
        df.iloc[:, 3].to_list(), optimizer._BayesianOptimizer__labels)

    optimizer.save_history(folder.joinpath('io_history.csv'))

    optimizer2 = BayesianOptimizer(
        regressor=GaussianProcessRegressor(),
        exp_space=ExplorationSpace(EXP_SPACE),
        eval_name=EVAL_NAME,
        acq_func=None,
    )
    optimizer2.load_history(folder.joinpath('io_history.csv'))

    assert np.array_equal(
        df.iloc[:, 0:2].to_numpy(), optimizer2._BayesianOptimizer__X)
    assert np.array_equal(
        df.iloc[:, 2:3].to_numpy(), optimizer2._BayesianOptimizer__y)
    assert np.array_equal(
        df.iloc[:, 3].to_list(), optimizer2._BayesianOptimizer__labels)


def test_optimization():
    optimizer = BayesianOptimizer(
        regressor=GaussianProcessRegressor(
            kernel=C()*RBF(10, 'fixed'),
            normalize_y=True
        ),
        exp_space=ExplorationSpace(EXP_SPACE),
        eval_name=EVAL_NAME,
        acq_func=UCB(c=C_VAL),
        obj_func=obj_func,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for i in range(N_TRIAL):
            next_params = optimizer.suggest()
            y = obj_func(next_params)
            optimizer.update(next_params, y, label=f'{i}')
            optimizer.plot_distribution(save_as=folder.joinpath(f'normal_dist-{i}.png'))
            optimizer.plot_transition(save_as=folder.joinpath(f'normal_trans-{i}.png'))
            optimizer.save_history(folder.joinpath('normal_history.csv'))
            plt.close('all')


def test_autohp_optimization():
    regressor = AutoHyperparameterRegressor(
        hyperparams={
            'length_scale': list(range(10, 110, 10)),
            'n_restarts_optimizer': list(range(0, 11, 1))
        },
        regressor_factory=lambda autohp:
            GaussianProcessRegressor(
                kernel=C()*RBF(autohp.select('length_scale'), 'fixed'),
                normalize_y=True,
                n_restarts_optimizer=autohp.select('n_restarts_optimizer')
            )
    )

    optimizer = BayesianOptimizer(
        regressor=regressor,
        exp_space=ExplorationSpace(EXP_SPACE),
        eval_name=EVAL_NAME,
        acq_func=UCB(c=C_VAL),
        obj_func=obj_func,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for i in range(N_TRIAL):
            next_params = optimizer.suggest()
            y = obj_func(next_params)
            optimizer.update(next_params, y, label=f'{i}')
            optimizer.plot_distribution(save_as=folder.joinpath(f'autohp_dist-{i}.png'))
            optimizer.plot_transition(save_as=folder.joinpath(f'autohp_trans-{i}.png'))
            optimizer.save_history(folder.joinpath('autohp_history.csv'))
            regressor.dump_hp_history(folder.joinpath('hp.csv'))
            plt.close('all')


def test_undup_optimization():
    optimizer = BayesianOptimizer(
        regressor=GaussianProcessRegressor(
            kernel=C()*RBF(10, 'fixed'),
            normalize_y=True
        ),
        exp_space=ExplorationSpace(EXP_SPACE),
        eval_name=EVAL_NAME,
        acq_func=UCB(c=C_VAL),
        obj_func=obj_func,
        sug_func=UnduplicatedSuggestor().argmax
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for i in range(N_TRIAL):
            next_params = optimizer.suggest()
            y = obj_func(next_params)
            optimizer.update(next_params, y, label=f'{i}')
            optimizer.plot_distribution(save_as=folder.joinpath(f'undup_dist-{i}.png'))
            optimizer.plot_transition(save_as=folder.joinpath(f'undup_trans-{i}.png'))
            optimizer.save_history(folder.joinpath('undup_history.csv'))
            plt.close('all')
