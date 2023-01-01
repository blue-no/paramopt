from pathlib import Path

import numpy as np
import pytest

from paramopt.parameter import ExplorationSpace

N_SPLITS = 100
X1_VALS = list(range(0, 11, 1))
X1_GVALS = np.linspace(0, 10, N_SPLITS)
X2_VALS = list(range(0, 101, 10))
X2_GVALS = np.linspace(0, 100, N_SPLITS)
POINTS = np.array([[i, j] for i in X1_VALS for j in X2_VALS])
GPOINTS = np.array([[i, j] for i in X1_GVALS for j in X2_GVALS])

folder = Path('tests', 'result', 'parameter')
folder.mkdir(exist_ok=True)


def test_exp_space():
    with pytest.raises(ValueError):
        ExplorationSpace({'x0': []})

    with pytest.raises(ValueError):
        ExplorationSpace({'x0': 0})

    with pytest.raises(ValueError):
        ExplorationSpace({'x0': {}})

    with pytest.raises(ValueError):
        ExplorationSpace({'x0': {'values': []}})

    with pytest.raises(ValueError):
        ExplorationSpace({'x0': {'values': 0}})

    with pytest.raises(ValueError):
        ExplorationSpace({'x0': {'values': {}}})

    space = ExplorationSpace(
        {'x1': X1_VALS,
         'x2': {'values': X2_VALS,
                'unit': 'K'}})

    assert space.ndim == 2
    assert space.axis_names[0] == 'x1'
    assert space.axis_names[1] == 'x2'
    assert space.axis_names_with_unit[0] == 'x1'
    assert space.axis_names_with_unit[1] == 'x2 [K]'
    assert space.axis_values()[0] == X1_VALS
    assert space.axis_values()[1] == X2_VALS
    assert np.array_equal(
        space.grid_axis_values(n_splits=N_SPLITS)[0], X1_GVALS)
    assert np.array_equal(
        space.grid_axis_values(n_splits=N_SPLITS)[1], X2_GVALS)
    assert np.array_equal(space.points(), POINTS)
    assert np.array_equal(space.grid_points(), GPOINTS)
    space.dump(folder.joinpath('space.json'))
    space2 = ExplorationSpace.load(folder.joinpath('space.json'))

    assert space._ExplorationSpace__params == space2._ExplorationSpace__params
