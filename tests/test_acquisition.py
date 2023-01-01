import numpy as np
import pytest

from paramopt.acquisitions import EI, UCB, BaseAcquisition

N_OBS = 5
N_AXVALS = 10


def _create_preds():
    mean = np.random.random(size=(N_AXVALS, 1))
    std = np.random.random(size=(N_AXVALS, 1))
    x = np.random.random(size=(N_OBS, 1))
    y = np.random.random(size=(N_OBS, 1))
    return (mean, std, x, y)


def test_base():
    acq = BaseAcquisition()
    with pytest.raises(NotImplementedError):
        acq(*_create_preds())


def test_ucb():
    acq = UCB(1.0)
    acq(*_create_preds())


def test_ei():
    acq = EI(1.0)
    acq(*_create_preds())
