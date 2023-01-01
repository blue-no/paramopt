import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter('ignore', PendingDeprecationWarning)
    from GPy.models import GPRegression

from sklearn.gaussian_process import GaussianProcessRegressor

from paramopt.imples import BaseImple
from paramopt.imples.gpy import GpyImple
from paramopt.imples.sklearn import SklearnImple

N = 10


def _create_data():
    x = np.random.random(size=(N, 1))
    y = np.random.random(size=(N, 1))
    return (x, y)


def test_base_imple():
    imple = BaseImple()
    with pytest.raises(NotImplementedError):
        imple.fit(*_create_data())
    with pytest.raises(NotImplementedError):
        imple.predict(_create_data()[0])


def test_sklearn_imple():
    imple = SklearnImple(
        regressor=GaussianProcessRegressor()
    )
    imple.fit(*_create_data())
    imple.predict(_create_data()[0])


def test_gpy_imple():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        imple = GpyImple(
            regressor=GPRegression(X=np.empty((1, 1)), Y=np.empty((1, 1)))
        )
    imple.fit(*_create_data())
    imple.predict(_create_data()[0])
