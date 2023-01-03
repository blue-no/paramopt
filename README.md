# ParamOpt
<img src="https://img.shields.io/badge/version-v2.0.2-blue"> <img src="https://img.shields.io/badge/coverage-93%25-green">

Python Library for Easy Bayesian Optimization.

## Install
```
pip install git+https://github.com/blue-no/paramopt.git
```

## Demo - 2D Exploration
```Python
from pathlib import Path
from paramopt import UCB, BayesianOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


folder = Path('demo_2d')
folder.mkdir(exist_ok=True)

def obj_func(X):
    return -(X[0]-5)**2-(X[1]-50)**2/50

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
    for i in range(10):
        y = obj_func(next_params)
        optimizer.update(X=next_params, y=y, label=i)
        optimizer.save_history(folder.joinpath('history.csv'))
        next_params = optimizer.suggest()
        optimizer.plot_distribution(save_as=folder.joinpath(f'dist-{i}.png'))
        optimizer.plot_transition(save_as=folder.joinpath(f'trans-{i}.png'))
```

Result:

<img src="https://user-images.githubusercontent.com/88641432/210295005-c5b22fe9-7d34-4da1-abad-57872749aa48.gif" width="380px"><img src="https://user-images.githubusercontent.com/88641432/210295012-b5359822-733f-4fc0-b2a8-bda8d94b9b58.gif" width="380px">

**Author:** Kota AONO  
**License:** MIT License
