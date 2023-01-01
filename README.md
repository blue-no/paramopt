# ParamOpt
Python Library for Easy Bayesian Optimization.

## Install
```
pip install git+https://github.com/blue-no/paramopt.git
```

## Demo - 2D Exploration
```Python
from pathlib import Path
from paramopt import UCB, BayesianOptimizer, ExplorationSpace
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


folder = Path('demo_2d')
folder.mkdir(exist_ok=True)

def obj_func(X):
    return -(X[0]-5)**2-(X[1]-50)**2/50

optimizer = BayesianOptimizer(
    regressor=GaussianProcessRegressor(
        kernel=C()*RBF(length_scale_bounds='fixed'), normalize_y=True),
    exp_space=ExplorationSpace({
        'distance':
            {'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             'unit': 'mm'},
        'temperature':
            {'values': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
             'unit': '°C'}
    }),
    eval_name='quality',
    acq_func=UCB(c=2.0),
    obj_func=obj_func,
    normalize_X=True)

for i in range(10):
    next_params = optimizer.suggest()
    y = obj_func(next_params)
    optimizer.update(X=next_params, y=y, label=i)
    optimizer.plot_distribution(save_as=folder.joinpath(f'dist-{i}.png'))
    optimizer.plot_transition(save_as=folder.joinpath(f'trans-{i}.png'))
    optimizer.save_history(folder.joinpath('history.csv'))
```

Result:

<img src="https://user-images.githubusercontent.com/88641432/210160117-516719d3-4011-43b3-ab26-4e1dc4af977c.gif" width="380px"><img src="https://user-images.githubusercontent.com/88641432/210160119-ed57d822-5943-4dac-a901-6e67ad8442b7.gif" width="380px">

**Author:** Kota AONO  
**License:** MIT License
