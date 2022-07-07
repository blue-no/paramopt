<h1 align="center"> Param Opt </h1>
<h4 align="center">Param Opt helps the researcher quickly and easily explore optimal experimental parameters.</h4>

## Table of Contents
* [Overview](#overview)
* [Preparation](#preparation)
* [Quick Start](#quick-start)
    * [Defining Target Parameters](#defining-target-parameters)
    * [Creating Dataset](#creating-dataset)
    * [Preparing GPR Model and Acquisition Function](#preparing-gpr-model-and-acquisition-function)
    * [Optimizing Parameters](#optimizing-parameters)
* [Other Useful Features](#other-useful-features)
    * [GPR with Hyperparameter Auto-adjustment Ability](#gpr-with-hyperparameter-auto-adjustment-ability)
    * [GIF Creation from Plot pngs](#gif-creation-from-plot-pngs)
* [License](#license)

## Overview

<img src=https://user-images.githubusercontent.com/88641432/177723799-e628ba9a-97f9-4bf0-8110-62b7319860d4.png>

Bayesian optimization is used for adjusting process parameters (= experimental parameters), such as instrument settings, chemical formulation rates, hyperparameters for machine learning models, and more.

**Param Opt** is a useful python package that is responsible for not only bayesian model training and prediction, but also reading and writing data and visualizing the optimization process.

## Preparation
Install Param Opt via pip:
```
pip install git+https://github.com/ut-hnl-lab/paramopt.git
```
The following packages are also required:
* Matplotlib
* Natsort
* Numpy
* Pandas
* Pillow
* Scipy
* Scikit-learn

## Quick Start
Here is an example of optimizing a combination of two parameters.

### Defining Target Parameters

<img src=https://user-images.githubusercontent.com/88641432/177726927-ca4f8f7c-3f78-4585-b0f5-da198d4179b8.png width="50%">

Define parameters to be adjusted.
```python
from paramopt.structures import ProcessParameter, ExplorationSpace

param1 = ProcessParameter(name="Heating Temperature", values=[150, 180, 210, 230, 250])
param2 = ProcessParameter(name="Heating Time", values=[10, 20, 40, 80, 150, 220])
```
`name` is the parameter name and `values` is a list of possible values of the parameter.

Then, define a exploration space consisting of the parameters.
```python
space = ExplorationSpace([param1, param2])
```

### Creating Dataset

<img src=https://user-images.githubusercontent.com/88641432/177725196-2f3043e4-31be-4939-ba48-c276f503246c.png width="45%">

Create a dataset consisting of an explanatory variables with `X_names` and objective variables with `Y_names`.
```python
from paramopt.structures import Dataset

dataset = Dataset(X_names=space.names, Y_names="Evaluation")
```
Basically, X_names is passed the parameter namew, and Y_names is passed the name of the evaluations.

The dataset is managed by the `BayesianOptimizer` class described below.

### Preparing GPR Model and Acquisition Function
Use `sklearn.gaussian_process` for the GPR model.
Acquisition functions are provided in this package.
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from paramopt.acquisitions import UCB

model = GaussianProcessRegressor(kernel=RBF(10), normalize_y=True)
acquisition = UCB(1.0)
```

### Optimizing Parameters
Let's optimize parameters in the bayesian optimization loop.
Here is a function that simulates an experiment and returns an evaluation value for a given parameter combination.
```python
def experiment(x1, x2):
    return x1*np.sin(x1*0.05+np.pi) + x2*np.cos(x2*0.05)
```

The optimization flow is as follows:
```python
from pathlib import Path
from paramopt.optimizers.sklearn import BayesianOptimizer

# Define optimizer
bo = BayesianOptimizer(
    workdir=Path.cwd(), exploration_space=space, dataset=dataset, model=model,
    acquisition=acquisition, objective_fn=experiment, random_seed=71)

# For max iterations:
for i in range(15):
    # Get better combination of parameters
    next_x = bo.suggest()
    # Get evaluation score with experiments
    y = experiment(*next_x)
    # Update optimizer
    bo.update(next_x, y, label=f"#{i+1}")
    # Check process with some plots
    bo.plot()
```

## Other Useful Features

### GPR with Hyperparameter Auto-adjustment Ability

<img src=https://user-images.githubusercontent.com/88641432/177728148-57ed7d52-07ec-4c5c-af1c-81afb7440860.png width="65%">

Sometimes GPR does not predict well like this:

<img src=https://user-images.githubusercontent.com/88641432/177728843-dea8cacb-60e5-4fbb-adf1-edeb894ccdde.png width="40%">

In this case, let's replace it with a model that automatically adjusts the hyperparameters.

```python
from paramopt.extensions import AutoHPGPR

def gpr_generator(exp, nro):
    return GaussianProcessRegressor(
        kernel=RBF(length_scale_bounds=(10**-exp, 10**exp)) \
                * ConstantKernel() \
                + WhiteKernel(),
        normalize_y=True, n_restarts_optimizer=nro)

model = AutoHPGPR(
    workdir=Path.cwd(), exploration_space=space, gpr_generator=gpr_generator,
    exp=list(range(1, 6)), nro=list(range(0, 10)))
```

The result is

<img src=https://user-images.githubusercontent.com/88641432/177729186-7dfe1249-8a2c-4ce7-9ec8-393e2b682970.png width="40%">

### GIF Creation from Plot pngs
Create a GIF movie from the obtained plot images

```python
paths = select_images()  # Opens a GUI dialog
create_gif(paths)
```
<img src="https://user-images.githubusercontent.com/88641432/177729552-23194201-8241-4c3f-b814-68e5bd69b4bb.PNG" width="40%"><img src=https://user-images.githubusercontent.com/88641432/177729289-6ab150dd-c487-488f-bb82-d52e94fb77e9.gif width="40%">


# License
MIT license.
