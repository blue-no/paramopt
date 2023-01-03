from .acquisitions import EI, UCB, BaseAcquisition
from .extensions import (AutoHyperparameter, AutoHyperparameterRegressor,
                         UnduplicatedSuggestor, create_gif, select_images)
from .optimizer import BayesianOptimizer
from .parameter import ExplorationSpace

__version__ = '2.0.2'
