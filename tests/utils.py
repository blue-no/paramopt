from typing import Callable
import numpy as np

from paramopt.optimizers import BaseOptimizer


def f1(x):
    return x * np.sin(x) + 10


def f2(x1, x2):
    return (x1 * np.sin(x1*0.05+np.pi) + x2 * np.cos(x2*0.05)) * 0.1 + 10


def search_1d(
    gpr: BaseOptimizer, f: Callable, prefix: str = '', noisy: bool = True
) -> None:
    gpr.add_parameter('param', np.arange(0, 10.5, 0.5))
    gpr.fit(0, f(0), label=prefix+str(0))
    gpr.fit(1, f(1), label=prefix+str(0))
    gpr.fit(10, f(10), label=prefix+str(0))
    gpr.plot(objective_fn=f, overwrite=True)

    for i in range(10):
        next_x, = gpr.next()
        y = f(next_x)
        if noisy:
            y += np.random.normal(scale=0.1)
        gpr.fit(next_x, y, label=prefix+str(i+1))
        gpr.plot(objective_fn=f, overwrite=True)


def search_2d(
    gpr: BaseOptimizer, f: Callable, prefix: str = '', noisy: bool = True
) -> None:
    gpr.add_parameter('param1', np.arange(180, 285, 5))
    gpr.add_parameter('param2', np.arange(10, 220, 10))
    gpr.fit((180, 10), f(180, 10), label=prefix+str(0))
    gpr.fit((185, 15), f(185, 15), label=prefix+str(0))
    gpr.fit((180, 210), f(180, 210), label=prefix+str(0))
    gpr.fit((280, 10), f(280, 10), label=prefix+str(0))
    gpr.fit((280, 210), f(280, 210), label=prefix+str(0))
    gpr.plot(objective_fn=f, overwrite=True)

    for i in range(15):
        next_x = gpr.next()
        y = f(*next_x)
        if noisy:
            y += np.random.normal(scale=1.0)
        gpr.fit(next_x, y, label=prefix+str(i+1))
        gpr.plot(objective_fn=f, overwrite=True)