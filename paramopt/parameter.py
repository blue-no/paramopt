import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np


class ExplorationSpace:

    def __init__(self, params: Union[List[Any], Dict[str, dict]]) -> None:
        self.__params = {}
        self.__load_params(params=params)

    @property
    def ndim(self) -> int:
        return len(self.__params)

    @property
    def axis_names_with_unit(self) -> List[str]:
        labels = []
        for name, values in self.__params.items():
            unit = values['unit']
            if unit is not None:
                label = f'{name} [{str(unit)}]'
            else:
                label = f'{name}'
            labels.append(label)
        return labels

    @property
    def axis_names(self) -> List[str]:
        return list(self.__params.keys())

    def axis_values(self) -> List[float]:
        axis_values = []
        for values in self.__params.values():
            if isinstance(values, (list, tuple)):
                axis_values.append(values)
                continue
            axis_values.append(values['values'])
        return tuple(axis_values)

    def grid_axis_values(self, n_splits: int = 100) -> List[float]:
        grid_axis_values = []
        for values in self.axis_values():
            if len(values) == 1:
                grid_values = values.copy()
            else:
                vmax, vmin = np.max(values), np.min(values)
                grid_values = np.linspace(vmin, vmax, n_splits)
            grid_axis_values.append(grid_values)
        return grid_axis_values

    def points(self) -> 'np.ndarray':
        mesh = np.array(np.meshgrid(*self.axis_values()))
        return mesh.T.reshape(-1, self.ndim)

    def grid_points(self, n_splits: int = 100) -> 'np.ndarray':
        mesh = np.array(np.meshgrid(*self.grid_axis_values(n_splits=n_splits)))
        return mesh.T.reshape(-1, self.ndim)

    @classmethod
    def load(cls, fp: Union[Path, str]) -> None:
        fp_ = Path(fp)
        with fp_.open('r') as f:
            params = json.load(f)
        return cls(params=params)

    def dump(self, fp: Union[Path, str]) -> None:
        fp_ = Path(fp)
        with fp_.open('w') as f:
            json.dump(self.__params, f, indent=2)

    def __load_params(self, params: Union[List[Any], Dict[str, dict]]) -> None:
        for name, values in params.items():
            if isinstance(values, (list, tuple)):
                if len(values) == 0:
                    raise ValueError('Value cannot be empty')
                params[name] = {'values': values, 'unit': None}
                continue

            if not isinstance(values, dict):
                raise ValueError(f'Value of {name} must be list, tuple or dict')

            keys = values.keys()
            if 'values' not in keys:
                raise ValueError(f'{name} must have key "values"')
            if 'unit' not in keys:
                values['unit'] = None
            if not isinstance(values['values'], (list, tuple)):
                raise ValueError(f'"values" of {name} must be list or tuple')
            if len(values['values']) == 0:
                raise ValueError('"values" cannot be empty')
        self.__params = params
