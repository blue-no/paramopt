import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np


class ExplorationSpace:

    def __init__(
        self,
        params: Union[Tuple[str, str, list], Tuple[str, list]]
    ) -> None:
        self._params = []
        self.__load_params(params=params)

    @property
    def ndim(self) -> int:
        return len(self._params)

    @property
    def axis_names_with_unit(self) -> List[str]:
        labels = []
        for (name, unit, _) in self._params:
            if unit is not None:
                label = f'{name} [{str(unit)}]'
            else:
                label = f'{name}'
            labels.append(label)
        return labels

    @property
    def axis_names(self) -> List[str]:
        names = []
        for (name, _, _) in self._params:
            names.append(name)
        return names

    def axis_values(self) -> List[float]:
        axis_values = []
        for (_, _, values) in self._params:
            axis_values.append(values)
        return axis_values

    def grid_axis_values(self, n_splits: int = 100) -> List[float]:
        grid_axis_values = []
        for (_, _, values) in self._params:
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
            json.dump(self._params, f, indent=2)

    def __load_params(self, params) -> None:
        for param in params:
            if len(param) == 2:
                name = str(param[0])
                unit = None
                values = list(param[1])
            elif len(param) == 3:
                name = str(param[0])
                unit = str(param[1]) if param[1] is not None else None
                values = list(param[2])
            else:
                raise ValueError('invalid format')

            if len(values) == 0:
                raise ValueError('Value cannot be empty')

            self._params.append((name, unit, values))
