from __future__ import annotations

import functools
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler


def apply_transform(
    train: dict[str, list], val: dict[str, list], test: dict[str, list], t,
):
    for f in ['mains_1', 'mains_2']:
        train[f] = t.fit_transform(train[f]).tolist()
        val[f] = t.transform(val[f]).tolist()
        test[f] = t.transform(test[f]).tolist()
    return train, val, test


class MinMax:
    def __init__(self, train_set: dict[str, list], keys: list[str]) -> None:
        scalers: dict[str, MinMaxScaler] = {}
        for k in keys:
            scalers[k] = MinMaxScaler()
            x = train_set[k].to_numpy().reshape(-1, 1)
            scalers[k].fit_transform(x)

        self.scalers = scalers

    def __call__(self, sample: Any) -> Any:
        for i, k in enumerate(self.scalers.keys()):
            scaler = self.scalers[k]
            sample['input'][i] = scaler.transform(
                sample['input'][i].reshape(1, -1),
            ).reshape(-1)
        return sample


def power(x, np_arange):
    return np.sum(np_arange * x)


class ApplyPowerLabel:
    def __init__(self) -> None:
        self.power_func_partial = None

    def __call__(self, sample) -> Any:
        if self.power_func_partial is None:
            self.power_func_partial = functools.partial(
                power, np_arange=np.arange(len(sample['target'])) ** 2,
            )
        sample['powerlabel'] = np.apply_along_axis(
            self.power_func_partial, 0, sample['target'],
        )
        return sample


class NumpyToTensor:
    def __init__(self) -> None:
        pass

    def __call__(self, sample):
        sample['input'] = torch.from_numpy(
            sample['input']).type(dtype=torch.float32)
        sample['target'] = torch.from_numpy(
            sample['target']).type(dtype=torch.float32)
        # sample['target'] = torch.from_numpy(sample['target']).type(dtype=torch.float32)
        return sample
