from __future__ import annotations

import datetime
from dataclasses import dataclass


@dataclass
class Metrics:
    acc: float
    accmacro: float
    f1macro: float
    f1micro: float
    # confmx: List[list]


@dataclass
class ExpResults:
    train_metrics: Metrics
    val_metrics: Metrics
    test_metrics: Metrics = None
    start_time: datetime.time = None
    training_time: datetime.timedelta = None
    macs: int = None
    flops: int = None
    params: str = None
    nas_params: str = None


@dataclass
class SklearnBaseConfig:
    nclass: int


@dataclass
class NNSklearnBaseConfig:
    nclass: int
    norm_type: str


@dataclass
class DataBaseConfig:
    data_aug: str = 'RANDOM'
    norm_type: str = 'minmax'


@dataclass
class HyperParm:
    epochs: int = 10
    patience: int = 2
    label_smoothing: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 5e-4
    # nclass: int = 2
    batch_size: int = 64


@dataclass
class TrainerOption:
    no_cuda: bool = False
    accelerator: str = 'gpu'
    devices: int = 1
    precision: int = 32
    auto_bs: bool = False
    profiler: str = ''
    strategy: str = ''
    fast_dev_run: bool = False
    limit_train_batches: float = -1
    limit_val_batches: float = -1
    limit_test_batches: float = -1
