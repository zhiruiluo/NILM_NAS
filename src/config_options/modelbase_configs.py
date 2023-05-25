from __future__ import annotations

import dataclasses

from simple_parsing import Serializable


@dataclasses.dataclass
class ModelBaseConfig(Serializable):
    batch_size: int = 32
    val_batch_size: int = 128
    test_batch_size: int = 128


@dataclasses.dataclass
class SklearnBaseConfig(ModelBaseConfig):
    # batch_size: int = 64
    label_mode: str = "multilabel"


@dataclasses.dataclass
class HyperParm(ModelBaseConfig):
    epochs: int = 20
    patience: int = 20
    label_smoothing: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 5e-4
    # batch_size: int = 32
    # val_batch_size: int = 128
    # test_batch_size: int = 128
    label_mode: str = "multilabel"


@dataclasses.dataclass
class NNSklearnBaseConfig(HyperParm):
    norm_type: str = "minmax"
    # batch_size: int = 64


@dataclasses.dataclass
class NNSklearnGridBaseConfig(HyperParm):
    norm_type: str = "minmax"
    # batch_size: int = 64


@dataclasses.dataclass
class EsitmatorBaseConfig(ModelBaseConfig):
    # batch_size: int = 64
    label_mode: str = "multilabel"
