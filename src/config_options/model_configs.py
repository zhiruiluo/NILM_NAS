from __future__ import annotations

import dataclasses
from typing import List

from simple_parsing import Serializable

from .utils import IntegerConvertor
from .utils import LambdaConvertor
from .utils import PowerConvertor


@dataclasses.dataclass
class ModelConfig(Serializable):
    ...


@dataclasses.dataclass
class LightningModel(ModelConfig):
    ...


@dataclasses.dataclass
class NNSklearnModel(ModelConfig):
    ...


@dataclasses.dataclass
class SklearnModel(ModelConfig):
    ...


@dataclasses.dataclass
class EstimatorModel(ModelConfig):
    ...


@dataclasses.dataclass
class ModelConfig_EasyNet(LightningModel):
    nclass: int = 1
    num_stages: int = 1
    in_channels: int = 1
    num_blocks: list[int] = dataclasses.field(default_factory=lambda: [3])
    block_width: list[int] = dataclasses.field(default_factory=lambda: [128])
    bottleneck_ratio: float = 1.0
    group_width: int = 2
    stride: int = 2
    se_ratio: int = 4


@dataclasses.dataclass
class ModelConfig_BasicV2Pad(LightningModel):
    chan_1: int = 32
    chan_2: int = 32
    chan_3: int = 32
    ker_1: int = 3
    ker_2: int = 3
    ker_3: int = 3
    stride_1: int = 2
    stride_2: int = 1
    stride_3: int = 1
    dropout: float = 0.5
    in_channels: int = 1
    nclass: int = 2


@dataclasses.dataclass
class ModelConfig_BasicV2(LightningModel):
    chan_1: int = 32
    chan_2: int = 32
    chan_3: int = 32
    ker_1: int = 3
    ker_2: int = 3
    ker_3: int = 3
    stride_1: int = 2
    stride_2: int = 1
    stride_3: int = 1
    dropout: float = 0.5
    in_channels: int = 1
    nclass: int = 2


@dataclasses.dataclass
class ModelConfig_BasicV3_Pool(LightningModel):
    chan_1: int = 32
    chan_2: int = 32
    chan_3: int = 32
    ker_1: int = 3
    ker_2: int = 3
    ker_3: int = 3
    stride_1: int = 2
    stride_2: int = 1
    stride_3: int = 1
    dropout: float = 0.5
    in_channels: int = 1
    nclass: int = 2


@dataclasses.dataclass
class ModelConfig_BitcnNILM(LightningModel):
    nclass: int = 2
    in_chan: int = 1
    out_chan: int = 128
    ker_size: int = 3


@dataclasses.dataclass
class ModelConfig_TSNet(LightningModel):
    nclass: int = 3
    n_phases: int = 3
    bit_string: str = '010100011101001010001110100101000111010'
    in_channels: int = 1
    out_channels: int = 32
    dropout: float = 0.5


@dataclasses.dataclass
class ModelConfig_RF(SklearnModel, Serializable):
    n_estimators: int = 100
    criterion: str = 'gini'
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    nclass: int = 2
