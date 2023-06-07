from __future__ import annotations

import dataclasses

from simple_parsing import Serializable
from typing import List, Literal

@dataclasses.dataclass
class DatasetConfig(Serializable):
    ...


@dataclasses.dataclass
class DataBaseConfig(Serializable):
    data_aug: str = "RANDOM"
    norm_type: str = "minmax"


@dataclasses.dataclass
class DatasetConfig_REDD(DatasetConfig):
    appliances: List[str] = dataclasses.field(
        default_factory=lambda: [
            "refrigerator",
            "microwave",
            "dishwasher",
            "washer_dryer",
        ],
    )
    house_no: int = 1
    splits: str = "3:1:1"
    index_buffer_flag: bool = True


@dataclasses.dataclass
class DatasetConfig_REDD_Bitcn(DatasetConfig):
    appliance: str = "microwave"
    win_size: int = 600
    stride: int = 60


@dataclasses.dataclass
class DatasetConfig_REDD_multilabel(DatasetConfig):
    appliances: List[str] = dataclasses.field(
        default_factory=lambda: [
            "refrigerator",
            "microwave",
            "dishwasher",
            "washer_dryer",
        ],
    )
    house_no: int = 1
    splits: str = "3:1:1"
    index_buffer_flag: bool = True
    win_size: int = 600
    stride: int = 60
    combine_mains: bool = False
    imbalance_sampler: bool = False


@dataclasses.dataclass
class DatasetConfig_REDD_ML_multihouse(DatasetConfig):
    appliances: List[str] = dataclasses.field(
        default_factory=lambda: [
            "refrigerator",
            "microwave",
            "dishwasher",
            "washer_dryer",
        ],
    )
    train_house_no: List[int] = dataclasses.field(default_factory=lambda: [2,3,5])
    test_house_no: List[int] = dataclasses.field(default_factory=lambda: [1])
    val_ratio: float = 0.2
    index_buffer_flag: bool = True
    win_size: int = 600
    stride: int = 60
    combine_mains: bool = False
    imbalance_sampler: bool = False
    drop_na_how: Literal['any', 'all'] = 'any'