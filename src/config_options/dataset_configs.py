import dataclasses
from typing import List
from simple_parsing import Serializable

@dataclasses.dataclass
class DatasetConfig(Serializable):
    ...

@dataclasses.dataclass
class DataBaseConfig(Serializable):
    data_aug: str = 'RANDOM'
    norm_type: str = 'minmax'    

@dataclasses.dataclass
class DatasetConfig_REDD(DatasetConfig):
    appliances: List[str] = dataclasses.field(default_factory=lambda: ['refrigerator','microwave','dishwasher','washer_dryer'])
    house_no: int = 1
    splits: str = '3:1:1'
    index_buffer_flag: bool = True
    
@dataclasses.dataclass
class DatasetConfig_REDD_Bitcn(DatasetConfig):
    appliance: str = 'microwave'
    win_size: int = 600
    stride: int = 60
    
@dataclasses.dataclass
class DatasetConfig_REDD_multilabel(DatasetConfig):
    appliances: List[str] = dataclasses.field(default_factory=lambda: ['refrigerator','microwave','dishwasher','washer_dryer'])
    house_no: int = 1
    splits: str = '3:1:1'
    index_buffer_flag: bool = True
    win_size: int = 600
    stride: int = 60
    
