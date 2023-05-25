from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Union

from simple_parsing import Serializable
from simple_parsing import subgroups

from .dataset_configs import *
from .model_configs import *
from .modelbase_configs import *
from .nas_configs import *
from src.context import get_project_root

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainerOption(Serializable):
    no_cuda: bool = False
    accelerator: str = 'gpu'
    devices: int = 1
    precision: str = 32
    auto_bs: bool = False
    profiler: str = ''
    strategy: str = ''
    fast_dev_run: bool = False
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches: float = 1.0


@dataclasses.dataclass
class ExpOption(Serializable):
    model: str = ''
    dataset: str = ''
    exp: int = 1
    gridid: int = 1
    nfold: int = 1
    nrepeat: int = 1


@dataclasses.dataclass
class ResultOption(Serializable):
    fold: int = 1
    repeat: int = 1


@dataclasses.dataclass
class SystemOption(Serializable):
    exp_name: str = 'exp1'
    job_name: str = 'job1'
    task_name: str = 'task1'
    db_name: str = 'nas_results.db'
    log_dir: str = get_project_root().joinpath('logging').resolve().as_posix()
    exp_dir: str = dataclasses.field(init=False)
    job_dir: str = dataclasses.field(init=False)
    task_dir: str = dataclasses.field(init=False)
    db_dir: str = dataclasses.field(init=False)
    address: str = ''
    disable_stream_output: bool = True
    debug: bool = False
    seed: int = 32

    def update_dir(self):
        self.log_dir = Path(self.log_dir)
        self.exp_dir = self.log_dir.joinpath(self.exp_name)
        self.job_dir = self.exp_dir.joinpath(self.job_name)
        self.task_dir = self.job_dir.joinpath(self.task_name)

        self.log_dir = str(self.log_dir)
        self.exp_dir = str(self.exp_dir)
        self.job_dir = str(self.job_dir)
        self.task_dir = str(self.task_dir)
        self.db_dir = str(self.exp_dir)

    def mkdir(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.exp_dir).mkdir(parents=True, exist_ok=True)
        Path(self.job_dir).mkdir(parents=True, exist_ok=True)
        Path(self.task_dir).mkdir(parents=True, exist_ok=True)

    def __post_init__(self):
        self.update_dir()


class groups_helper:
    def __init__(self, groups: dict) -> None:
        self.group_dict = groups

    def default(self):
        key = list(self.group_dict.keys())[0]
        return key

    def get_groups(self):
        return self.group_dict

    def get_name(self, dataclass_cls):
        for k, v in self.group_dict.items():
            if isinstance(dataclass_cls, v):
                return k
        raise ValueError(f'{dataclass_cls} not found')


class ModelGroups(groups_helper):
    groups = {
        'BasicV2': ModelConfig_BasicV2,
        'BitcnNILM': ModelConfig_BitcnNILM,
        'BasicV3_Pool': ModelConfig_BasicV3_Pool,
        'TSNet': ModelConfig_TSNet,
        'RF': ModelConfig_RF,
    }

    def __init__(self) -> None:
        super().__init__(self.groups)


class DatasetGroups(groups_helper):
    groups = {
        'REDD': DatasetConfig_REDD,
        'REDD_Bitcn': DatasetConfig_REDD_Bitcn,
        'REDD_multilabel': DatasetConfig_REDD,
    }

    def __init__(self) -> None:
        super().__init__(self.groups)


class ModelbaseGroups(groups_helper):
    groups = {
        'lightning': HyperParm,
        'sklearn': SklearnBaseConfig,
        'nnsklearn': NNSklearnBaseConfig,
        'estimator': EsitmatorBaseConfig,
    }

    def __init__(self) -> None:
        super().__init__(self.groups)

    def get_modelbase_by_model(self, model: type):
        modelbase = None
        if issubclass(model, LightningModel):
            modelbase = self.groups['lightning']
        elif issubclass(model, NNSklearnModel):
            modelbase = self.groups['nnsklearn']
        elif issubclass(model, SklearnModel):
            modelbase = self.groups['sklearn']
        elif issubclass(model, EstimatorModel):
            modelbase = self.groups['estimator']
        return modelbase


@dataclasses.dataclass
class MyProgramArgs(Serializable):
    systemOption: SystemOption = dataclasses.field(
        default_factory=SystemOption)
    expOption: ExpOption = dataclasses.field(default_factory=ExpOption)
    resultOption: ResultOption = dataclasses.field(
        default_factory=ResultOption)
    datasetConfig: DatasetConfig = subgroups(
        DatasetGroups().get_groups(), default=DatasetGroups().default(),
    )
    dataBaseConfig: DataBaseConfig = dataclasses.field(
        default_factory=DataBaseConfig)
    modelConfig: ModelConfig = subgroups(
        ModelGroups().get_groups(), default=ModelGroups().default(),
    )
    modelBaseConfig: ModelBaseConfig = subgroups(
        ModelbaseGroups().get_groups(), default=ModelbaseGroups().default(),
    )
    trainerOption: TrainerOption = dataclasses.field(
        default_factory=TrainerOption)
    nasOption: NASOption = dataclasses.field(default_factory=NASOption)

    def __post_init__(self):
        if type(self.modelConfig) == str:
            self.modelConfig = ModelGroups().get_groups()[self.modelConfig]()

        if type(self.datasetConfig) == str:
            self.datasetConfig = DatasetGroups().get_groups()[
                self.datasetConfig]()

        if self.expOption.model == '':
            self.expOption.model = ModelGroups().get_name(self.modelConfig)

        if self.expOption.dataset == '':
            self.expOption.dataset = DatasetGroups().get_name(self.datasetConfig)
