from __future__ import annotations

import logging
import os
import sys
import traceback

from src.base_module.base_estimator import BaseEstimator
from src.base_module.base_estimator import training_flow_estimator
from src.base_module.base_lightning import LightningBaseModule
from src.base_module.base_lightning import LightningTrainerFactory
from src.base_module.base_nnsklearn import NNSklearnBaseModule
from src.base_module.base_nnsklearn import training_flow_nnsklearn
from src.base_module.base_nnsklearngrid import NNSklearnGridBaseModule
from src.base_module.base_nnsklearngrid import training_flow_nnsklearngrid
from src.base_module.base_sklearn import SklearnBaseModule
from src.base_module.base_sklearn import training_flow_sklearn
from src.base_module.configs import ExpResults
from src.config_options.option_def import MyProgramArgs
from src.dataset import DatasetSelection
from src.model import ModelSelection
from src.trainer.base_training import BaseTraining

sys.path.append('.')


logger = logging.getLogger(__name__)


class LightningTraining(BaseTraining):
    def __init__(self, args: MyProgramArgs, model, dataset):
        super().__init__(args, model, dataset, LightningBaseModule)
        self._check_type(model, LightningBaseModule)

    def train(self) -> ExpResults:
        try:
            trainer_fac = LightningTrainerFactory(self.args)
            results = trainer_fac.training_flow(self.model, self.dataset)
            return results
        except Exception:
            logger.error(f'Error pid {os.getpid()}: {traceback.format_exc()}')
            print(f'Error pid {os.getpid()}: {traceback.format_exc()}')
            exit()


class SklearnTraining(BaseTraining):
    def __init__(self, args: MyProgramArgs, model, dataset):
        super().__init__(args, model, dataset, SklearnBaseModule)

    def train(self):
        results = training_flow_sklearn(self.args, self.model, self.dataset)
        return results


class NNSklearnGridTraining(BaseTraining):
    def __init__(self, args: MyProgramArgs, model, dataset):
        super().__init__(args, model, dataset, NNSklearnGridBaseModule)

    def train(self) -> ExpResults:
        results = training_flow_nnsklearngrid(
            self.args, self.model, self.dataset)
        return results


class NNSklearnTraining(BaseTraining):
    def __init__(self, args: MyProgramArgs, model, dataset):
        super().__init__(args, model, dataset, NNSklearnBaseModule)

    def train(self) -> ExpResults:
        results = training_flow_nnsklearn(self.args, self.model, self.dataset)
        return results


class EstimatorTraining(BaseTraining):
    def __init__(self, args, model, dataset):
        super().__init__(args, model, dataset, BaseEstimator)

    def train(self):
        results = training_flow_estimator(self.args, self.model, self.dataset)
        return results


class TrainingFactory:
    def __init__(self, args, model, dataset):
        self.args = args
        self.model = model
        self.dataset = dataset

    def get_training(self) -> BaseTraining:
        args, model, dataset = self.args, self.model, self.dataset
        if isinstance(model, LightningBaseModule):
            logger.info('[TrainingFactory] get lightning')
            training = LightningTraining(args, model, dataset)
        elif isinstance(model, NNSklearnBaseModule):
            logger.info('[TrainingFactory] get nnsklearn')
            training = NNSklearnTraining(args, model, dataset)
        elif isinstance(model, SklearnBaseModule):
            logger.info('[TrainingFactory] get sklearn')
            training = SklearnTraining(args, model, dataset)
        elif isinstance(model, NNSklearnGridBaseModule):
            logger.info('[TrainingFactory] get nnsklearngrid')
            training = NNSklearnGridTraining(args, model, dataset)
        elif isinstance(model, BaseEstimator):
            logger.info('[TrainingFactory] get base estimator')
            training = EstimatorTraining(args, model, dataset)
        else:
            raise ValueError(f'Model {model} is invalid')
        return training


class ModelFactory:
    def __init__(self) -> None:
        pass

    def get_model_cls(self, args: MyProgramArgs):
        ms = ModelSelection()
        model_cls = ms.getClass(args.expOption.model)
        return model_cls

    def get_model(self, nclass, args: MyProgramArgs):
        args.modelConfig.nclass = nclass
        model_cls = self.get_model_cls(args)
        return model_cls(args)


class DatasetFactory:
    def __init__(self) -> None:
        pass

    def get_dataset_cls(self, args: MyProgramArgs):
        ds = DatasetSelection()
        dataset_cls = ds.getClass(args.expOption.dataset)
        return dataset_cls

    def get_dataset(self, args: MyProgramArgs) -> LightningBaseModule:
        dataset_cls = self.get_dataset_cls(args)
        return dataset_cls(args)
