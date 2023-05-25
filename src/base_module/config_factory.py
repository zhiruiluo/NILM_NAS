import logging

from .base_lightning import LightningBaseModule
from .base_lightningdata import LightningBaseDataModule
from .base_nnsklearn import NNSklearnBaseModule
from .base_sklearn import SklearnBaseModule
from .configs import DataBaseConfig, HyperParm, NNSklearnBaseConfig, SklearnBaseConfig

logger = logging.getLogger(__name__)


class ConfigFactory:
    def get_config_cls_by_cls(self, model_cls: type):
        base_ = model_cls.__base__
        if base_ is LightningBaseModule:
            config_cls = HyperParm, "hyperParm"
        elif base_ is LightningBaseDataModule:
            config_cls = DataBaseConfig, "dataBaseConfig"
        elif base_ is NNSklearnBaseModule:
            config_cls = NNSklearnBaseConfig, "nnSklearnBaseConfig"
        elif base_ is SklearnBaseModule:
            config_cls = SklearnBaseConfig, "sklearnBaseConfig"
        else:
            raise ValueError(f"type {model_cls} has invalid base {base_}")
        return config_cls

    # def get_config_cls_by_model_obj(self, model_obj):
    #     return

    # def get_config_by_config_name(self, config_name: str):
    #     return
