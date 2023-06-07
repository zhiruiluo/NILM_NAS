from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from src.base_module.base_sklearn import SklearnBaseModule
from src.config_options import MyProgramArgs
from src.config_options.model_configs import ModelConfig_RF


class RF(SklearnBaseModule):
    def __init__(self, args: MyProgramArgs) -> None:
        super().__init__(args)
        config: ModelConfig_RF = args.modelConfig
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            criterion=config.criterion,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
        )

    def get_skmodel(self):
        return self.model

    def on_reshape(self, x_all):
        shape = x_all.shape
        return x_all.reshape(shape[0],-1)
