from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from src.base_module.base_sklearn import SklearnBaseModule
from src.config_options import MyProgramArgs
from src.config_options.model_configs import ModelConfig_KNC


class KNC(SklearnBaseModule):
    def __init__(self, args: MyProgramArgs) -> None:
        super().__init__(args)
        config: ModelConfig_KNC = args.modelConfig
        self.model = KNeighborsClassifier(
            n_neighbors=config.n_neighbors,
            weights=config.weights,
            algorithm=config.algorithm,
            leaf_size=config.leaf_size,
            n_jobs=args.nasOption.num_cpus,
        )

    def get_skmodel(self):
        return self.model

    def on_reshape(self, x_all):
        shape = x_all.shape
        return x_all.reshape(shape[0],-1)
