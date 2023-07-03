from __future__ import annotations

from src.base_module.base_sklearn import SklearnBaseModule
from src.config_options import MyProgramArgs
from src.config_options.model_configs import ModelConfig_MLSVM
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


class MLSVM(SklearnBaseModule):
    def __init__(self, args: MyProgramArgs) -> None:
        super().__init__(args)
        config: ModelConfig_MLSVM = args.modelConfig
        self.model = OneVsRestClassifier(SVC(
            C=config.C,
            kernel=config.kernel,
            degree=config.degree,
            gamma=config.gamma,
            max_iter=config.max_iter
        ), n_jobs=args.nasOption.num_cpus)
        
    def get_skmodel(self):
        return self.model
    
    def on_reshape(self, x_all):
        shape = x_all.shape
        return x_all.reshape(shape[0],-1)