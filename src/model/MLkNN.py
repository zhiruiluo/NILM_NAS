from __future__ import annotations


from src.base_module.base_sklearn import SklearnBaseModule
from src.config_options import MyProgramArgs
from src.config_options.model_configs import ModelConfig_MLkNN
from skmultilearn.adapt import MLkNN as mlknn


class MLkNN(SklearnBaseModule):
    def __init__(self, args: MyProgramArgs) -> None:
        super().__init__(args)
        config: ModelConfig_MLkNN = args.modelConfig
        self.model = mlknn(
            config.k, 
            config.s, 
            ignore_first_neighbours=config.ignore_first_neighbours
        )
        
    def get_skmodel(self):
        return self.model
    
    def on_reshape(self, x_all):
        shape = x_all.shape
        return x_all.reshape(shape[0],-1)