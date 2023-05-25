from __future__ import annotations

from src.base_module.configs import ExpResults
from src.config_options.option_def import MyProgramArgs


class BaseTraining:
    def __init__(self, args: MyProgramArgs, model, dataset, base_type):
        self.args = args
        self.model = model
        self.dataset = dataset
        self._check_type(model, base_type)

    def _check_type(self, model, base_type):
        if not isinstance(model, base_type):
            raise TypeError(
                f'Model is not the type {base_type}, but {type(model)}!')

    def train(self) -> ExpResults:
        pass
