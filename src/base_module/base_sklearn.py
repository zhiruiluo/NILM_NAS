from __future__ import annotations

import logging

import torch

from src.base_module.base_estimator import BaseEstimator
from src.base_module.configs import ExpResults
from src.base_module.metrics_helper import Metrics_Helper
from src.config_options import MyProgramArgs
from src.utilts import get_datetime_now_tz

logger = logging.getLogger(__name__)


class SklearnBaseModule(BaseEstimator):
    def __init__(self, args: MyProgramArgs) -> None:
        super().__init__()
        self.args = args
        self.metrics_init(args.modelConfig.nclass,
                          args.modelBaseConfig.label_mode)

    def get_skmodel(self):
        return

    def on_train(self, x_all, y_all):
        logger.info('[on_train]')
        self.get_skmodel().fit(x_all, y_all)
        y_hat = self.get_skmodel().predict(x_all)
        return y_hat

    def on_val(self, x_all, y_all):
        y_hat = self.get_skmodel().predict(x_all)
        return y_hat

    def on_test(self, x_all, y_all):
        y_hat = self.get_skmodel().predict(x_all)
        return y_hat

    def on_reshape(self, x_all):
        return x_all

    def fit(self, datamodule):
        logger.info('[SklearnBase] start fitting')
        datamodule.setup('fit')
        for phase, dl in zip(
            ['train', 'val'],
            [datamodule.train_dataloader(), datamodule.val_dataloader()],
        ):
            samples, s_type = self._dataloader_to_numpy(dl)
            if s_type == 'tuple':
                x_all = samples[0]
                y_all = samples[1]
            elif s_type == 'dict':
                x_all = samples['input']
                y_all = samples['target']

            x_all = self.on_reshape(x_all)
            if phase == 'train':
                y_hat = self.on_train(x_all, y_all)
            else:
                y_hat = self.on_val(x_all, y_all)

            logger.debug(f'y_hat {y_hat.shape} {y_all.shape}')
            self.metrics(phase, torch.tensor(y_hat), torch.tensor(y_all))
            self.metrics_end(phase)

    def test(self, datamodule):
        logger.info('[SklearnBase] start testing')
        datamodule.setup('test')
        phase = 'test'
        samples, s_type = self._dataloader_to_numpy(
            datamodule.test_dataloader())
        if s_type == 'tuple':
            x_all = samples[0]
            y_all = samples[1]
        elif s_type == 'dict':
            x_all = samples['input']
            y_all = samples['target']
        x_all = self.on_reshape(x_all)
        y_hat = self.on_test(x_all, y_all)
        self.metrics(phase, torch.tensor(y_hat), torch.tensor(y_all))
        self.metrics_end(phase)
        return self.metrics_log


def training_flow_sklearn(args, model: SklearnBaseModule, dataset) -> ExpResults:
    time_on_fit_start = get_datetime_now_tz()
    model.fit(dataset)
    time_on_fit_end = get_datetime_now_tz()

    time_on_test_start = get_datetime_now_tz()
    sklearn_results = model.test(dataset)
    time_on_test_end = get_datetime_now_tz()

    results = Metrics_Helper.from_results(
        results=sklearn_results,
        start_time=time_on_fit_start,
        training_time=time_on_fit_end - time_on_fit_start,
        params=args,
    )
    return results
