from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

import numpy as np
import pytz
import torch
from ml_toolkit.utils.prettyprint import pretty_print_confmx_pandas

from src.base_module.metrics_helper import Metrics_Helper, get_metrics
from src.config_options import MyProgramArgs

logger = logging.getLogger(__name__)


class BaseEstimator:
    def __init__(self) -> None:
        pass

    def metrics_init(self, nclass, label_mode):
        if label_mode == "multilabel":
            multilabel = True
            metric_keys = [
                "acc_perclass",
                "accmacro",
                "f1_perclass",
                "f1macro",
                "confmx",
            ]
            # metric_keys = ['acc','accmacro', 'loss', 'f1macro', 'f1micro', 'f1none', 'confmx']
        elif label_mode == "multiclass":
            multilabel = False
            metric_keys = [
                "acc",
                "accmacro",
                "f1macro",
                "f1micro",
                "f1none",
                "f1binary",
                "confmx",
            ]
        else:
            raise ValueError("label mode is invalid")

        self.all_metrics = {}
        for phase in ["train", "val", "test"]:
            self.all_metrics[phase + "_metrics"] = get_metrics(
                metric_keys,
                nclass,
                multilabel=multilabel,
            )
        self.metrics_log = {}

    def metrics(self, phase, pred, label):
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            if mk == "acc":
                metric(pred, label)
            else:
                metric.update(pred, label.to(torch.long))

    def metrics_end(self, phase):
        metrics = {}
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            metrics[mk] = metric.compute()
            metric.reset()

        self.log_epoch_end(phase, metrics)
        if phase == "test":
            self.stored_test_confmx = metrics["confmx"]

    def get_test_confmx(self):
        if self.stored_test_confmx is not None:
            return self.stored_test_confmx.cpu().numpy().tolist()
        return []

    def log_epoch_end(self, phase, metrics):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                if k == "confmx":
                    value = v.type(torch.long).cpu().numpy().tolist()
                    log_str = pretty_print_confmx_pandas(v)
                else:
                    value = v.item()
                    log_str = str(value)
            else:
                value = v
                log_str = str(v)

            self.metrics_log[f"{phase}_{k}"] = value
            logger.info(f"[{phase}_{k}] {log_str}")

    def _dataloader_to_numpy(self, dl):
        batch = next(iter(dl))
        if isinstance(batch, tuple):
            samples = self._dataloader_tuple_to_numpy(dl)
            return samples, "tuple"
        elif isinstance(batch, dict):
            samples = self._dataloader_dict_to_numpy(dl)
            return samples, "dict"

    def _dataloader_tuple_to_numpy(self, dl):
        x_all = []
        y_all = []
        for x, y in dl:
            x_all.append(x.cpu().numpy())
            y_all.append(y.cpu().numpy())

        x_all = np.vstack(x_all)
        y_all = np.concatenate(y_all)
        return x_all, y_all

    def _dataloader_dict_to_numpy(self, dl):
        all_samples = defaultdict(list)
        for batch in dl:
            for k, v in batch.items():
                all_samples[k].append(v.numpy())

        all_samples = {k: np.concatenate(v) for k, v in all_samples.items()}
        for k, v in all_samples.items():
            logger.info(f"{k} {v.shape}")
        return all_samples


def training_flow_estimator(args: MyProgramArgs, model, dataset):
    time_on_fit_start = datetime.now(pytz.timezone("America/Denver"))
    model.fit(dataset)
    time_on_fit_end = datetime.now(pytz.timezone("America/Denver"))

    time_on_test_start = datetime.now(pytz.timezone("America/Denver"))
    test_results = model.test(dataset)
    time_on_test_end = datetime.now(pytz.timezone("America/Denver"))

    results = Metrics_Helper.from_results(
        results=test_results,
        start_time=time_on_fit_start,
        training_time=time_on_fit_end - time_on_fit_start,
        params=args,
    )
    return results
