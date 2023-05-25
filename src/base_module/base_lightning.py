import logging
import os
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
from ml_toolkit.utils.cuda_status import get_num_gpus
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, TQDMProgressBar)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from src.base_module.configs import ExpResults
from src.base_module.metrics_helper import Metrics_Helper, get_metrics
from src.config_options.modelbase_configs import HyperParm
from src.config_options.option_def import MyProgramArgs
from src.FlopsProfiler import FlopsProfiler
from src.utilts import get_datetime_now_tz

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = True):
    if trainable_only:
        total_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
    return total_params


class LightningBaseModule(pl.LightningModule):
    def __init__(self, args: MyProgramArgs):
        super().__init__()
        self.save_hyperparameters(ignore=["args"])
        self.args = args
        self.baseconfig: HyperParm = self.args.modelBaseConfig
        self.metrics_init(self.args.modelConfig.nclass, self.baseconfig.label_mode)
        
    def metrics_init(self, nclass, label_mode):
        if label_mode == 'multilabel':
            multilabel = True
            metric_keys = ['acc','acc_perclass', 'accmacro',
                           'loss', 'f1_perclass', 'f1macro', 'confmx']
        elif label_mode == 'multiclass':
            multilabel = False
            metric_keys = ['acc', 'accmacro', 'loss', 'f1macro',
                           'f1micro', 'f1none', 'f1binary', 'confmx']
        else:
            raise ValueError('label mode is invalid')

        self.all_metrics = nn.ModuleDict()

        for phase in ["train", "val", "test"]:
            self.all_metrics[phase + "_metrics"] = nn.ModuleDict(
                get_metrics(metric_keys, nclass, multilabel=multilabel)
            )

    def forward(self, batch):
        return batch

    def metrics(self, phase, pred, label, loss):
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            if mk == "loss":
                metric.update(loss)
            elif mk == "acc":
                metric(pred, label)
                self.log(
                    f"{phase}_acc_step",
                    metric,
                    sync_dist=True,
                    prog_bar=True,
                    batch_size=self.args.modelBaseConfig.batch_size,
                )
            else:
                metric.update(pred, label.to(torch.long))

    def metrics_end(self, phase):
        metrics = {}
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            metrics[mk] = metric.compute().cpu().tolist()
            metric.reset()
        # logger.info(metrics)
        self.log_epoch_end(phase, metrics)
        if phase == "test":
            self.stored_test_confmx = metrics["confmx"]

    def get_test_confmx(self):
        if self.stored_test_confmx is not None:
            return self.stored_test_confmx
        return []

    def log_epoch_end(self, phase, metrics):
        logger.info(f"Current Epoch: {self.current_epoch}")
        for k, v in metrics.items():
            if k == 'confmx':
                logger.info(f'[{phase}_confmx] \n{metrics["confmx"]}')
                continue
            if isinstance(v, list):
                for i, vi in enumerate(v):
                    self.log(f"{phase}_{k}_{i}", vi)
            else:
                self.log(f"{phase}_{k}", v)
            logger.info(f"[{phase}_{k}] {v}")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.args.modelBaseConfig.lr,
            weight_decay=self.args.modelBaseConfig.weight_decay,
        )
        scheduler = StepLR(optimizer, step_size=7)
        if self.args.nasOption.backend == 'no_scheduler':
            return {"optimizer": optimizer}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_predict(self, y):
        a, y_hat = torch.max(y, dim=1)
        return y_hat

    def shared_my_step(self, batch, batch_nb, phase):
        # batch
        predictions = self.forward(batch)
        loss = predictions['loss']
        output = predictions['output']
        target = batch['target']

        self.log(
            f"{phase}_loss_step",
            loss,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.modelBaseConfig.batch_size,
        )
        self.log(
            f"{phase}_loss_epoch",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.args.modelBaseConfig.batch_size,
        )
        self.metrics(phase, output, target, loss)

        return loss

    def training_step(self, batch, batch_nb):
        phase = "train"
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def on_train_epoch_end(self) -> None:
        phase = "train"
        self.metrics_end(phase)

    def validation_step(self, batch, batch_nb):
        phase = "val"
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def on_validation_epoch_end(self) -> None:
        phase = "val"
        self.metrics_end(phase)

    def test_step(self, batch, batch_nb):
        phase = "test"
        target = batch['target']
        predictions = self.forward(batch)
        output = predictions['output']
        loss = predictions['loss']
        self.metrics(phase, output, target, loss)
        return

    def on_test_epoch_end(self) -> None:
        phase = "test"
        self.metrics_end(phase)


class LightningTrainerFactory:
    def __init__(self, args: MyProgramArgs) -> None:
        self.args = args

    def _get_logger(self, phase: str):
        name = f"tensorboard_log"
        version = "{}_{}".format(
            phase, time.strftime("%m%d-%H%M", time.localtime()))

        tb_logger = TensorBoardLogger(
            save_dir=self.args.systemOption.task_dir,
            name=name,
            version=version,
        )

        csv_logger = CSVLogger(
            save_dir=self.args.systemOption.task_dir, name=name, version=version)

        return [tb_logger, csv_logger]

    def _configure_callbacks(self):
        callbacks = []
        monitor = "val_f1macro"
        # monitor = 'val_acc_epoch'

        earlystop = EarlyStopping(
            monitor=monitor, patience=self.args.modelBaseConfig.patience, mode="max"
        )
        callbacks.append(earlystop)

        ckp_cb = ModelCheckpoint(
            dirpath=self.args.systemOption.task_dir,
            filename="bestCKP" + "-{epoch:02d}-{val_f1macro:.3f}",
            monitor=monitor,
            save_top_k=1,
            mode="max",
        )
        callbacks.append(ckp_cb)

        pb_cb = TQDMProgressBar(refresh_rate=0.05)
        callbacks.append(pb_cb)

        lr_cb = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_cb)

        if self.args.nasOption.enable and self.args.nasOption.backend == 'ray_tune':
            from ray.tune.integration.pytorch_lightning import \
                TuneReportCallback
            logger.info(
                f"[callbacks] ray_tune backend with TuneReportCallback")
            if self.baseconfig.label_mode == 'multilabel':
                # metric_keys = ['acc_perclass', 'accmacro', 'f1_perclass', 'f1macro', 'confmx']
                metrics = {
                    "val_loss": "val_loss",
                    "val_acc": "val_acc",
                    "val_f1macro": "val_f1macro",
                    "val_epoch": "val_epoch",
                }
            elif self.baseconfig.label_mode == 'multiclass':
                # metric_keys = ['acc','accmacro', 'loss', 'f1macro', 'f1micro', 'f1none','f1binary', 'confmx']
                metrics = {
                    "val_loss": "val_loss",
                    "val_acc": "val_acc",
                    "val_f1macro": "val_f1macro",
                    "val_epoch": "val_epoch",
                }
            else:
                raise ValueError('invalid label mode')

            callbacks.append(
                TuneReportCallback(
                    metrics=metrics,
                    on=["validation_epoch_end"])
            )

        return callbacks

    def _auto_accelerator(self):
        accelerator = 'cpu'
        if get_num_gpus() > 0 and not self.args.trainerOption.no_cuda:
            accelerator = 'gpu'
        return accelerator

    def get_profiler(self):
        from pytorch_lightning.profilers import (AdvancedProfiler,
                                                 SimpleProfiler)
        if self.args.trainerOption.profiler == 'simple':
            return SimpleProfiler(dirpath=self.args.systemOption.task_dir, filename='profiler.txt')

        return None

    def get_fit_trainer(self):
        self.args.trainerOption.accelerator = self._auto_accelerator()
        callbacks = [*self._configure_callbacks()]
        params = {
            "accelerator": self.args.trainerOption.accelerator,
            "devices": self.args.trainerOption.devices,
            "fast_dev_run": self.args.trainerOption.fast_dev_run,
            "precision": self.args.trainerOption.precision,
            "max_epochs": self.args.modelBaseConfig.epochs,
            # "auto_scale_batch_size": False
            # if self.args.trainerOption.auto_bs == ""
            # else self.args.trainerOption.auto_bs,
            "logger": self._get_logger('fit'),
            "callbacks": callbacks,
            "profiler": self.get_profiler(),
            "limit_train_batches": self.args.trainerOption.limit_train_batches,
            "limit_val_batches": self.args.trainerOption.limit_val_batches,
            "limit_test_batches": self.args.trainerOption.limit_test_batches,
        }
        logger.info(f"[fit_trainer] {params}")
        return pl.Trainer(**params)

    def get_val_trainer(self):
        accelerator = self._auto_accelerator()
        params = {
            "accelerator": accelerator,
            "devices": 1,
            "max_epochs": 1,
            "callbacks": [TQDMProgressBar(refresh_rate=0.05)],
            "limit_val_batches": self.args.trainerOption.limit_val_batches,
            "logger": self._get_logger('val'),
        }
        logger.info(f"[val_trainer] {params}")
        return pl.Trainer(**params)

    def get_test_trainer(self):
        accelerator = self._auto_accelerator()
        params = {
            "accelerator": accelerator,
            "devices": 1,
            "max_epochs": 1,
            "callbacks": [TQDMProgressBar(refresh_rate=0.05)],
            "limit_test_batches": self.args.trainerOption.limit_test_batches,
            "logger": self._get_logger('test'),
        }
        logger.info(f"[test_trainer] {params}")
        return pl.Trainer(**params)

    def training_flow(self, model, dataset, no_test: bool = False) -> ExpResults:
        logger.info("[start training flow]")

        flops_profiler = FlopsProfiler(self.args)
        dataset.setup("fit")
        flops = None
        for batch in dataset.train_dataloader():
            if isinstance(batch, dict):
                flops = flops_profiler.get_flops(model, args=[batch])
            break

        fit_trainer = self.get_fit_trainer()
        time_on_fit_start = get_datetime_now_tz()
        fit_trainer.fit(model, datamodule=dataset)
        time_on_fit_end = get_datetime_now_tz()
        fit_results = fit_trainer.logged_metrics

        ckp_cb = fit_trainer.checkpoint_callback
        earlystop_cb = fit_trainer.early_stopping_callback

        logger.info(
            "Interrupted %s, early stopped epoch %s",
            fit_trainer.interrupted,
            earlystop_cb.stopped_epoch,
        )
        # validation
        val_trainer = self.get_val_trainer()
        val_trainer.validate(
            model, ckpt_path=ckp_cb.best_model_path, datamodule=dataset)
        val_results = val_trainer.logged_metrics

        # test model
        test_trainer = self.get_test_trainer()
        if not no_test and os.path.isfile(ckp_cb.best_model_path) and not fit_trainer.interrupted and not val_trainer.interrupted:
            time_on_test_start = get_datetime_now_tz()
            test_results = test_trainer.test(
                model, ckpt_path=ckp_cb.best_model_path, datamodule=dataset)[0]
            time_on_test_end = get_datetime_now_tz()

        # delete check point
        if os.path.isfile(ckp_cb.best_model_path):
            os.remove(ckp_cb.best_model_path)

        # convert test_result dictionary to dictionary
        if not fit_trainer.interrupted and not val_trainer.interrupted and not test_trainer.interrupted:
            results = {}
            results.update(fit_results)
            results.update(val_results)
            if not no_test:
                results.update(test_results)
            results = Metrics_Helper.from_results(
                results,
                start_time=time_on_fit_start,
                training_time=time_on_fit_end - time_on_fit_start,
                flops=flops,
                params=self.args,
            )
            return results

        return None
