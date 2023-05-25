from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pytz
import torch
from ml_toolkit.utils.decorator import disk_buffer
from ml_toolkit.utils.normalization import get_norm_cls
from ml_toolkit.utils.prettyprint import pretty_print_confmx_pandas
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.base_module.base_lightning import LightningTrainerFactory
from src.base_module.configs import ExpResults, Metrics
from src.base_module.metrics_helper import get_metrics
from src.config_options import MyProgramArgs
from src.database.Persistence import PersistenceFactory

logger = logging.getLogger(__name__)


class FeatureHook:
    def __init__(self) -> None:
        self.features = defaultdict(list)
        self.handles = []

    def register_modules(self, modules):
        for name, module in modules.items():
            self.handles.append(
                module.register_forward_hook(self.module_forward_hook(name)),
            )
            logger.debug(f"module registered {name}")

    def module_forward_hook(self, name):
        def hook(model, input, output):
            logger.debug("module_forward_hook invoked")
            out = output.clone().detach().cpu()
            logger.debug(f"{type(output)} {output.shape}")
            self.features[name].append(out)

        return hook

    def get_features(self, name):
        return torch.vstack(self.features[name])

    def remove_all_hooks(self):
        while len(self.handles):
            hook = self.handles.pop()
            hook.remove()

    def reset(self):
        self.features = defaultdict(list)


class NNSklearnGridBaseModule:
    def __init__(self, args: MyProgramArgs) -> None:
        self.args = args
        self.metrics_init(args.modelConfig.nclass)
        self.phase_feature = {}

    def metrics_init(self, nclass):
        self.all_metrics = {}
        for phase in ["train", "val", "test"]:
            self.all_metrics[phase + "_metrics"] = get_metrics(
                ["acc", "accmacro", "f1macro", "f1micro", "f1none", "confmx"],
                nclass,
            )
            # self.all_metrics[phase + "_metrics"] = {
            #     "acc": Accuracy(),
            #     "accmacro": Accuracy(num_classes=nclass, average="macro"),
            #     "f1macro": F1Score(num_classes=nclass, average="macro"),
            #     "f1micro": F1Score(num_classes=nclass, average="micro"),
            #     "f1none": F1Score(num_classes=nclass, average="none"),
            #     "confmx": ConfusionMatrix(nclass),
            # }
        self.metrics_log = {}

    def get_test_confmx(self):
        if self.stored_test_confmx is not None:
            return self.stored_test_confmx.cpu().numpy().tolist()
        return []

    def metrics(self, phase, pred, label):
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            metric(pred, label)

    def metrics_end(self, phase):
        metrics = {}
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            metrics[mk] = metric.compute()
            metric.reset()

        self.log_epoch_end(phase, metrics)
        if phase == "test":
            self.stored_test_confmx = metrics["confmx"]

    def log_epoch_end(self, phase, metrics):
        self.metrics_log[f"{phase}_acc"] = metrics["acc"].item()
        self.metrics_log[f"{phase}_accmacro"] = metrics["accmacro"].item()
        self.metrics_log[f"{phase}_f1micro"] = metrics["f1micro"].item()
        self.metrics_log[f"{phase}_f1macro"] = metrics["f1macro"].item()
        # self.metrics_log[f'{phase}_confmx'] = metrics['confmx'].type(torch.long).cpu().numpy().tolist()

        logger.info(f'[{phase}_acc] {metrics["acc"]}')
        logger.info(f'[{phase}_accmacro_epoch] {metrics["accmacro"]}')
        logger.info(f'[{phase}_f1_score] {metrics["f1micro"]}')
        logger.info(f'[{phase}_f1_score_macro] {metrics["f1macro"]}')
        logger.info(
            f'[{phase}_confmx] \n{pretty_print_confmx_pandas(metrics["confmx"].detach().cpu().type(torch.long))}',
        )

    def register_hook(self):
        return

    def get_features(self):
        return

    def get_nnmodel(self):
        return

    def get_skmodel(self):
        return

    def get_features_folder(self):
        return

    def get_grid(self):
        return

    def fit_nn(self, dataset):
        trainer_fac = LightningTrainerFactory(self.args)
        trainer = trainer_fac.get_trainer()
        results = trainer_fac.training_flow(
            trainer,
            self.get_nnmodel(),
            dataset,
            no_test=True,
        )
        return results

    def dl_to_feature(self, phase, dl):

        y_all = []
        x1_all = []
        x_all = None
        for batch in dl:
            if len(batch) == 3:
                x_batch, x1_batch, y_batch = batch
                logger.debug(f"[x1_batch] {x1_batch.shape}")
                x1_all.append(x1_batch.cpu().numpy())
            else:
                x_batch, y_batch = batch
            self.get_nnmodel()(x_batch)
            y_all.append(y_batch.cpu().numpy())

        features = self.get_features()
        logger.debug(f"features {features.shape}")
        # x_fea = rearrange(features.numpy(), 'n c v t -> n (c v t)')
        x_fea = features.numpy()
        logger.info(f"[{phase}] x_fea.shape {x_fea.shape}")
        if self.args.modelBaseConfig.norm_type:
            if phase == "train":
                norm_cls = get_norm_cls(self.args.modelBaseConfig.norm_type)
                self.norm_cls = norm_cls(mask_axis=1)
                x_fea = self.norm_cls.fit_transform(x_fea)
            else:
                x_fea = self.norm_cls.transform(x_fea)

        if len(x1_all) != 0:
            logger.debug(f"x1 {np.concatenate(x1_all).shape}")
            # x1 = rearrange(np.concatenate(x1_all), 'n c v t-> n (c v t)')
            x1 = np.concatenate(x1_all)
            logger.debug(f"{x_fea.shape} {x1.shape}")
            x_all = np.concatenate([x_fea, x1], axis=1)
        else:
            x_all = x_fea

        logger.debug(x_all.shape)
        y_all = np.concatenate(y_all)
        return x_all, y_all

    def share_nnsklearn_epoch(self, phase, dl):
        logger.info(f"[share_nnsklearn_epoch] {phase}")
        with disk_buffer(
            self.dl_to_feature,
            keys=f"{self.args.datasetConfig.question_no}_{phase}",
            folder=f".temp/{self.get_features_folder()}",
        ) as dl_to_feature_:
            x_all, y_all = dl_to_feature_(phase, dl)

        x_all, y_all = self.dl_to_feature(phase, dl)

        if phase == "train":
            self.get_skmodel().fit(x_all, y_all)

        y_hat = self.get_skmodel().predict(x_all)
        self.metrics(phase, torch.tensor(y_hat), torch.tensor(y_all))
        self.metrics_end(phase)
        # self.feature_hook.reset()

    def fit_sklearn(self, datamodule):
        datamodule.setup("fit")
        self.register_hook()
        for phase, dl in zip(
            ["train", "val"],
            [datamodule.train_dataloader(), datamodule.val_dataloader()],
        ):
            self.share_nnsklearn_epoch(phase, dl)
        self.feature_hook.remove_all_hooks()

    def test_sklearn(self, datamodule) -> ExpResults:
        datamodule.setup("test")
        phase = "test"
        self.register_hook()
        self.share_nnsklearn_epoch(phase, datamodule.test_dataloader())
        self.feature_hook.remove_all_hooks()

        return self.metrics_log


def from_results(
    results: dict,
    start_time: datetime,
    training_time: timedelta,
    flops: int,
    params: MyProgramArgs,
) -> ExpResults:
    my_results = {}
    for phase in ["train", "val", "test"]:
        metrics = Metrics(
            acc=results[f"{phase}_acc"],
            accmacro=results[f"{phase}_accmacro"],
            f1macro=results[f"{phase}_f1macro"],
            f1micro=results[f"{phase}_f1micro"],
            # confmx=results[f'{phase}_confmx']
        )
        my_results[f"{phase}_metrics"] = metrics
    my_results["start_time"] = start_time
    my_results["training_time"] = training_time
    my_results["flops"] = flops
    my_results["params"] = params
    return ExpResults(**my_results)


def training_flow_nnsklearngrid(
    args,
    model: NNSklearnGridBaseModule,
    dataset,
) -> ExpResults:
    time_on_fit_start = datetime.now(pytz.timezone("America/Denver"))
    nn_results = model.fit_nn(dataset)
    time_on_fit_end = datetime.now(pytz.timezone("America/Denver"))
    time_train_nn = time_on_fit_end - time_on_fit_start

    grid_best_results = None
    best_val_f1macro = 0
    for p in tqdm(ParameterGrid(model.get_grid())):
        new_args = model.set_skmodel_params(p)

        sklearn_fit_start = datetime.now(pytz.timezone("America/Denver"))
        model.fit_sklearn(dataset)
        sklearn_fit_end = datetime.now(pytz.timezone("America/Denver"))
        time_fit_skmodel = sklearn_fit_end - sklearn_fit_end

        time_on_test_start = datetime.now(pytz.timezone("America/Denver"))
        sklearn_results = model.test_sklearn(dataset)
        time_on_test_end = datetime.now(pytz.timezone("America/Denver"))
        logger.info(f"[nnsklearn grid] {p}")
        results = from_results(
            results=sklearn_results,
            start_time=time_on_fit_start,
            training_time=time_train_nn + time_fit_skmodel,
            flops=nn_results.flops,
            params=new_args,
        )

        persistence = PersistenceFactory(
            db_name=args.systemOption.db_name,
            db_dir=args.systemOption.db_dir,
        ).get_persistence()
        persistence.save_expresults(results, new_args)

        logger.info(f"[nnsklearn grid] {results}")
        if best_val_f1macro < results.val_metrics.f1macro:
            best_val_f1macro = results.val_metrics.f1macro
            grid_best_results = results

    return grid_best_results
