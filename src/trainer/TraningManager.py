from __future__ import annotations

import json
import logging
import os
import pprint
import random
import traceback

import numpy as np
import torch
import gc

from src.base_module.base_lightning import count_parameters
from src.config_options.option_def import MyProgramArgs
from src.database.Persistence import PersistenceFactory
from src.project_logging import LoggerManager
from src.trainer.Imp_training import (DatasetFactory, ModelFactory,
                                      TrainingFactory)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


logger = logging.getLogger(__name__)


class TrainingManager:
    def __init__(self, args: MyProgramArgs):
        self.args = args
        LoggerManager.get_task_logger(self.args)
        # assert get_num_gpus() >= self.args.nasOption.num_gpus
        logger.info(self.args)
        self.persistence = PersistenceFactory(
            db_name=self.args.systemOption.db_name,
            db_dir=self.args.systemOption.db_dir,
        ).get_persistence()
        os.sched_setaffinity(0, list(range(os.cpu_count())))

    def init_training(self):
        dataset = DatasetFactory().get_dataset(self.args)
        dataset.prepare_data()

        model = ModelFactory().get_model(dataset.nclass, self.args)

        self.tfac = TrainingFactory(self.args, model, dataset)
        self.training = self.tfac.get_training()

    def train(self):
        self.init_training()
        results = self.training.train()
        trial_ret = {"setting": vars(self.args).copy(), "results": results}
        logger.info(f"test results {pprint.pformat(trial_ret)}")
        self.persistence.save_expresults(results, self.args)
        return results

def trainable(config: dict, debug=False):
    if "args" in config:
        args: MyProgramArgs = config["args"]
    
    if "nas_params" in config:
        nas_params: dict = config["nas_params"]

    if args.trainerOption.verbose:
        print("[start trainable]")

    if not debug:
        if args.trainerOption.verbose:
            print("[trainable] non_debug_mode")
        LoggerManager.get_task_logger(args)
        logger.info("[trainable] non_debug_mode")
    else:
        if args.trainerOption.verbose:
            print("[trainable] debug_mode")
        LoggerManager.get_task_debug_logger(args)
        logger.info("[trainable] debug_mode")

    # assert get_num_gpus() >= self.args.nasOption.num_gpus
    logger.info(args)
    try:
        # if

        # os.sched_setaffinity(0, list(range(os.cpu_count())))

        # init training
        dataset = DatasetFactory().get_dataset(args)
        dataset.prepare_data()
        dataset.setup('fit')
        for i in dataset.train_dataloader():
            ...
        dataset.setup('test')
        model = ModelFactory().get_model(dataset.nclass, args)

        tfac = TrainingFactory(args, model, dataset)
        training = tfac.get_training()

        results = training.train()
        trial_ret = {"setting": vars(args).copy(), "results": results}
        logger.info(f"test results {pprint.pformat(trial_ret)}")
        
        
        if args.nasOption.enable:
            nas_p = {}
            if "nas_params" in locals():
                nas_p.update(nas_params)
            if hasattr(model, 'parameters'):
                nas_p.update({
                        "params": count_parameters(model, trainable_only=False),
                        "trainable_params": count_parameters(model, trainable_only=True),
                    },)
                results.nas_params = json.dumps(
                    nas_p
                )
        persistence = PersistenceFactory(
            db_name=args.systemOption.db_name,
            db_dir=args.systemOption.db_dir,
        ).get_persistence()
        persistence.save_expresults(results, args)
        logger.info(f"Training ended, results saved!")
        
        del model
        del dataset
        del persistence
        gc.collect()

    except Exception as e:
        logger.error(f"Error pid {os.getpid()}: {traceback.format_exc()}")
        print(f"Error pid {os.getpid()}: {traceback.format_exc()}")
        raise e

    return {
        "val_acc": results.val_metrics.acc,
        "val_accmacro": results.val_metrics.accmacro,
        "val_f1macro": results.val_metrics.f1macro,
        "flops": results.flops,
    }
