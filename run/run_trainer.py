from __future__ import annotations

import functools
import logging
import os
import sys
sys.path.append(".")
import ray
from ray import tune
from ray.air import RunConfig

from src.cluster_deploy.slurm_deploy import slurm_launch
from src.config_options import MyProgramArgs
from src.config_options.options import OptionManager, replace_consistant
from src.database.Persistence import PersistenceFactory
from src.project_logging import LoggerManager
from src.trainer.TraningManager import trainable
from memory_profiler import profile
import gc

@profile
def main():
    opt = OptionManager()
    args = opt.replace_params(
        {
            "modelConfig": "BitcnNILM",
            "datasetConfig": "REDD_multilabel",
            "datasetConfig.splits": '4:2:4',
            "datasetConfig.combine_mains": True,
            "nasOption.enable": True,
            "nasOption.num_cpus": 16,
            "nasOption.num_gpus": 0,
            "nasOption.search_strategy": "random",
            "nasOption.backend": "no_report",
            "nasOption.num_samples": 1,
            "modelBaseConfig.label_mode": "multilabel",
            "modelBaseConfig.epochs": 1,
            "modelBaseConfig.patience": 1,
            "modelBaseConfig.label_smoothing": 0.2,
            "modelBaseConfig.lr": 1e-3,
            "modelBaseConfig.weight_decay": 1e-3,
            "modelBaseConfig.batch_size": 128,
            "modelBaseConfig.val_batch_size": 512,
            "modelBaseConfig.test_batch_size": 512,
            "trainerOption.limit_train_batches": 0.1,
            # "trainerOption.limit_val_batches": 0.1,
            # "trainerOption.limit_test_batches": 0.1,
            "modelBaseConfig.lr_scheduler": 'none',
        },
    )
    persistance = PersistenceFactory(
        db_name=args.systemOption.db_name,
        db_dir=args.systemOption.db_dir,
    ).get_persistence()

    del persistance

    root_logger = LoggerManager.get_main_logger(args, "raytune")
    root_logger.info(args)
    
    space = {
        "datasetConfig": {
            # "combine_mains": True,
            "imbalance_sampler": False,
            "win_size": 60,
            "house_no": 1,
            "stride": 30,
            "drop_na_how": 'any',
        },
        "modelConfig": {
            "head_type": 'ASL',
            "in_chan": 1,
        },
    }
    gc.enable()
    gc.collect()
    config = replace_consistant(args, space)
    for i in range(3):
        gc.collect()
        trainable(config)
        
    gc.collect()
    
main()