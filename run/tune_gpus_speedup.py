from __future__ import annotations

import os
import sys

sys.path.append(".")
from src.config_options import OptionManager
from src.trainer.TraningManager import trainable



def launch_jobs(config):
    os.sched_setaffinity(0, list(range(os.cpu_count())))
    trainable(config)


def main():
    opt = OptionManager()
    args = opt.replace_params(
        {
            "modelConfig": "BasicV3_Pool",
            "datasetConfig": "REDD_Bitcn",
            "nasOption.enable": True,
            "nasOption.num_cpus": 16,
            "nasOption.num_gpus": 0,
            "nasOption.search_strategy": "BayesOptimization",
            "nasOption.backend": "no_scheduler",
            "trainerOption.precision": "32",
            "trainerOption.profiler": "",
            "datasetConfig.win_size": 600,
            "datasetConfig.stride": 1,
            "modelBaseConfig.label_mode": "multiclass",
            "modelBaseConfig.epochs": 20,
            "modelBaseConfig.patience": 20,
            "modelBaseConfig.label_smoothing": 0.2,
            "modelBaseConfig.lr": 1e-3,
            "modelBaseConfig.weight_decay": 1e-3,
            "modelBaseConfig.batch_size": 128,
        },
    )

    config = {"args": args}

    trainable(config)


main()
