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
import json


logger = logging.getLogger(__name__)


def trainable_wrapper(config: dict, args: MyProgramArgs):
    config.pop('model')
    os.sched_setaffinity(0, list(range(os.cpu_count())))
    config = replace_consistant(args, config)
    return trainable(config)
    

def parse_best_tsnet_json():
    with open('./run/ukdale/multilabel/best_pf_f1macro_424.json', mode='r') as fp:
        list_tsnets = json.load(fp)
        
    return list_tsnets
    
        
def loop(args: MyProgramArgs):
    metric = "val_acc"
    mode = "max"

    list_tsnets = parse_best_tsnet_json()
    
    space = {
        "model": tune.grid_search(list_tsnets),
        "datasetConfig": {
            "imbalance_sampler": tune.grid_search([False]),
            "win_size": tune.sample_from(lambda spec: spec.config.model['win_size']),
            "stride": tune.grid_search([30]),
            "house_no": tune.grid_search([2]),
        },
        "modelConfig":{
            "n_phases": 3,
            "n_ops": 5,
            "in_channels": 1,
            "bit_string": tune.sample_from(lambda spec: spec.config.model['bit_string']),
            "out_channels": tune.sample_from(lambda spec: spec.config.model['out_channels']),
            "head_type": "ASL",
        }
    }

    trainable_partial = functools.partial(trainable_wrapper, args=args)

    trainable_resource = tune.with_resources(
        trainable_partial,
        resources={"CPU": args.nasOption.num_cpus, "GPU": args.nasOption.num_gpus},
    )

    tuner = tune.Tuner(
        trainable_resource,
        tune_config=tune.TuneConfig(
            # mode=mode,
            # metric=metric,
            # search_alg=BayesOptSearch(),
            num_samples=args.nasOption.num_samples,
            chdir_to_trial_dir=False,
            reuse_actors=False,
        ),
        run_config=RunConfig(
            name=args.systemOption.exp_name,
        ),
        param_space=space,
    )
    # path = Path('~/ray_results').joinpath(args.systemOption.exp_name).joinpath('tuner.pkl')
    # if path.is_file():
    #     tuner.restore(path)
    results = tuner.fit()


@slurm_launch(
    exp_name="TSNET_pareto",
    num_nodes=2,
    num_gpus=2,
    partition="epscor",
    log_dir="logging/UKDALE_424",
    load_env="conda activate p39c116\n"
    + "export PL_DISABLE_FORK=1",
    command_suffix="--address='auto' --exp_name={{EXP_NAME}}",
)
def main():
    print(f"[getcwd] {os.getcwd()}")
    opt = OptionManager()
    args = opt.replace_params(
        {
            "modelConfig": "TSNet",
            "datasetConfig": "UKDALE_multilabel",
            "datasetConfig.splits": '4:2:4',
            "trainerOption.monitor": 'val_f1macro',
            "trainerOption.mode": "max",
            "nasOption.enable": True,
            "nasOption.num_cpus": 8,
            "nasOption.num_gpus": 1,
            "nasOption.search_strategy": "random",
            "nasOption.backend": "no_report",
            "nasOption.num_samples": 1,
            "modelBaseConfig.label_mode": "multilabel",
            "modelBaseConfig.epochs": 100,
            "modelBaseConfig.patience": 100,
            "modelBaseConfig.label_smoothing": 0.2,
            "modelBaseConfig.lr": 1e-3,
            "modelBaseConfig.weight_decay": 1e-3,
            "modelBaseConfig.batch_size": 128,
            "modelBaseConfig.val_batch_size": 512,
            "modelBaseConfig.test_batch_size": 512,
            "modelBaseConfig.lr_scheduler": 'none',
            # "trainerOption.limit_train_batches": 0.1,
            # "trainerOption.limit_val_batches": 0.1,
            # "trainerOption.limit_test_batches": 0.1,
        },
    )
    persistance = PersistenceFactory(
        db_name=args.systemOption.db_name,
        db_dir=args.systemOption.db_dir,
    ).get_persistence()

    del persistance

    root_logger = LoggerManager.get_main_logger(args, "raytune")
    root_logger.info(args)

    if not args.nasOption.enable:
        return

    if os.environ.get("head_node_ip", None):
        ray.init(
            address=args.systemOption.address,
            _node_ip_address=os.environ["head_node_ip"],
        )
    else:
        ray.init()
    
    loop(args)


main()
