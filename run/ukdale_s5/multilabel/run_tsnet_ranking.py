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
import pandas as pd

logger = logging.getLogger(__name__)


def trainable_wrapper(config: dict, args: MyProgramArgs):
    os.sched_setaffinity(0, list(range(os.cpu_count())))
    config = replace_consistant(args, config)
    # config['args'].modelConfig = space_to_model_config(config['modelConfig'])
    return trainable(config)

def parse_random_rank_csv():
    with open('./run/ukdale/multilabel/random_rank_1.csv', mode='r') as fp:
        df_rank = pd.read_csv(fp)
    rank = []
    for l, s in df_rank.iterrows():
        rank.append(json.loads(s['modelConfig'])['bit_string'])
    return rank

def test_rank():
    rank = parse_random_rank_csv()
    print(rank)

def get_bit_string(spec, list_tsnets: list):
    win_size = spec.config.datasetConfig.win_size
    # stride = spec.config.datasetConfig.stride
    house_no = spec.config.datasetConfig.house_no
    
    for tsnet in list_tsnets:
        if tsnet['house_no'] == house_no and tsnet['win_size'] == win_size:
            return tsnet['bit_string']

        
def loop(args: MyProgramArgs):
    metric = "val_acc"
    mode = "max"

    rank = parse_random_rank_csv()
    
    space = {
        "modelBaseConfig": {
            # "lr": tune.sample_from(lambda spec: spec.config['modelBaseConfig']['batch_size']/64*1e-3),
            "lr": 0.025,
            # "optimizer": 'SGD',
            # "lr_scheduler": "CosineAnnealingLR",
            "batch_size": tune.grid_search([512]),
            "epochs": tune.grid_search([50]),
            "patience": tune.sample_from(lambda spec: spec.config['modelBaseConfig']['epochs'])
        },
        "datasetConfig": {
            "win_size": tune.grid_search([150]),
            "stride": tune.grid_search([10]),
            "house_no": tune.grid_search([2]),
            "splits": "4:2:4",
        },
        "modelConfig":{
            "n_phases": 3,
            "n_ops": 4,
            "in_channels": 1,
            "bit_string": tune.grid_search(rank),
            # "out_channels": tune.grid_search([16, 20, 24, 28, 32, 48, 64]),
            "out_channels": tune.grid_search([32]),
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
    exp_name="RK",
    num_nodes=4,
    num_gpus=2,
    partition="epscor",
    log_dir='logging/UKDALE_424',
    load_env="conda activate p39c116\n"
    + "export OMP_NUM_THREADS=10\n"
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
            "nasOption.enable": True,
            "nasOption.num_cpus": 16,
            "nasOption.num_gpus": 1,
            "nasOption.search_strategy": "random",
            "nasOption.backend": "no_report",
            "nasOption.num_samples": 1,
            "datasetConfig.win_size": 60,
            "datasetConfig.stride": 30,
            "modelBaseConfig.label_mode": "multilabel",
            "modelBaseConfig.epochs": 20,
            "modelBaseConfig.patience": 20,
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

if __name__ == '__main__':
    main()
