from __future__ import annotations

import sys

sys.path.append(".")
import functools
import logging
import os

import ray
from ray import tune
from ray.air import RunConfig

from src.cluster_deploy.slurm_deploy import slurm_launch
from src.config_options import MyProgramArgs
from src.config_options.options import OptionManager, replace_consistant
from src.database.Persistence import PersistenceFactory
from src.project_logging import LoggerManager
from src.trainer.TraningManager import trainable


logger = logging.getLogger(__name__)


def trainable_wrapper(config: dict, args: MyProgramArgs):
    os.sched_setaffinity(0, list(range(os.cpu_count())))
    config = replace_consistant(args, config)
    # config['args'].modelConfig = space_to_model_config(config['modelConfig'])
    return trainable(config)


def loop(args: MyProgramArgs):
    metric = "val_acc"
    mode = "max"

    space = {
        "datasetConfig": {
            "combine_mains": tune.grid_search([True]),
            "imbalance_sampler": tune.grid_search([False]),
            "win_size": tune.grid_search([60, 150, 300]),
            "stride": tune.grid_search([30]),
            "house_no": tune.grid_search([1,3]),
            "drop_na_how": tune.grid_search(['any'])
        },
        # "modelConfig": {
        #     "n_neighbors": tune.grid_search([3,4,5,6,7]),
        #     "weights": tune.grid_search(['uniform','distance']),
        #     "algorithm": tune.grid_search(["auto",'ball_tree','kd_tree','brute']),
        #     "leaf_size": tune.grid_search([30,10,20,40,50]),
        # },
        "modelConfig": {
            "n_neighbors": tune.grid_search([3,5]),
            "weights": tune.grid_search(['uniform','distance']),
            "algorithm": tune.grid_search(["auto"]),
            "leaf_size": tune.grid_search([30,20,40]),
        },
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
    exp_name="ML_MLkNN",
    num_nodes=2,
    num_gpus=0,
    partition="epscor",
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
            "modelConfig": "KNC",
            "datasetConfig": "REDD_multilabel",
            "nasOption.enable": True,
            "nasOption.num_cpus": 4,
            "nasOption.num_gpus": 0,
            "nasOption.search_strategy": "random",
            "nasOption.backend": "no_report",
            "nasOption.num_samples": 1,
            # "datasetConfig.win_size": 600,
            # "datasetConfig.stride": 1,
            "modelBaseConfig.label_mode": "multilabel",
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
