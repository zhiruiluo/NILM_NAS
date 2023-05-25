from __future__ import annotations
from pathlib import Path
from src.config_options.model_configs import ModelConfig_BasicV2
from src.config_options import MyProgramArgs
from src.database.Persistence import PersistenceFactory
from src.trainer.TraningManager import trainable
from src.project_logging import LoggerManager
from src.config_options.options import OptionManager
from src.cluster_deploy.slurm_deploy import slurm_launch
import logging
from ray.tune.search.bayesopt import BayesOptSearch
from ray.air import RunConfig
from ray import tune
import ray
import os

import sys

sys.path.append('.')


logger = logging.getLogger(__name__)


def search_space_basicv2_dict(args):
    space = {
        'modelConfig': {
            'chan_1': tune.uniform(5, 9),
            'chan_2': tune.uniform(5, 10),
            'chan_3': tune.uniform(5, 10),
            'ker_1': tune.uniform(3, 9),
            'ker_2': tune.uniform(3, 7),
            'ker_3': tune.uniform(3, 4),
            'stride_1': tune.uniform(1, 3),
            'stride_2': tune.uniform(0, 3),
            'stride_3': tune.uniform(0, 2),
            'dropout': tune.uniform(5, 7),
        },
        'args': args,
    }
    return space


def space_to_model_config(config: dict) -> ModelConfig_BasicV2:
    return ModelConfig_BasicV2(
        chan_1=int(2 ** int(config['chan_1'])),
        chan_2=int(2 ** int(config['chan_2'])),
        chan_3=int(2 ** int(config['chan_3'])),
        ker_1=int(config['ker_1']),
        ker_2=int(config['ker_2']),
        ker_3=int(config['ker_3']),
        stride_1=int(2 ** int(config['stride_1'])),
        stride_2=int(2 ** int(config['stride_2'])),
        stride_3=int(2 ** int(config['stride_3'])),
        dropout=int(config['dropout']) / 10.0,
    )


def trainable_wrapper(config: dict):
    os.sched_setaffinity(0, list(range(os.cpu_count())))
    config['args'].modelConfig = space_to_model_config(config['modelConfig'])
    return trainable(config)


def loop(args: MyProgramArgs):
    space = search_space_basicv2_dict(args)

    metric = 'val_acc'
    mode = 'max'

    trainable_resource = tune.with_resources(
        trainable_wrapper,
        resources={'CPU': args.nasOption.num_cpus,
                   'GPU': args.nasOption.num_gpus},
    )

    tuner = tune.Tuner(
        trainable_resource,
        tune_config=tune.TuneConfig(
            mode=mode,
            metric=metric,
            search_alg=BayesOptSearch(),
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
    exp_name='BO',
    num_nodes=4,
    num_gpus=2,
    partition='epscor',
    load_env='conda activate p39c116\n'
    + 'export OMP_NUM_THREADS=10\n'
    + 'export PL_DISABLE_FORK=1',
    command_suffix="--address='auto' --exp_name={{EXP_NAME}}",
)
def main():
    print(f'[getcwd] {os.getcwd()}')
    opt = OptionManager()
    args = opt.replace_params(
        {
            'modelConfig': 'BasicV2',
            'datasetConfig': 'REDD_Bitcn',
            'nasOption.enable': True,
            'nasOption.num_cpus': 16,
            'nasOption.num_gpus': 1,
            'nasOption.search_strategy': 'BayesOptimization',
            'nasOption.backend': 'no_report',
            'nasOption.num_samples': 100,
            'modelBaseConfig.label_mode': 'multiclass',
            'modelBaseConfig.epochs': 20,
            'modelBaseConfig.patience': 20,
            'modelBaseConfig.label_smoothing': 0.2,
            'modelBaseConfig.lr': 1e-3,
            'modelBaseConfig.weight_decay': 1e-3,
            'modelBaseConfig.batch_size': 32,
            'modelBaseConfig.val_batch_size': 32,
            'modelBaseConfig.test_batch_size': 32,
            # "trainerOption.limit_train_batches": 0.1,
            # "trainerOption.limit_val_batches": 0.1,
            # "trainerOption.limit_test_batches": 0.1,
        },
    )
    persistance = PersistenceFactory(
        db_name=args.systemOption.db_name, db_dir=args.systemOption.db_dir,
    ).get_persistence()

    del persistance

    root_logger = LoggerManager.get_main_logger(args, 'raytune')
    root_logger.info(args)

    if not args.nasOption.enable:
        return

    if os.environ.get('head_node_ip', None):
        ray.init(
            address=args.systemOption.address,
            _node_ip_address=os.environ['head_node_ip'],
        )
    else:
        ray.init()

    for house_no in [1]:
        # for question_no in [5418]:
        args.datasetConfig.house_no = house_no
        logger.info(f'[Search] searching for house_no {house_no}')
        loop(args)


main()
