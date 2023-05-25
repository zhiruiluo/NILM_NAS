from __future__ import annotations

import json
import logging
import os
import pprint
import traceback
from typing import Dict

from src.base_module.base_lightning import count_parameters
from src.config_options.option_def import MyProgramArgs
from src.database.Persistence import PersistenceFactory
from src.project_logging import LoggerManager
from src.trainer.Imp_training import DatasetFactory
from src.trainer.Imp_training import ModelFactory
from src.trainer.Imp_training import TrainingFactory

logger = logging.getLogger(__name__)


class TrainingManager:
    def __init__(self, args: MyProgramArgs):
        self.args = args
        LoggerManager.get_task_logger(self.args)
        # assert get_num_gpus() >= self.args.nasOption.num_gpus
        logger.info(self.args)
        self.persistence = PersistenceFactory(
            db_name=self.args.systemOption.db_name, db_dir=self.args.systemOption.db_dir,
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
        trial_ret = {'setting': vars(self.args).copy(), 'results': results}
        logger.info(f'test results {pprint.pformat(trial_ret)}')
        self.persistence.save_expresults(results, self.args)
        return results


def trainable(config: dict, debug=False):
    print('[start trainable]')
    if 'args' in config:
        args: MyProgramArgs = config['args']

    if 'nas_params' in config:
        nas_params: dict = config['nas_params']

    # from ray.air import session

    if not debug:
        print('[trainable] non_debug_mode')
        LoggerManager.get_task_logger(args)
        logger.info('[trainable] non_debug_mode')
    else:
        print('[trainable] debug_mode')
        LoggerManager.get_task_debug_logger(args)
        logger.info('[trainable] debug_mode')

    # assert get_num_gpus() >= self.args.nasOption.num_gpus
    logger.info(args)
    try:
        # if

        # os.sched_setaffinity(0, list(range(os.cpu_count())))

        # init training
        dataset = DatasetFactory().get_dataset(args)
        dataset.prepare_data()

        model = ModelFactory().get_model(dataset.nclass, args)

        tfac = TrainingFactory(args, model, dataset)
        training = tfac.get_training()

        results = training.train()
        # session.report({"accuracy": results.test_metrics.acc})
        # session.report({"f1macro": results.test_metrics.f1macro})
        trial_ret = {'setting': vars(args).copy(), 'results': results}
        logger.info(f'test results {pprint.pformat(trial_ret)}')

        if 'nas_params' in locals():
            results.nas_params = json.dumps(nas_params)
        if args.nasOption.enable:
            results.nas_params = json.dumps(
                {
                    'params': count_parameters(model, trainable_only=False),
                    'trainable_params': count_parameters(model, trainable_only=True),
                },
            )
        persistence = PersistenceFactory(
            db_name=args.systemOption.db_name, db_dir=args.systemOption.db_dir,
        ).get_persistence()
        persistence.save_expresults(results, args)
        logger.info(f'Training ended, results saved!')

    except Exception as e:
        logger.error(f'Error pid {os.getpid()}: {traceback.format_exc()}')
        print(f'Error pid {os.getpid()}: {traceback.format_exc()}')
        raise e

    return {
        'val_acc': results.val_metrics.acc,
        'val_f1macro': results.val_metrics.f1macro,
    }
