from __future__ import annotations

import hashlib
import logging
import os
import time
from logging import config
from pathlib import Path

from .config_options.option_def import MyProgramArgs


def logging_setup_dict_config() -> dict:
    return {
        'loggers': {
            'src': {
                'level': 'INFO',
                'propagate': False,
                'handlers': ['console', 'file', 'errorlog'],
            },
        },
        'version': 1,
        'disable_existing_loggers': True,
        'handlers': {
            'console': {
                'formatter': 'default',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
                'level': 'NOTSET',
            },
            'errorlog': {
                'delay': True,
                'formatter': 'default',
                'mode': 'a',
                'encoding': 'utf-8',
                'level': 'ERROR',
                'class': 'logging.FileHandler',
            },
            'file': {
                'class': 'logging.FileHandler',
                'formatter': 'default',
                'mode': 'a',
                'level': 'NOTSET',
                'encoding': 'utf-8',
            },
        },
        'root': {'level': 'INFO', 'handlers': ['console', 'file', 'errorlog']},
        'formatters': {
            'default': {
                'datefmt': '%Y/%m/%d-%H:%M:%S',
                'format': '[%(asctime)s-%(process)d-%(levelname)-7s-%(name)s-#%(lineno)d] %(message)s',
            },
        },
    }


def get_truncated_sha(src, length):
    m = hashlib.sha256(bytes(str(src), encoding='utf8'))
    return m.hexdigest()[:length]


def timestamp():
    return time.strftime('%m%d_%H%M', time.localtime())


def get_time_sha(args):
    return f'{timestamp()}_{get_truncated_sha(args, 20)}'


def setup_logger(dir, fn, disable_stream_output, debug):
    config_dict = logging_setup_dict_config()
    dir = Path(dir)
    print(f'[setup_logger] {dir}')
    dir.mkdir(parents=True, exist_ok=True)
    for fh, suffix in zip(['file', 'errorlog'], ['.log', '.err']):
        config_dict['handlers'][fh]['filename'] = dir.joinpath(fn + suffix)

    if debug:
        for k in config_dict['loggers'].keys():
            config_dict['loggers'][k]['level'] = 'DEBUG'
        config_dict['root']['level'] = 'DEBUG'

    if disable_stream_output:
        for k in config_dict['loggers'].keys():
            config_dict['loggers'][k]['handlers'] = ['file', 'errorlog']
        config_dict['root']['handlers'] = ['file', 'errorlog']

    config.dictConfig(config_dict)
    return logging.getLogger()


class LoggerManager:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_main_logger(args: MyProgramArgs, main_name: str):
        Path(args.systemOption.exp_dir).mkdir(parents=True, exist_ok=True)
        return setup_logger(
            args.systemOption.job_dir,
            main_name,
            args.systemOption.disable_stream_output,
            args.systemOption.debug,
        )

    @staticmethod
    def get_task_logger(args: MyProgramArgs):
        args.systemOption.task_name = (
            f'{args.expOption.dataset}_{args.expOption.model}_TSHA{get_time_sha(args)}'
        )
        args.systemOption.update_dir()
        args.systemOption.mkdir()
        args.save_yaml(args.systemOption.task_dir + '/hparam.yaml')
        print(
            '[get_task_logger]',
            args.systemOption.task_dir,
            '[get_cwd_task]',
            os.getcwd(),
        )
        setup_logger(
            args.systemOption.task_dir,
            f'train_{get_time_sha(args)}',
            disable_stream_output=args.systemOption.disable_stream_output,
            debug=args.systemOption.debug,
        )
