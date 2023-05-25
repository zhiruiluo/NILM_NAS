from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List

from simple_parsing import parse

params_appliance = {
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [1, 2, 3],
        'channels': [11, 6, 16],
        'train_build': [2, 3],
        'test_build': 1,
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [1, 2, 3],
        'channels': [5, 9, 7],
        'train_build': [2, 3],
        'test_build': 1,
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [1, 2, 3],
        'channels': [6, 10, 9],
        'train_build': [2, 3],
        'test_build': 1,
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [1, 2, 3],
        'channels': [20, 7, 13],
        'train_build': [2, 3],
        'test_build': 1,
    },
}


@dataclass
class Config:
    data_dir: str = './data/low_freq'
    appliances: list = field(default=['fridge'])
    aggregate_mean: float = 522
    aggregate_std: float = 814
    save_path: str = './data/low_freq/multilabel/'


def get_arguments():
    args = parse(Config)
    return args


def generate_dataset(args: Config):
    sample_seconds = 6
    validation_percent = 10
    nrows = None
    debug = False

    print(args)
    for appliance_name in args.appliances:
        for h in params_appliance[appliance_name]['houses']:
            chn_num = str(
                params_appliance[appliance_name]['channels'][
                    params_appliance[appliance_name]['houses'].index(h)
                ],
            )
            path = (
                Path(args.data_dir)
                .joinpath(f'house_{h}')
                .joinpath(f'channel_{chn_num}.dat')
            )
            print('    ' + path.as_posix())


def main():
    args = get_arguments()
    generate_dataset(args)
