from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def load_redd_bitcn(data_root, appliance_name):
    data_root = Path(data_root)

    df_train = pd.read_csv(
        data_root.joinpath(appliance_name + '_train_.csv'),
        sep=',',
        header=None,
        index_col=None,
    )

    df_val = pd.read_csv(
        data_root.joinpath(appliance_name + '_validation_.csv'),
        sep=',',
        header=None,
        index_col=None,
    )

    def_test = pd.read_csv(
        data_root.joinpath(appliance_name + '_test_.csv'),
        sep=',',
        header=None,
        index_col=None,
    )

    return df_train.to_numpy(), df_val.to_numpy(), def_test.to_numpy()


def test_load_redd_bitcn():
    train, val, test = load_redd_bitcn('data/low_freq/', 'microwave')
    print(train.shape)
    print(val.shape)
    print(test.shape)
    print(train[0])
