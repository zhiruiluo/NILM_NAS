from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from ml_toolkit.utils.decorator import disk_buffer
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from ..sampler import ImbalancedDatasetSampler
from .redd_load import load_redd_bitcn
from src.base_module.base_lightningdata import LightningBaseDataModule
from src.config_options.dataset_configs import DatasetConfig_REDD_Bitcn
from src.config_options.option_def import MyProgramArgs
from src.context import get_project_root


class Seq2pointDataset(Dataset):
    def __init__(self, np_data: np.ndarray, win_size: int, stride: int) -> None:
        self.np_data = np_data
        # self.input = torch.tensor(np_data[:,0], dtype=torch.float32)
        # self.target = torch.tensor(np_data[:,1], dtype=torch.long)
        win_view = sliding_window_view(
            np.arange(self.np_data.shape[0]), window_shape=win_size,
        )
        self.indices = list(range(0, win_view.shape[0], stride))
        self.length = len(self.indices)
        self.win_view = win_view

    def get_labels(self):
        labels = []
        for i in self.indices:
            labels.append(self.np_data[self.win_view[i][-1], 1])

        return labels

    def __getitem__(self, index):
        idx = self.indices[index]
        win_indices = self.win_view[idx]
        # return {'input': self.input[idx], "target": self.target[idx]}
        return {
            'input': torch.tensor(self.np_data[win_indices, 0], dtype=torch.float32),
            'target': torch.tensor(self.np_data[win_indices[-1], 1], dtype=torch.long),
        }

    def __len__(self):
        return self.length


def apply_transform(
    train: dict[str, list], val: dict[str, list], test: dict[str, list], t,
):
    for f in ['mains_1', 'mains_2']:
        train[f] = t.fit_transform(train[f]).tolist()
        val[f] = t.transform(val[f]).tolist()
        test[f] = t.transform(test[f]).tolist()
    return train, val, test


def normalize(train: dict[str, list], val: dict[str, list], test: dict[str, list]):
    from sklearn.preprocessing import Normalizer

    norm = Normalizer()
    return apply_transform(train, val, test, norm)


def minmax(train: dict[str, list], val: dict[str, list], test: dict[str, list]):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    return apply_transform(train, val, test, scaler)


def standardize(train: dict[str, list], val: dict[str, list], test: dict[str, list]):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    return apply_transform(train, val, test, scaler)


def data_augmentation(train):
    df_train = pd.DataFrame.from_dict(train)
    df_train_last = df_train.applymap(lambda x: x[-1])

    def power(x):
        c = 0
        for i, v in enumerate(x[2:], 1):
            c += i * v
        return c

    df_train_last['powerlabel'] = df_train_last.apply(
        power, axis=1).astype(int)
    from imblearn.over_sampling import RandomOverSampler

    sampler = RandomOverSampler()
    # df_train_last['powerlabel'].hist(figsize=(5,5))
    # plt.savefig('results/powerlabel.png')

    df_train_resampled, df_y_resampled = sampler.fit_resample(
        df_train, df_train_last['powerlabel'],
    )
    # df_y_resampled.hist(figsize=(5,5))
    # plt.savefig('results/resampled.png')
    return df_train_resampled.to_dict('list')


class REDD_Bitcn(pl.LightningDataModule):
    def __init__(self, args: MyProgramArgs) -> None:
        super().__init__()
        self.config: DatasetConfig_REDD_Bitcn = args.datasetConfig
        self.prepare_data_per_node = False
        self.save_hyperparameters('args')
        self.data_root = 'data/low_freq/'
        self.args = args

    def visualize(self):
        folder = get_project_root().joinpath('.temp').as_posix()
        with disk_buffer(
            func=load_redd_bitcn, keys=str(self.config.appliance), folder=folder,
        ) as bf_fn:
            train, val, test = bf_fn(self.data_root, self.config.appliance)

        df_train = pd.DataFrame.from_dict(train)
        df_train_last = df_train.applymap(lambda x: x[-1])
        print(df_train_last)
        # classes = df_train_last.columns[2:]

        def power(x):
            c = 0
            for i, v in enumerate(x[2:], 1):
                c += i * v
            return c

        df_train_last['powerlabel'] = df_train_last.apply(
            power, axis=1).astype(int)

        df_train_last.hist(figsize=(10, 10))
        plt.savefig('results/power_label.png')

    def prepare_data(self) -> None:
        folder = get_project_root().joinpath('.temp').as_posix()
        with disk_buffer(
            func=load_redd_bitcn, keys=str(self.config.appliance), folder=folder,
        ) as bf_fn:
            train, val, test = bf_fn(self.data_root, self.config.appliance)

        # train, val, test = minmax(train, val, test)
        # train = data_augmentation(train)
        self.train_set = Seq2pointDataset(
            train, self.config.win_size, self.config.stride,
        )
        self.val_set = Seq2pointDataset(
            val, self.config.win_size, self.config.stride)
        self.test_set = Seq2pointDataset(
            test, self.config.win_size, self.config.stride)
        self.nclass = 2

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            ...
        if stage == 'test':
            ...

    def _to_dataloader(
        self, dataset, shuffle, batch_size, num_workers, drop_last, sampler=None,
    ):
        if sampler:
            shuffle = False

        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=drop_last,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            # prefetch_factor=1,
        )

    def train_dataloader(self) -> DataLoader:
        sampler = ImbalancedDatasetSampler(self.train_set)
        return self._to_dataloader(
            self.train_set,
            True,
            self.args.modelBaseConfig.batch_size,
            num_workers=2,
            drop_last=False,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return self._to_dataloader(
            self.val_set,
            False,
            self.args.modelBaseConfig.val_batch_size,
            num_workers=2,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self._to_dataloader(
            self.test_set,
            False,
            self.args.modelBaseConfig.test_batch_size,
            num_workers=1,
            drop_last=False,
        )


def test_redd():
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.replace_params({'datasetConfig': 'REDD_Bitcn'})
    ds = REDD_Bitcn(args)
    ds.prepare_data()
    ds.setup('fit')
    for batch in ds.train_dataloader():
        print(batch)
        break


def test_visual():
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.replace_params({'datasetConfig': 'REDD_Bitcn'})
    ds = REDD_Bitcn(args)
    ds.visualize()
