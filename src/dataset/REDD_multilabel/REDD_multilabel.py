from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from ml_toolkit.utils.decorator import disk_buffer
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from ..sampler import ImbalancedDatasetSampler
from .custom_transform import ApplyPowerLabel
from .custom_transform import MinMax
from .custom_transform import NumpyToTensor
from .redd_load import dataset_by_house
from .redd_load import get_dataset
from src.base_module.base_lightningdata import LightningBaseDataModule
from src.config_options.dataset_configs import DatasetConfig_REDD_multilabel
from src.config_options.option_def import MyProgramArgs
from src.context import get_project_root

appliances = {
    'house_1': {
        'refrigerator': [5],
        'microwave': [11],
        'dishwasher': [6],
        'washer_dryer': [20],
    },
    'house_2': {
        'refrigerator': [9],
        'microwave': [6],
        'dishwasher': [10],
        'washer_dryer': [7],
    },
}


class Seq2PointMultilabelDataset(Dataset):
    def __init__(
        self,
        df_data: pd.DataFrame,
        selected_channels: list[str],
        combine_mains: bool = False,
        win_size: int = 300,
        stride: int = 1,
        transform=None,
    ) -> None:
        self.selected_channels = selected_channels
        self.transform = transform

        self.input = df_data[['mains_1', 'mains_2']].to_numpy()
        threshold = 10
        target = df_data.drop(['mains_1', 'mains_2'], axis=1).applymap(
            lambda x: 1 if x > threshold else 0,
        )
        self.target = target.to_numpy()

        self.win_view = sliding_window_view(
            np.arange(self.input.shape[0]), window_shape=win_size,
        )
        # selftarget_win_view = sliding_window_view(np.arange(self.input.shape[0]), window_shape=win_size)
        self.indices = np.arange(0, self.win_view.shape[0], stride)
        self.length = len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        win_indices = self.win_view[idx]
        sample = {
            'input': self.input[win_indices, 0],
            'target': self.target[win_indices[-1]],
        }
        if self.transform:
            return self.transform(sample)
        return sample

    def __len__(self):
        return self.length

    def get_labels(self):
        labels = self.target[self.win_view[self.indices][:, -1]]
        y = np.arange(labels.shape[1]) ** 2
        powerlabel = np.apply_along_axis(lambda x: np.inner(y, x), 1, labels)
        # print(powerlabel)
        return powerlabel


class MultilabelDataset(Dataset):
    def __init__(
        self,
        dc,
        selected_channels: list[str],
        seq_to_point: bool = True,
        combine_mains: bool = False,
        transform=None,
    ) -> None:
        self.dc = dc
        self.selected_channels = selected_channels
        self.seq_to_point = seq_to_point
        self.transform = transform

        self.input = np.stack(
            (self.dc.get('mains_1', None), self.dc.get('mains_2', None)),
            dtype=np.float32,
        )

        keys = set(self.dc.keys())
        keys.discard('mains_1')
        keys.discard('mains_2')
        keys = sorted(list(keys))

        labels = []
        set_chs = set()
        for chs in selected_channels:
            for ch in chs:
                set_chs.add(str(ch))

        for k in keys:
            if k.split('_')[-1] in set_chs:
                # print(np.shape(self.dc[k]))
                # exit()
                labels.append(np.array(self.dc[k]))

        self.target = np.stack(labels, axis=2, dtype=np.float32)
        self._len = self.input.shape[0]

    def __getitem__(self, index):
        sample = {'input': self.input[index], 'target': self.target[index, -1]}
        if self.transform:
            return self.transform(sample)
        return sample

    def __len__(self):
        return self._len

    # def get_labels(self):
    #     powerlabel = np.apply_along_axis(
    #         lambda x: np.sum(np.arange(len(x))**2*x),
    #         1, self.target[:,-1,:])
    #     print(powerlabel)
    #     return powerlabel.tolist()


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


class REDD_multilabel(pl.LightningDataModule):
    def __init__(self, args: MyProgramArgs) -> None:
        super().__init__()
        self.config: DatasetConfig_REDD_multilabel = args.datasetConfig
        self.prepare_data_per_node = False
        self.save_hyperparameters('args')
        self.data_root = 'data/low_freq/'
        self.args = args

    def visualize(self):
        folder = get_project_root().joinpath('.temp').as_posix()
        with disk_buffer(
            func=dataset_by_house, keys=str(self.config.house_no), folder=folder,
        ) as bf_dataset_by_house:
            train, val, test = bf_dataset_by_house(
                self.config.house_no, self.data_root)

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

    def prepare_data_(self) -> None:
        folder = get_project_root().joinpath('.temp').as_posix()
        with disk_buffer(
            func=dataset_by_house, keys=str(self.config.house_no), folder=folder,
        ) as bf_dataset_by_house:
            train, val, test = bf_dataset_by_house(
                self.config.house_no, self.data_root)

        selected_channels = [
            appliances[f'house_{self.config.house_no}'][app]
            for app in self.config.appliances
        ]
        transform = transforms.Compose(
            [
                MinMax(train, ['mains_1', 'mains_2']),
                # ApplyPowerLabel(),
                NumpyToTensor(),
            ],
        )
        self.train_set = MultilabelDataset(
            train, selected_channels, transform=transform,
        )
        self.val_set = MultilabelDataset(
            val, selected_channels, transform=transform)
        self.test_set = MultilabelDataset(
            test, selected_channels, transform=transform)
        self.nclass = len(train) - 2

    def prepare_data(self):
        folder = get_project_root().joinpath('.temp').as_posix()

        selected_channels = [
            appliances[f'house_{self.config.house_no}'][app]
            for app in self.config.appliances
        ]
        with disk_buffer(
            func=get_dataset,
            keys=str(self.config.house_no) + 'redd_ml_s2p',
            folder=folder,
        ) as bf_get_dataset:
            train, val, test = bf_get_dataset(
                house_no=self.config.house_no,
                data_root=self.data_root,
                channels=[s_ch for s_chs in selected_channels for s_ch in s_chs],
            )

        transform = transforms.Compose(
            [MinMax(train, ['mains_1', 'mains_2']),
             ApplyPowerLabel(), NumpyToTensor()],
        )
        self.train_set = Seq2PointMultilabelDataset(
            train,
            selected_channels,
            win_size=self.config.win_size,
            stride=self.config.stride,
            transform=transform,
        )
        self.val_set = Seq2PointMultilabelDataset(
            val,
            selected_channels,
            win_size=self.config.win_size,
            stride=self.config.stride,
            transform=transform,
        )
        self.test_set = Seq2PointMultilabelDataset(
            test,
            selected_channels,
            win_size=self.config.win_size,
            stride=self.config.stride,
            transform=transform,
        )
        self.nclass = len(selected_channels)

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
            num_workers=0,
            drop_last=False,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return self._to_dataloader(
            self.val_set,
            False,
            self.args.modelBaseConfig.val_batch_size,
            num_workers=1,
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
    from pyinstrument import Profiler

    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.args
    p = Profiler()
    with p:
        ds = REDD_multilabel(args)
        ds.prepare_data()
        ds.setup('fit')
        for batch in ds.train_dataloader():
            print(batch)
            break
    print(p.output_text())


def test_visual():
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.args
    ds = REDD_multilabel(args)
    ds.visualize()
