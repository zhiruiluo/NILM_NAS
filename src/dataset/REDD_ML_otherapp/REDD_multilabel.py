from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from ml_toolkit.utils.decorator import disk_buffer
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config_options.dataset_configs import DatasetConfig_REDD_multilabel
from src.config_options.option_def import MyProgramArgs
from src.context import get_project_root

from ..sampler import ImbalancedDatasetSampler
from .custom_transform import ApplyPowerLabel, MinMax, NumpyToTensor
from .redd_load import dataset_by_house, get_dataset

appliances = {
    "house_1": {
        'train':{
            "refrigerator": [5],
            "microwave": [11],
            "dishwasher": [6],
            "washer_dryer": [20],
        },
        "test": {
            "refrigerator": [5],
            "microwave": [11],
            "dishwasher": [6],
            "washer_dryer": [19],
        }
    },
    "house_2": {
        "refrigerator": [9],
        "microwave": [6],
        "dishwasher": [10],
        "washer_dryer": [7],
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
        if not combine_mains:
            self.input = df_data[["mains_1", "mains_2"]].to_numpy()
            target = df_data.drop(["mains_1", "mains_2"], axis=1)
        else:
            self.input = df_data[['mains_comb']].to_numpy()
            target = df_data.drop(["mains_1", "mains_2","mains_comb"], axis=1)
        threshold = 10
        target = target.applymap(
            lambda x: 1 if x > threshold else 0,
        )
        self.target = target.to_numpy()

        self.win_view = sliding_window_view(
            np.arange(self.input.shape[0]),
            window_shape=win_size,
        )
        # selftarget_win_view = sliding_window_view(np.arange(self.input.shape[0]), window_shape=win_size)
        self.indices = np.arange(0, self.win_view.shape[0], stride)
        self.length = len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        win_indices = self.win_view[idx]
        sample = {
            "input": self.input[win_indices],
            "target": self.target[win_indices[-1]],
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


class REDD_multilabel(pl.LightningDataModule):
    def __init__(self, args: MyProgramArgs) -> None:
        super().__init__()
        self.config: DatasetConfig_REDD_multilabel = args.datasetConfig
        self.prepare_data_per_node = False
        self.save_hyperparameters("args")
        self.data_root = "data/low_freq/"
        self.args = args

    def visualize(self):
        folder = get_project_root().joinpath(".temp").as_posix()
        with disk_buffer(
            func=dataset_by_house,
            keys=str(self.config.house_no),
            folder=folder,
        ) as bf_dataset_by_house:
            train, val, test = bf_dataset_by_house(self.config.house_no, self.data_root)

        df_train = pd.DataFrame.from_dict(train)
        df_train_last = df_train.applymap(lambda x: x[-1])
        print(df_train_last)
        # classes = df_train_last.columns[2:]

        def power(x):
            c = 0
            for i, v in enumerate(x[2:], 1):
                c += i * v
            return c

        df_train_last["powerlabel"] = df_train_last.apply(power, axis=1).astype(int)

        df_train_last.hist(figsize=(10, 10))
        plt.savefig("results/power_label.png")

    def prepare_data(self):
        folder = get_project_root().joinpath(".temp").as_posix()

        selected_channels = [
            appliances[f"house_{self.config.house_no}"][app] for app in self.config.appliances
        ]
        with disk_buffer(
            func=get_dataset,
            keys=str(self.config.house_no) + "redd_ml_s2p",
            folder=folder,
        ) as bf_get_dataset:
            train, val, test = bf_get_dataset(
                house_no=self.config.house_no,
                data_root=self.data_root,
                channels=[s_ch for s_chs in selected_channels for s_ch in s_chs],
            )

        if self.config.combine_mains:
            train['mains_comb'] = train[['mains_1','mains_2']].sum(axis=1)
            val['mains_comb'] = val[['mains_1','mains_2']].sum(axis=1)
            test['mains_comb'] = test[['mains_1','mains_2']].sum(axis=1)
            transform = transforms.Compose(
                [MinMax(train, ['mains_comb']), NumpyToTensor()] 
            )
        else:
            transform = transforms.Compose(
                [MinMax(train, ["mains_1", "mains_2"]), NumpyToTensor()],
            )
        self.train_set = Seq2PointMultilabelDataset(
            train,
            selected_channels,
            combine_mains=self.config.combine_mains,
            win_size=self.config.win_size,
            stride=self.config.stride,
            transform=transform,
        )
        self.val_set = Seq2PointMultilabelDataset(
            val,
            selected_channels,
            combine_mains=self.config.combine_mains,
            win_size=self.config.win_size,
            stride=self.config.stride,
            transform=transform,
        )
        self.test_set = Seq2PointMultilabelDataset(
            test,
            selected_channels,
            combine_mains=self.config.combine_mains,
            win_size=self.config.win_size,
            stride=self.config.stride,
            transform=transform,
        )
        self.nclass = len(selected_channels)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            ...
        if stage == "test":
            ...

    def _to_dataloader(
        self,
        dataset,
        shuffle,
        batch_size,
        num_workers,
        drop_last,
        sampler=None,
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
    args = opt.replace_params({'datasetConfig': 'REDD_multilabel'})
    p = Profiler()
    with p:
        ds = REDD_multilabel(args)
        ds.prepare_data()
        ds.setup("fit")
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
