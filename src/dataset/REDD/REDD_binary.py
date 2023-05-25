from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from ml_toolkit.utils.decorator import disk_buffer
from torch.utils.data import DataLoader, Dataset

from src.config_options.dataset_configs import DatasetConfig_REDD
from src.config_options.option_def import MyProgramArgs
from src.context import get_project_root

from .redd_load import dataset_by_house


class BinaryDataset(Dataset):
    def __init__(self, dc, appliance: str, seq_to_point: bool = True) -> None:
        self.dc = dc
        self.seq_to_point = seq_to_point

        mains_1 = torch.tensor(self.dc.get("mains_1", None), dtype=torch.float32)
        mains_2 = torch.tensor(self.dc.get("mains_2", None), dtype=torch.float32)
        self.input = torch.stack((mains_1, mains_2), axis=1)
        keys = set(self.dc.keys())
        keys.discard("mains_1")
        keys.discard("mains_2")
        keys = sorted(list(keys))

        labels = None
        for k in keys:
            if k == appliance:
                labels = torch.tensor(self.dc[k], dtype=torch.float32)

        self.target = labels
        self._len = self.input.shape[0]

    def __getitem__(self, index):
        return {"input": self.input[index], "target": self.target[index, -1]}

    def __len__(self):
        return self._len


def apply_transform(
    train: dict[str, list],
    val: dict[str, list],
    test: dict[str, list],
    t,
):
    for f in ["mains_1", "mains_2"]:
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

    df_train_last["powerlabel"] = df_train_last.apply(power, axis=1).astype(int)
    from imblearn.over_sampling import RandomOverSampler

    sampler = RandomOverSampler()
    # df_train_last['powerlabel'].hist(figsize=(5,5))
    # plt.savefig('results/powerlabel.png')

    df_train_resampled, df_y_resampled = sampler.fit_resample(
        df_train,
        df_train_last["powerlabel"],
    )
    # df_y_resampled.hist(figsize=(5,5))
    # plt.savefig('results/resampled.png')
    return df_train_resampled.to_dict("list")


class REDD_binary(pl.LightningDataModule):
    def __init__(self, args: MyProgramArgs) -> None:
        super().__init__()
        self.config: DatasetConfig_REDD = args.datasetConfig
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

    def prepare_data(self) -> None:
        folder = get_project_root().joinpath(".temp").as_posix()
        with disk_buffer(
            func=dataset_by_house,
            keys=str(self.config.house_no),
            folder=folder,
        ) as bf_dataset_by_house:
            train, val, test = bf_dataset_by_house(self.config.house_no, self.data_root)

        train, val, test = minmax(train, val, test)
        train = data_augmentation(train)
        self.train_set = MultilabelDataset(train)
        self.val_set = MultilabelDataset(val)
        self.test_set = MultilabelDataset(test)
        self.nclass = len(train) - 2

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
        return self._to_dataloader(
            self.train_set,
            True,
            self.args.modelBaseConfig.batch_size,
            num_workers=0,
            drop_last=False,
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
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.args
    ds = REDD(args)
    ds.prepare_data()
    ds.setup("fit")
    for batch in ds.train_dataloader():
        print(batch)
        break


def test_visual():
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.args
    ds = REDD(args)
    ds.visualize()
