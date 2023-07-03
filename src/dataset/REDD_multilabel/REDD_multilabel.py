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

data_preprocessing = {
    "appliances": {
        "on_threshold": {
            "kettle": 200,
            "microwave": 200,
            "refrigerator": 50,
            "dishwasher": 10,
            "washer_dryer": 20
        },
        "mean_power": {
            "kettle": 700,
            "microwave": 500,
            "refrigerator": 200,
            "dishwasher": 700,
            "washer_dryer": 400
        },
        "std_power": {
            "kettle": 1000,
            "microwave": 800,
            "refrigerator": 400,
            "dishwasher": 1000,
            "washer_dryer": 700
        }
    },
    "sliding_window": {
        "train": {"win_size": 300, "stride": 60,},
        "val": {"win_size": 300, "stride":60,},
        "test": {"win_size": 300, "stride": 60,}
    }
}


appliances = {
    "house_1": {
        "refrigerator": [5],
        "microwave": [11],
        "dishwasher": [6],
        "washer_dryer": [20],
    },
    "house_2": {
        "refrigerator": [9],
        "microwave": [6],
        "dishwasher": [10],
        "washer_dryer": [7],
    },
    "house_3": {
        "refrigerator": [7],
        "microwave": [16],
        "dishwasher": [9],
        "washer_dryer": [13],
    },
    "house_4": {
        "refrigerator": [],
        "microwave": [],
        "dishwasher": [15],
        "washer_dryer": [7],
    },
    "house_5": {
        "refrigerator": [18],
        "microwave": [3],
        "dishwasher": [20],
        "washer_dryer": [9],
    },
    "house_6": {
        "refrigerator": [8],
        "microwave": [],
        "dishwasher": [9],
        "washer_dryer": [4],
    },
}

def apply_threshold(df_target: pd.DataFrame) -> pd.DataFrame:
    for col_k in df_target.columns:
        n = '_'.join(col_k.split('_')[:-1])
        if n == 'dishwaser':
            n = 'dishwasher'
        app_thd = data_preprocessing["appliances"]["on_threshold"][n]
        cond = df_target[col_k] < app_thd
        df_target[col_k] = df_target[col_k].where(cond, 1).mask(cond, 0)

    return df_target


class Seq2PointMultilabelDataset(Dataset):
    def __init__(
        self,
        df_data: pd.DataFrame,
        combine_mains: bool = False,
        win_size: int = 300,
        stride: int = 1,
        transform=None,
    ) -> None:
        self.transform = transform

        if not combine_mains:
            input = df_data[["mains_1", "mains_2"]]
            target = df_data.drop(["mains_1", "mains_2"], axis=1)
        else:
            input = df_data[['mains_comb']]
            target = df_data.drop(["mains_1", "mains_2","mains_comb"], axis=1)
        
        mask = ~pd.isna(target)
        self.mask = mask.to_numpy()
        
        target[pd.isna(target)] = 0
        target = apply_threshold(target)

        self.input = input.to_numpy()
        self.target = target.to_numpy()

        self.win_view = sliding_window_view(
            np.arange(self.input.shape[0]),
            window_shape=win_size,
        )
        self.indices = np.arange(0, self.win_view.shape[0], stride)
        self.length = len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        win_indices = self.win_view[idx]
        sample = {
            "input": self.input[win_indices],
            "target": self.target[win_indices[-1]],
            "mask": self.mask[win_indices[-1]]
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
        return powerlabel


class REDD_multilabel(pl.LightningDataModule):
    def __init__(self, args: MyProgramArgs) -> None:
        super().__init__()
        self.config: DatasetConfig_REDD_multilabel = args.datasetConfig
        self.prepare_data_per_node = False
        self.save_hyperparameters("args")
        self.data_root = "data/low_freq/"
        self.args = args

    def prepare_data(self):
        folder = get_project_root().joinpath(".temp").as_posix()

        selected_channels = [
            appliances[f"house_{self.config.house_no}"][app] for app in self.config.appliances
        ]
        chs = [s_ch for s_chs in selected_channels for s_ch in s_chs]
        with disk_buffer(
            func=get_dataset,
            keys=str(self.config.house_no) + f"_redd_ml_s2p_{'_'.join(map(str,chs))}",
            folder=folder,
        ) as bf_get_dataset:
            df_house = bf_get_dataset(
                house_no=self.config.house_no,
                data_root=self.data_root,
                channels=chs,
                drop_na_how=self.config.drop_na_how,
            )
            
            l = len(df_house)
            # val = df_house[0: round(l * 0.16)]
            # test = df_house[round(l * 0.16): round(l * 0.36) ]
            # train = df_house[round(l * 0.36): ]

            if self.config.splits == '3:1:1':
                if self.config.house_no == 3:
                    val = df_house[0: round(l * 0.2)].copy()
                    test = df_house[round(l * 0.2): round(l * 0.4) ].copy()
                    train = df_house[round(l * 0.4): ].copy()
                elif self.config.house_no == 1:
                    train = df_house[0:round(l*0.6)].copy()
                    val = df_house[round(l*0.6): round(l*0.8)].copy()
                    test = df_house[round(l*0.8):].copy()
            elif self.config.splits == '3:3:4':
                # if self.config.house_no == 3:
                train = df_house[0: round(l * 0.3)].copy()
                val = df_house[round(l * 0.3): round(l * 0.6) ].copy()
                test = df_house[round(l * 0.6): ].copy()
            elif self.config.splits == '4:2:4':
                train = df_house[0: round(l* 0.4)].copy()
                val = df_house[round(l* 0.4): round(l*0.6)].copy()
                test = df_house[round(l*0.6):].copy()

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
            combine_mains=self.config.combine_mains,
            win_size=self.config.win_size,
            stride=self.config.stride,
            transform=transform,
        )
        self.val_set = Seq2PointMultilabelDataset(
            val,
            combine_mains=self.config.combine_mains,
            win_size=self.config.win_size,
            stride=self.config.stride,
            transform=transform,
        )
        self.test_set = Seq2PointMultilabelDataset(
            test,
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
        sampler = None
        if self.config.imbalance_sampler:
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
            num_workers=0,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self._to_dataloader(
            self.test_set,
            False,
            self.args.modelBaseConfig.test_batch_size,
            num_workers=0,
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

def test_count():
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.replace_params({'datasetConfig': 'REDD_multilabel',
                               'datasetConfig.splits': '4:2:4',
                               'datasetConfig.house_no': 3,
                               'datasetConfig.stride': 5, 
                               'datasetConfig.win_size': 60,
                               'datasetConfig.combine_mains': True})
    ds = REDD_multilabel(args)
    ds.prepare_data()
    ds.setup("fit")
    positive = np.array([0,0,0,0], dtype=np.int32)
    negative = np.array([0,0,0,0], dtype=np.int32)
    for batch in ds.train_dataloader():
        npa = batch['target'].numpy()
        s = npa.sum(axis=0).astype(np.int32)
        positive += s
        negative += npa.shape[0] - s
        
    print(positive, negative)
    
    positive = np.array([0,0,0,0], dtype=np.int32)
    negative = np.array([0,0,0,0], dtype=np.int32)
    for batch in ds.val_dataloader():
        npa = batch['target'].numpy()
        s = npa.sum(axis=0).astype(np.int32)
        positive += s
        negative += npa.shape[0] - s
        
    print(positive, negative)
    
    ds.setup('test')
    positive = np.array([0,0,0,0], dtype=np.int32)
    negative = np.array([0,0,0,0], dtype=np.int32)
    for batch in ds.test_dataloader():
        npa = batch['target'].numpy()
        s = npa.sum(axis=0).astype(np.int32)
        positive += s
        negative += npa.shape[0] - s
        
    print(positive, negative)