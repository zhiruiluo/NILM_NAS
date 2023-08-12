from __future__ import annotations

import logging
import sys
sys.path.append('.')
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from ml_toolkit.utils.decorator import disk_buffer
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from torchvision import transforms

from src.config_options.dataset_configs import DatasetConfig_UKDALE_multilabel
from src.config_options.option_def import MyProgramArgs
from src.context import get_project_root

from src.dataset.sampler import ImbalancedDatasetSampler
from src.dataset.UKDALE_multilabel.custom_transform import ApplyPowerLabel, MinMax, NumpyToTensor, MinMaxNumpy
from src.dataset.UKDALE_multilabel.ukdale_load import dataset_by_house, get_dataset
import math
from memory_profiler import profile

logger = logging.getLogger(__name__)

data_preprocessing = {
    "appliances": {
        "on_threshold": {
            "kettle": 200,
            "microwave": 200,
            "refrigerator": 50,
            "dishwasher": 10,
            "washingmachine": 20
        },
        "mean_power": {
            "kettle": 700,
            "microwave": 500,
            "refrigerator": 200,
            "dishwasher": 700,
            "washingmachine": 400
        },
        "std_power": {
            "kettle": 1000,
            "microwave": 800,
            "refrigerator": 400,
            "dishwasher": 1000,
            "washingmachine": 700
        }
    },
    "sliding_window": {
        "train": {"win_size": 300, "stride": 60,},
        "val": {"win_size": 300, "stride":60,},
        "test": {"win_size": 300, "stride": 60,}
    }
}

app_name_mapper = {
    'house_1': {
        'aggregate': 'aggregate',
        'kettle': 'kettle',
        'refrigerator': 'fridge',
        'dishwasher': 'dishwasher',
        'washingmachine': 'washing_machine',
        'microwave': 'microwave'
    },
    'house_2': {
        'aggregate': 'aggregate',
        'kettle': 'kettle',
        'refrigerator': 'fridge',
        'dishwasher': 'dish_washer',
        'washingmachine': 'washing_machine',
        'microwave': 'microwave'
    }
}


appliances = {
    "house_1": {
        "kettle": [10],
        "fridge": [12],
        "microwave": [13],
        "dishwasher": [6],
        "washing_machine": [5],
    },
    "house_2": {
        "kettle": [8],
        "fridge": [14],
        "microwave": [15],
        "dish_washer": [13],
        "washing_machine": [12],
    },
    "house_3": {
        "kettle": [2],
        "fridge": [],
        "microwave": [],
        "dish_washer": [],
        "washing_machine": [],
    },
    "house_4": {
        "fridge": [],
        "microwave": [],
        "dish_washer": [],
        "washing_machine": [],
    },
    "house_5": {
        "kettel": [18],
        "fridge": [19],
        "microwave": [23],
        "dish_washer": [22],
        "washing_machine": [24],
    },
}

def convert_df_columns(df_house, house_no):
    reverse_name_mapper = {}
    for k,v in app_name_mapper[house_no].items():
        reverse_name_mapper[v] = k
        
    columns = []
    for col_k in df_house.columns:
        n = '_'.join(col_k.split('_')[:-1])
        mapped_name = reverse_name_mapper[n]
        columns.append(mapped_name + '_'+ col_k.split('_')[-1])
    
    df_house.columns = columns
    return df_house
    
def apply_threshold(df_target: pd.DataFrame) -> pd.DataFrame:
    for col_k in df_target.columns:
        n = ''.join(col_k.split('_')[:-1])
        app_thd = data_preprocessing["appliances"]["on_threshold"][n]
        cond = df_target[col_k] < app_thd
        df_target[col_k] = df_target[col_k].where(cond, 1).mask(cond, 0)

    return df_target


def to_memmap(df_data, memmap_path: str, keys_pairs: dict):
    input = df_data[["aggregate_1"]]
    target = df_data.drop(["aggregate_1"], axis=1)
    mask = ~pd.isna(target)
    
    target[pd.isna(target)] = 0
    target = apply_threshold(target)
    
    input = input.to_numpy()
    target = target.to_numpy()
    mask = mask.to_numpy()
    
    
    key_pairs_lst = [f'{k}={v}' for k,v in keys_pairs.items()]
    path_tmp = Path(memmap_path)
    for n, ndarray in zip(['input','target','mask'],[input,target,mask]):
        path = path_tmp.joinpath(f'memmap_{n}_'+'_'.join(key_pairs_lst)+'.npy')
        np.save(path.as_posix(), ndarray)
        logger.info(f'[to_memmap] saved {n} to path {path}')
    
def load_memmap(memmap_path, keys_pairs: dict, mode: str='r'):
    key_pairs_lst = [f'{k}={v}' for k,v in keys_pairs.items()]
    path_tmp = Path(memmap_path)
    ret = {}
    for n, dtype in zip(['input','target','mask'],[np.float64, np.float64, bool]):
        path = path_tmp.joinpath(f'memmap_{n}_'+'_'.join(key_pairs_lst)+'.npy')
        if not path.is_file():
            return None
           
        # ret[n] = np.memmap(path, dtype=dtype, mode=mode, shape=shape)
        ret[n] = np.load(path, mmap_mode=mode)
    return ret
    
def get_memmap(df_data, memmap_path, keys_pairs):
    create_flag = False
    key_pairs_lst = [f'{k}={v}' for k,v in keys_pairs.items()]
    path_tmp = Path(memmap_path)
    path_tmp.mkdir(parents=True, exist_ok=True)
    for n in ['input','target','mask']:
        path = path_tmp.joinpath(f'memmap_{n}_'+'_'.join(key_pairs_lst)+'.npy')
        if not path.is_file():
            create_flag = True
            break
    
    if create_flag:
        if df_data is None:
            return None
        to_memmap(df_data, memmap_path, keys_pairs)
    
    return load_memmap(memmap_path, keys_pairs)

def create_win_view_tempfile(win_view, dir, keys_pairs):
    key_pairs_lst = [f'{k}={v}' for k,v in keys_pairs.items()]
    key_paris_str = '_'.join(key_pairs_lst)
    win_view_path = Path(dir).joinpath('win_view_memmap')
    win_view_path.mkdir(parents=True, exist_ok=True)
    view_path = win_view_path.joinpath('winview_'+key_paris_str+'.npy')
    np.save(view_path, win_view)
    return view_path

class Sliding_Window_View():
    def __init__(self, length: int, win_size: int, stride: int) -> None:
        self._length = length 
        self.win_size = win_size
        self.stride = stride
    
    def __getitem__(self, index):
        start = index * self.stride
        end = start + self.win_size
        return np.arange(start, end)

    def __len__(self):
        return math.floor((self._length - (self.win_size - 1) - 1) / self.stride + 1)
    
    def to_numpy(self):
        w = []
        for i in self.__len__():
            w.append(self.__getitem__(i))
        return np.concatenate(w, axis=1)
    

class S2P_ML_Dataset_memmap(Dataset):
    def __init__(
        self,
        df_data,
        memmap_path: str,
        keys_pairs: dict,
        win_size: int = 300,
        stride: int = 1,
        transform = None,
    ) -> None:
        self.transform = transform
        memmaps = get_memmap(df_data, memmap_path, keys_pairs)
        self.input = memmaps['input']
        self.target = memmaps['target']
        self.mask = memmaps['mask']
        win_view = Sliding_Window_View(self.input.shape[0], win_size, stride)
        self.win_view = win_view 
        
        self.length = len(self.win_view)
        
    def __getitem__(self, index):
        win_indices = self.win_view[index]
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
        labels = self.target[self.win_view.to_numpy()[:, -1]]
        y = np.arange(labels.shape[1]) ** 2
        powerlabel = np.apply_along_axis(lambda x: np.inner(y, x), 1, labels)
        return powerlabel


class UKDALE_multilabel(pl.LightningDataModule):
    def __init__(self, args: MyProgramArgs) -> None:
        super().__init__()
        self.config: DatasetConfig_UKDALE_multilabel = args.datasetConfig
        self.prepare_data_per_node = False
        self.save_hyperparameters("args")
        self.data_root = "data/ukdale/"
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
        print(self.config.appliances)
        house_no = f'house_{self.config.house_no}'
        selected_channels = [
            appliances[house_no][app_name_mapper[house_no][app]] for app in self.config.appliances
        ]
        chs = [s_ch for s_chs in selected_channels for s_ch in s_chs]
        
        self.nclass = len(selected_channels)
        # self.num_instances = hosue_no_length[house_no]
        memmap_path = '.temp/memmap_temp/'
        keys_pairs = {'dt':'UKDALE_mls2p','house_no': self.config.house_no, 'chs': '_'.join(map(str,chs)),'dropna': self.config.drop_na_how, 'splits': self.config.splits}
        train = get_memmap(df_data=None, memmap_path=memmap_path, keys_pairs=keys_pairs|{'phase':'train'})
        val = get_memmap(df_data=None, memmap_path=memmap_path, keys_pairs=keys_pairs|{'phase':'val'})
        test = get_memmap(df_data=None, memmap_path=memmap_path, keys_pairs=keys_pairs|{'phase':'test'})
        
        if train is None or val is None or test is None:
            with disk_buffer(
                func=get_dataset,
                keys=str(self.config.house_no) + f"_ukdale_ml_s2p_{'_'.join(map(str,chs))}",
                folder=folder,
            ) as bf_get_dataset:
                df_house = bf_get_dataset(
                    house_no=self.config.house_no,
                    data_root=self.data_root,
                    channels=chs,
                    drop_na_how=self.config.drop_na_how,
                )
                df_house = convert_df_columns(df_house, house_no=house_no)
                l = len(df_house)
                
                if self.config.splits == '3:1:1':
                    val = df_house[0: round(l*0.2)]
                    test = df_house[round(l * 0.2): round(l * 0.4) ]
                    train = df_house[round(l * 0.4): ]
                elif self.config.splits == '4:1:5':
                    train = df_house[0:round(l*0.4)]
                    val = df_house[round(l*0.4): round(l*0.5)]
                    test = df_house[round(l*0.5):]
                elif self.config.splits == '3:3:4':
                    train = df_house[0:round(l*0.3)]
                    val = df_house[round(l*0.3): round(l*0.6)]
                    test = df_house[round(l*0.6):]
                elif self.config.splits == '4:2:4':
                    train = df_house[0: round(l* 0.4)].copy()
                    val = df_house[round(l* 0.4): round(l*0.6)].copy()
                    test = df_house[round(l*0.6):].copy()


                train = get_memmap(train, memmap_path=memmap_path, keys_pairs=keys_pairs|{'phase': 'train'})
                val = get_memmap(val, memmap_path=memmap_path, keys_pairs=keys_pairs|{'phase': 'val'})
                test = get_memmap(test, memmap_path=memmap_path, keys_pairs=keys_pairs|{'phase': 'test'})
                
                
        transform = transforms.Compose(
            [MinMaxNumpy(train['input']), NumpyToTensor()]
        )
        
        self.train_set = S2P_ML_Dataset_memmap(
            train,
            '.temp/memmap_temp/',
            keys_pairs|{'phase':'train'},
            win_size=self.config.win_size,
            stride=self.config.stride,
            transform=transform
        )
        self.val_set = S2P_ML_Dataset_memmap(
            val,
            '.temp/memmap_temp/',
            keys_pairs|{'phase':'val'},
            win_size=self.config.win_size,
            stride=self.config.stride,
            transform=transform
        )
        self.test_set = S2P_ML_Dataset_memmap(
            test,
            '.temp/memmap_temp/',
            keys_pairs|{'phase':'test'},
            win_size=self.config.win_size,
            stride=self.config.stride,
            transform=transform
        )

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
        if self.config.random_sampler:
            sampler = RandomSampler(self.train_set, 
                                    num_samples=round(len(self.train_set)*self.config.random_sampler_ratio))
        return self._to_dataloader(
            self.train_set,
            True,
            self.args.modelBaseConfig.batch_size,
            num_workers=1,
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


def test_ukdale():
    from pyinstrument import Profiler

    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.replace_params({'datasetConfig': 'UKDALE_multilabel',
                               'datasetConfig.random_sampler': False,
                               'datasetConfig.random_sampler_ratio': 0.3,
                               'modelBaseConfig.batch_size': 1,
                               'datasetConfig.house_no':2,
                               'datasetConfig.stride': 30,
                               'datasetConfig.win_size': 60})
    p = Profiler()
    with p:
        ds = UKDALE_multilabel(args)
        ds.prepare_data()
        ds.setup("fit")
        for batch in ds.train_dataloader():
            print(batch)
            break
    print(p.output_text())

def test_count():
    from pyinstrument import Profiler

    from src.config_options import OptionManager
    opt = OptionManager()
    args = opt.replace_params({'datasetConfig': 'UKDALE_multilabel',
                               'datasetConfig.random_sampler': True,
                               'datasetConfig.random_sampler_ratio': 0.3,
                               'datasetConfig.house_no':2,
                               'datasetConfig.stride': 30,
                               'datasetConfig.win_size': 150})
    p = Profiler()
    with p:
        
        ds = UKDALE_multilabel(args)
        ds.prepare_data()
        ds.setup("fit")
        print(len(ds.train_dataloader()))
        print(len(ds.val_dataloader()))
        ds.setup("test")
        print(len(ds.test_dataloader()))
    print(p.output_text())


def test_label_count():
    from src.config_options import OptionManager
    opt = OptionManager()
    args = opt.replace_params({'datasetConfig': 'UKDALE_multilabel',
                               'datasetConfig.splits': '4:2:4',
                               'datasetConfig.house_no':2,
                               'datasetConfig.stride': 30,
                               'datasetConfig.win_size': 60})
    
    ds = UKDALE_multilabel(args)
    ds.prepare_data()
    ds.setup("fit")
    
    positive_train = np.array([0,0,0,0,0], dtype=np.int32)
    negative_train = np.array([0,0,0,0,0], dtype=np.int32)
    for batch in ds.train_dataloader():
        npa = batch['target'].numpy()
        s = npa.sum(axis=0).astype(np.int32)
        positive_train += s
        negative_train += npa.shape[0] - s
        
    print(positive_train, negative_train)
    
    positive_val = np.array([0,0,0,0,0], dtype=np.int32)
    negative_val = np.array([0,0,0,0,0], dtype=np.int32)
    for batch in ds.val_dataloader():
        npa = batch['target'].numpy()
        s = npa.sum(axis=0).astype(np.int32)
        positive_val += s
        negative_val += npa.shape[0] - s
        
    
    
    ds.setup('test')
    positive_test = np.array([0,0,0,0,0], dtype=np.int32)
    negative_test = np.array([0,0,0,0,0], dtype=np.int32)
    for batch in ds.test_dataloader():
        npa = batch['target'].numpy()
        s = npa.sum(axis=0).astype(np.int32)
        positive_test += s
        negative_test += npa.shape[0] - s

    print('train',positive_train, negative_train)
    print('val  ', positive_val, negative_val)
    print('test ',positive_test, negative_test)
    
    
def test_visual():
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.args
    ds = UKDALE_multilabel(args)
    ds.visualize()

if __name__ == '__main__':
    test_count()