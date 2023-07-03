from __future__ import annotations

import glob
import re
from collections import defaultdict
from pathlib import Path

from typing import Literal
import pandas as pd


def natural_sort_key(s, _nsre=re.compile("([0-9]+)")):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


def parse_labels(path) -> dict:
    labels = pd.read_csv(path, sep=" ", header=None, index_col=0)
    labels = labels.to_dict()[1]
    for k in labels.keys():
        labels[k] = labels[k] + "_" + str(k)
    return labels


def read_merge_data(path, labels, channels, merge_how="left"):
    file = Path(path).joinpath("channel_1.dat")
    # file = path + 'channel_1.dat'
    df = pd.read_table(
        file,
        sep=" ",
        names=["unix_time", labels[1]],
        dtype={"unix_time": "int64", labels[1]: "float64"},
    )

    num_apps = len(glob.glob(path + "/channel*"))
    print(num_apps)
    for i in channels:
        file = path + f"/channel_{i}.dat"
        data = pd.read_table(
            file,
            sep=" ",
            names=["unix_time", labels[i]],
            dtype={"unix_time": "int64", labels[i]: "float64"},
        )
        df = pd.merge(df, data, how=merge_how, on="unix_time")
    df["timestamp"] = df["unix_time"].astype("datetime64[s]")
    df = df.set_index(df["timestamp"].values)
    df.drop(["unix_time", "timestamp"], axis=1, inplace=True)

    return df


def list_df_to_dict_list(li_df: list[pd.DataFrame]) -> dict[str, list]:
    dc = defaultdict(list)
    for df in li_df:
        for k, v in df.to_dict("list").items():
            dc[k].append(v)
    return dc


def get_dataset(
    house_no: int,
    data_root: str,
    channels: list,
    sample_seconds: int = 6,
    fillna_limit=1,
    drop_na_how: Literal['any','all'] = 'any'
):

    data_root = Path(data_root)
    path = data_root.joinpath(f"house_{house_no}")
    labels = parse_labels(path.joinpath("labels.dat"))
    df_house = read_merge_data(str(path), labels, [2] + channels, merge_how="outer")
    df_house = (
        df_house.resample(str(sample_seconds) + "S")
        .mean()
        .fillna(method="backfill", limit=fillna_limit)
    )
    df_house = df_house.dropna(how=drop_na_how)
    # l = len(df_house)

    # train = df_house[0 : round(0.6 * l)]
    # val = df_house[round(0.6 * l) : round(0.8 * l)]
    # test = df_house[round(0.8 * l) :]
    # return train, val, test
    return df_house


def test_get_dataset():
    train, val, test = get_dataset(1, "data/low_freq", [5, 6, 11, 20], sample_seconds=6)
    print(train.shape, val.shape, test.shape)


def dataset_by_house(house_no: int, data_root: str):
    data_root = Path(data_root)
    path = data_root.joinpath(f"house_{house_no}")
    labels = parse_labels(path.joinpath("labels.dat"))
    df_house = read_merge_data(str(path), labels)

    df_30mins = df_house.groupby(pd.Grouper(freq="30min"))
    all_30mins = []
    for name, group in df_30mins:
        if len(group) < 1800:
            continue
        df_30 = group.interpolate(limit=10, limit_area=None)
        cols_to_binarize = df_30.columns[2:]
        for col_name in cols_to_binarize:
            threshold = 5
            df_30[col_name] = df_30[col_name].apply(lambda x: 1 if x > threshold else 0)
        all_30mins.append(df_30)

    l = len(all_30mins)

    train = list_df_to_dict_list(all_30mins[0 : round(0.6 * l)])
    val = list_df_to_dict_list(all_30mins[round(0.6 * l) : round(0.8 * l)])
    test = list_df_to_dict_list(all_30mins[round(0.8 * l) :])

    return train, val, test


def test_dataset_by_house():
    dataset_by_house(1, "data/low_freq")


def test_read_merge_data():
    path = Path("data/low_freq/house_1")
    labels = parse_labels(path.joinpath("labels.dat"))
    print(labels)
    df = read_merge_data(path.as_posix(), labels)
    df_30mins = df.groupby(pd.Grouper(freq="30min"))
    all_30mins = []
    for name, group in df_30mins:
        if len(group) < 1800:
            continue
        df_30 = group.interpolate(limit=10, limit_area=None)
        cols_to_binarize = df_30.columns[2:]
        for col_name in cols_to_binarize:
            threshold = 5
            df_30[col_name] = df_30[col_name].apply(lambda x: 1 if x > threshold else 0)
        all_30mins.append(df_30)
