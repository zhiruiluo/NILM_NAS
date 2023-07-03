from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from sqlalchemy import Column, DateTime, Float, Integer, Interval, String, Table
from sqlalchemy.orm import registry

mapper_registry = registry()


@mapper_registry.mapped
@dataclass
class Results:
    __table__ = Table(
        "results",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("model", String),
        Column("model_params", String),
        Column("dataset", String),
        Column("data_params", String),
        Column("train_acc", Float),
        Column("train_f1macro", Float),
        Column("train_f1micro", Float),
        Column("train_loss", Float),
        Column("train_confmx", String),
        Column("val_acc", Float),
        Column("val_f1macro", Float),
        Column("val_f1micro", Float),
        Column("val_loss", Float),
        Column("val_confmx", String),
        Column("test_acc", Float),
        Column("test_f1macro", Float),
        Column("test_f1micro", Float),
        Column("test_loss", Float),
        Column("test_confmx", String),
        Column("start_time", DateTime),
        Column("training_time", Interval),
        Column("flops", Integer),
        Column("params", String),
        Column("nas_params", String),
    )
    id: int = field(init=False)
    model: str = None
    model_params: str = None
    dataset: str = None
    data_params: str = None
    train_acc: float = None
    train_f1macro: float = None
    train_f1micro: float = None
    train_loss: float = None
    train_confmx: str = None
    val_acc: float = None
    val_f1macro: float = None
    val_f1micro: float = None
    val_loss: float = None
    val_confmx: str = None
    test_acc: float = None
    test_f1micro: float = None
    test_f1macro: float = None
    test_loss: float = None
    test_confmx: str = None
    start_time: datetime.datetime = None
    training_time: datetime.timedelta = None
    flops: int = None
    params: str = None
    nas_params: str = None


@mapper_registry.mapped
@dataclass
class MultilabelResults:
    __table__ = Table(
        "multilabel_results",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("model", String),
        Column("model_params", String),
        Column("dataset", String),
        Column("data_params", String),
        Column("train_accmacro", Float),
        Column("train_f1macro", Float),
        Column("val_accmacro", Float),
        Column("val_f1macro", Float),
        Column("test_accmacro", Float),
        Column("test_f1macro", Float),
        Column("start_time", DateTime),
        Column("training_time", Interval),
        Column("macs", Integer),
        Column("flops", Integer),
        Column("params", String),
        Column("nas_params", String),
    )
    id: int = field(init=False)
    model: str = None
    model_params: str = None
    dataset: str = None
    data_params: str = None
    train_accmacro: float = None
    train_f1macro: float = None
    val_accmacro: float = None
    val_f1macro: float = None
    test_accmacro: float = None
    test_f1macro: float = None
    start_time: datetime.datetime = None
    training_time: datetime.timedelta = None
    macs: int = None
    flops: int = None
    params: str = None
    nas_params: str = None
