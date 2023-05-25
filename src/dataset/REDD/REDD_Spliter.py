from __future__ import annotations

import logging

import numpy as np
from ml_toolkit.datautils.data_spliter import IndexSpliter
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class REDDSpliter(IndexSpliter):
    def __init__(
        self,
        key_name,
        labels,
        splits: str,
        nrepeat: int,
        index_buffer_flag: bool = True,
        index_buffer_path: str = "./spliting/",
        seed: int = 42,
    ) -> None:
        super().__init__(
            key_name,
            labels,
            splits,
            nrepeat,
            index_buffer_flag,
            index_buffer_path,
        )
        self.seed = seed

    def _splits_(self, labels):
        if self.splits == "3:1:1":
            train_index, test_index = train_test_split(
                np.arange(len(labels)),
                train_size=0.8,
                stratify=labels,
                shuffle=True,
                random_state=self.seed,
            )
            train_labels = labels[train_index]
            train_index, val_index = train_test_split(
                train_index,
                train_size=0.75,
                stratify=train_labels,
                shuffle=True,
                random_state=self.seed,
            )
        elif self.splits == "3:1:1t":
            train_index, test_index = train_test_split(
                np.arange(len(labels)),
                train_size=0.8,
                stratify=labels,
                shuffle=True,
                random_state=self.seed,
            )
            train_labels = labels[train_index]
            train_index, val_index = train_test_split(
                train_index,
                train_size=0.75,
                stratify=train_labels,
                shuffle=True,
                random_state=self.seed,
            )

        return train_index, val_index, test_index
