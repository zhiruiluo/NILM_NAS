from __future__ import annotations

import os

from ml_toolkit.modscaner import ModuleScannerBase

from src.context import get_project_root


def model_dict_init():
    model_dict = {
        "BasicV2": "src.model.basicv2.BasicV2",
        "BasicV3_Pool": "src.model.basicv3_pool.BasicV3_Pool",
        "BitcnNILM": "src.model.BitcnNILM.BitcnNILM",
        "TSNet": "src.model.TSNet.models.TSNet",
        "RF": "src.model.RandomForest.RF",
    }
    return model_dict


class ModelSelection(ModuleScannerBase):
    def __init__(self) -> None:
        realp = os.path.dirname(os.path.realpath(__file__))
        root = os.path.relpath(realp, get_project_root())
        super().__init__(root, "Model", model_dict_init)
