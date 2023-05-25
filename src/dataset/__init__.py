import os
from pathlib import Path
import logging
from ml_toolkit.modscaner import ModuleScannerBase

from src.context import get_project_root

logger = logging.getLogger(__name__)


def dataset_dict_init():
    return {
        "REDD": "src.dataset.REDD.REDD.REDD",
        "REDD_multilabel": "src.dataset.REDD_multilabel.REDD_multilabel.REDD_multilabel",
        "REDD_Bitcn": "src.dataset.REDD_Bitcn.REDD_Bitcn.REDD_Bitcn"
    }

class DatasetSelection(ModuleScannerBase):
    def __init__(self) -> None:
        realp = os.path.dirname(os.path.realpath(__file__))
        root = os.path.relpath(realp, get_project_root())
        super().__init__(root, "Dataset", dataset_dict_init)