from __future__ import annotations

import logging
from typing import List
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import spectral_norm

from src.base_module.base_lightning import LightningBaseModule
from src.config_options.model_configs import ModelConfig_BasicV2
from src.config_options.option_def import MyProgramArgs
from src.utilts import get_padding_one_more_or_same


##
# dilation
# number of layers
# filters and kernel stride
# groups
# skip connections
#
class prototype(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


def test_prototype():
    prototype()
