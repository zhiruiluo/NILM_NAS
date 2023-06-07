from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import spectral_norm
import torchvision.models as models
from src.base_module.base_lightning import LightningBaseModule
from src.config_options.model_configs import ModelConfig_BasicV2
from src.config_options.option_def import MyProgramArgs

logger = logging.getLogger(__name__)


