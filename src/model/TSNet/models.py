from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from src.base_module.base_lightning import LightningBaseModule
from src.config_options.model_configs import ModelConfig_TSNet
from src.config_options.option_def import MyProgramArgs

from .bitstring_decoder import convert, decode_genome
from .decoder import ConnAndOpsDecoder


class TSNet(LightningBaseModule):
    def __init__(self, args: MyProgramArgs):
        super().__init__(args)
        config: ModelConfig_TSNet = args.modelConfig

        list_genome = decode_genome(
            convert(np.array([int(x) for x in config.bit_string]), config.n_phases),
        )
        chan = []
        for i in range(config.n_phases):
            if i == 0:
                chan.append((config.in_channels, config.out_channels))
            else:
                chan.append((config.out_channels, config.out_channels))
        self.backbone = ConnAndOpsDecoder(list_genome, chan).get_model()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(config.dropout),
            spectral_norm(nn.Linear(config.out_channels, config.nclass)),
        )

    def forward(self, batch):
        x = batch["input"]
        x = self.backbone(x)
        print(x.shape)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def test_models():
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.replace_params(
        {
            "expOptions.model": "TSNet",
            "modelConfig": "TSNet",
        },
    )
    model = TSNet(args)
    batch = {"input": torch.randn(1, 1, 12, 256)}
    print(model(batch).shape)
