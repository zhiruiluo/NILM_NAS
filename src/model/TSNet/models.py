from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from einops import rearrange

from src.base_module.base_lightning import LightningBaseModule
from src.config_options.model_configs import ModelConfig_TSNet
from src.config_options.option_def import MyProgramArgs

from .bitstring_decoder import convert, decode_genome
from .decoder import ConnAndOpsDecoder
from src.model.multilabel_head import MultilabelLinearFocal, MultilabelLinear, SharedBCELinear

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
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
        )
        
        if args.modelBaseConfig.label_mode == "multilabel":
            if config.head_type == 'CE':
                self.classifier = MultilabelLinear(config.out_channels, config.nclass, dropout=config.dropout)
            elif config.head_type == 'Focal':
                self.classifier = MultilabelLinearFocal(config.out_channels, config.nclass, dropout=config.dropout)
            elif config.head_type == 'SBCE':
                self.classifier = SharedBCELinear(config.out_channels, config.nclass, dropout=config.dropout)
            else:
                raise ValueError(f'invalid head type: {config.head_type}')
        elif args.modelBaseConfig.label_mode == "multiclass":
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                spectral_norm(nn.Linear(config.out_channels, config.nclass)),
            )
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=args.modelBaseConfig.label_smoothing,
            )

    def loss(self, predictions, batch):
        pred = predictions["pred"]
        target = batch["target"]
        return self.loss_fn(pred, target)

    def forward(self, batch):
        x = batch["input"]
        x = rearrange(x, 'b t c-> b c t')
        x = self.backbone(x)
        x = self.pool(x)
        
        predictions = {}
        if self.args.modelBaseConfig.label_mode == 'multilabel':
            batch['feature'] = x
            predictions = self.classifier(batch)
            # predictions["output"] = torch.round(F.sigmoid(x))
            # predictions["pred"] = torch.round(F.sigmoid(x))
        else:
            x = self.classifier(x)
            predictions["output"] = torch.max(x, dim=1)[1]
            predictions["pred"] = x
            predictions["loss"] = self.loss(predictions, batch)
        return predictions


def test_models():
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.replace_params(
        {
            "expOptions.model": "TSNet",
            "modelConfig": "TSNet",
            "modelConfig.in_channels": 2,
        },
    )
    model = TSNet(args)
    batch = {"input": torch.randn(16, 600,2)}
    print(model(batch).shape)
