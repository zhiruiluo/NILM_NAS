from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import spectral_norm

from src.base_module.base_lightning import LightningBaseModule
from src.config_options.model_configs import ModelConfig_TSNet
from src.config_options.option_def import MyProgramArgs
from src.model.multilabel_head import *

from .bitstring_decoder import convert, decode_genome
from .decoder import ConnAndOpsDecoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class TSNet(LightningBaseModule):
    def __init__(self, args: MyProgramArgs):
        super().__init__(args)
        config: ModelConfig_TSNet = args.modelConfig

        list_genome = decode_genome(
            convert(np.array([int(x)
                    for x in config.bit_string]), config.n_phases),
        )
        chan = []
        if isinstance(config.out_channels, int):
            for i in range(config.n_phases):
                if i == 0:
                    chan.append((config.in_channels, config.out_channels))
                else:
                    chan.append((config.out_channels, config.out_channels))
        elif isinstance(config.out_channels, list):
            for i in range(config.n_phases):
                if i == 0:
                    chan.append((config.in_channels, config.out_channels[i]))
                else:
                    chan.append((config.out_channels[i-1], config.out_channels[i]))
        
        decoder = ConnAndOpsDecoder(list_genome, chan)
        self.backbone = decoder.get_model()
        
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
        )

        if isinstance(config.out_channels, int):
            in_dim = config.out_channels
        elif isinstance(config.out_channels, list):
            if len(decoder._channels) == 0:
                in_dim = config.in_channels
            else:
                in_dim = decoder._channels[-1][-1]
            
        if args.modelBaseConfig.label_mode == "multilabel":
            if config.head_type == 'CE':
                self.classifier = MultilabelLinear(
                    in_dim, config.nclass, dropout=config.dropout)
            elif config.head_type == 'Focal':
                self.classifier = MultilabelLinearFocal(
                    in_dim, config.nclass, dropout=config.dropout)
            elif config.head_type == 'SBCE':
                self.classifier = SharedBCELinear(
                    in_dim, config.nclass, dropout=config.dropout)
            elif config.head_type == 'ASL':
                self.classifier = MultilabelLinearASL(
                    in_dim, config.nclass, dropout=config.dropout)
            elif config.head_type == 'MaskFocal':
                self.classifier = MultilabelLinearMaskFocal(
                    in_dim, config.nclass, config.dropout)
            else:
                raise ValueError(f'invalid head type: {config.head_type}')
        elif args.modelBaseConfig.label_mode == "multiclass":
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                spectral_norm(nn.Linear(in_dim, config.nclass)),
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
        else:
            x = self.classifier(x)
            predictions["output"] = torch.max(x, dim=1)[1]
            predictions["pred"] = x
            predictions["loss"] = self.loss(predictions, batch)
        return predictions
    
    
def test_models():
    from src.config_options import OptionManager
    from src.search.nas_problem_with_channel import decode_channels
    opt = OptionManager()
    args = opt.replace_params(
        {
            "expOptions.model": "TSNet",
            "modelConfig": "TSNet",
            "modelConfig.in_channels": 1,
        },
    )
    bit_string = '101000000010111101101000000000101001100000000001111011011101'
    out_channels, new_bs = decode_channels(bit_string, 3)
    print(out_channels, new_bs)
    args.modelConfig = ModelConfig_TSNet(
        nclass=3, n_phases=3, n_ops=4, 
        bit_string=new_bs,
        in_channels=1, out_channels=out_channels, dropout=0.5, head_type='ASL')
    
    model = TSNet(args)
    # print(model)
    batch = {"input": torch.randn(16, 300, 1), 'target': torch.empty(16,3).random_(2)}
    out = model(batch)
    out['loss'].backward()
    
