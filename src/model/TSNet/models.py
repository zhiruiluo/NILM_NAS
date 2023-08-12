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


def lstm(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0,bidirectional=False):
    return nn.LSTM(input_size,hidden_size,num_layers,bias,batch_first,dropout,bidirectional)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class LSTM_subnet(nn.Module):
    def __init__(self, input_size, input_seq_size, hidden_size, out_features=32, dropout=0.5) -> None:
        super().__init__()
        self.bilstm = lstm(input_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm((input_seq_size,hidden_size))
        self.leaky_relu = nn.LeakyReLU()
        self.lstm = lstm(hidden_size, hidden_size, batch_first=True)
        self.lazy_linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_size, out_features)),
        )
        
    def forward(self, x):
        # good 1: leaky_rely, ln (seq, hidden)
        output, (hn, cn) = self.bilstm(x)
        output = rearrange(output, "b t (d h) -> b t h d", d=2)
        output = torch.sum(output, dim=3)
        output = self.leaky_relu(output)
        output = self.ln(output)
        _, (hn, cn) = self.lstm(output)
        print(hn.shape)
        return self.lazy_linear(hn)


class TSNet(LightningBaseModule):
    def __init__(self, args: MyProgramArgs):
        super().__init__(args)
        config: ModelConfig_TSNet = args.modelConfig
        self.config = config
        
        if self.config.lstm_out_features > 0:
            self.lstm_sub = LSTM_subnet(self.config.in_channels, args.datasetConfig.win_size, config.lstm_hidden_features, out_features=config.lstm_out_features, dropout=config.dropout)
        else:
            self.lstm_sub = None
        
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
        
        decoder = ConnAndOpsDecoder(list_genome, chan, attention=config.atten)
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
        
        if config.lstm_out_features > 0:
            in_dim += config.lstm_out_features
        
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
        # if self.config.temporal_atten:
        #     x = self.temporal_atten(x) * x + x
        x = self.pool(x)

        if self.lstm_sub:
            lstm_out = self.lstm_sub(batch["input"])
            x = torch.concatenate((x,lstm_out), dim=1)
        
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
            "datasetConfig":'REDD_multilabel',
            "expOptions.model": "TSNet",
            "modelConfig": "TSNet",
            "modelConfig.in_channels": 1,
            "datasetConfig.win_size": 60,
        },
    )
    bit_string = '111111011101100101101000110100010001110110011001111010010000110000001101101000'
    # out_channels, new_bs = decode_channels(bit_string, 3)
    # print(out_channels, new_bs)
    args.modelConfig = ModelConfig_TSNet(
        nclass=3, n_phases=3, n_ops=5, 
        bit_string=bit_string,
        in_channels=1, out_channels=[32, 32, 64], dropout=0.5, head_type='ASL', atten=True, lstm_out_features=32,)
    
    model = TSNet(args)
    # print(model)
    batch = {"input": torch.randn(1, 60, 1), 'target': torch.empty(1,3).random_(2)}
    out = model(batch)
    print(out)
    out['loss'].backward()
    
