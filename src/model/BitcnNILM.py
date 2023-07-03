from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.base_module.base_lightning import LightningBaseModule
from src.config_options.model_configs import ModelConfig_BitcnNILM
from src.config_options.option_def import MyProgramArgs
from src.model.multilabel_head import HeadModule

dilations = [1, 2, 4, 8, 16, 32, 64, 128]


class Residual_Block(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, dilation) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_chan,
                out_chan,
                kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.BatchNorm1d(out_chan),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.3),
        )
        # self.non_causal_dilated= nn.Conv1d(in_chan, out_chan, kernel_size, dilation=dilation, padding='same')
        # self.bn = nn.BatchNorm1d(out_chan)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.3)
        self.block2 = nn.Sequential(
            nn.Conv1d(
                in_chan,
                out_chan,
                kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.BatchNorm1d(out_chan),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.3),
        )
        self.resual_connection = nn.Conv1d(in_chan, out_chan, 1, padding="same")
        self.relu = nn.ReLU(inplace=True)

    # def init(self):
    #     self.block1[0].

    def forward(self, x):
        pre_x = self.resual_connection(x)
        x = self.block1(x)
        x = self.block2(x)
        res_x = x + pre_x
        res_x = self.relu(res_x)
        return res_x, x


class BitcnNILM(LightningBaseModule):
    def __init__(self, args: MyProgramArgs):
        super().__init__(args)
        self.args = args
        config: ModelConfig_BitcnNILM = self.args.modelConfig
        self.conv = nn.Conv1d(config.in_chan, 128, 1, padding="same")
        self.residual_block_list = nn.ModuleList()
        for d in dilations:
            self.residual_block_list.append(Residual_Block(128, 128, 3, dilation=d))
        self.f = nn.Flatten()
        
        if args.modelBaseConfig.label_mode == "multilabel":
            if config.head_type != '':
                self.linear = HeadModule[config.head_type](128, config.nclass, 0.5)
            else:
                raise ValueError(f'invalid head type: {config.head_type}')
        elif args.modelBaseConfig.label_mode == "multiclass":
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=args.modelBaseConfig.label_smoothing,
            )
            self.linear = nn.Linear(128, config.nclass)

    def loss(self, predictions, batch):
        pred = predictions["pred"]
        target = batch["target"]
        # return F.binary_cross_entropy_with_logits(output, target)
        return self.loss_fn(pred, target)

    def forward(self, batch):
        x = batch["input"]
        x = rearrange(x, "b t f -> b f t")

        x = self.conv(x)
        skip_connectiosn = []
        for i in range(len(dilations)):
            x, skip_out = self.residual_block_list[i](x)
            skip_connectiosn.append(skip_out)
        s = None
        for i in range(len(skip_connectiosn)):
            if s is None:
                s = skip_connectiosn[i]
            else:
                s += skip_connectiosn[i]

        s = s[:, :, -1]
        s = self.f(s)
        # if self.args.modelBaseConfig.label_mode == "multilabel":
            # x = self.linear(s)
            
        predictions = {}
        if self.args.modelBaseConfig.label_mode == "multilabel":
            batch['feature'] = s
            predictions = self.linear(batch)
            # predictions["output"] = torch.round(F.sigmoid(x))
            # predictions["pred"] = torch.round(F.sigmoid(x))
        else:
            predictions["output"] = torch.max(x, dim=1)[1]
            predictions["pred"] = x        
            predictions["loss"] = self.loss(predictions, batch)
        return predictions


def test_BitcnNILM():
    pass

    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.replace_params(
        {"modelConfig": "BitcnNILM", "modelBaseConfig.label_mode": "multiclass",'modelConfig.in_chan': 2},
    )
    model = BitcnNILM(args)
    out = model(torch.randn(32, 1, 100))
    print(out.shape)

def test_BitcnNILM_multilabel():
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.replace_params(
        {"modelConfig": "BitcnNILM", "modelBaseConfig.label_mode": "multilabel",'modelConfig.in_chan': 2,
         "modelConfig.nclass": 4, "modelConfig.head_type": "SASL"},
    )
    model = BitcnNILM(args)
    model.eval()
    batch={'input': torch.randn(32, 600, 2),'target': torch.empty((32, 4), dtype=torch.long).random_(2),}
    out = model(batch)
    print(out['output'].shape)

def test_BitcnNILM_mask():
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.replace_params(
        {"modelConfig": "BitcnNILM", "modelBaseConfig.label_mode": "multilabel",'modelConfig.in_chan': 1,
         "modelConfig.nclass": 4, "modelConfig.head_type": "Mask"},
    )
    model = BitcnNILM(args)
    model.eval()
    batch={
        'input': torch.randn(2, 600,1, requires_grad=True),
        'target':  torch.tensor([[0,0,1,1],[1,1,0,0]],dtype=torch.long),
        'mask':  torch.tensor([[1,1,1,1],[1,1,1,0]], dtype=torch.long),
    }
    out = model(batch)
    print(out)
    print(out['output'].shape)
