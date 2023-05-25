from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import spectral_norm

from src.base_module.base_lightning import LightningBaseModule
from src.config_options.model_configs import ModelConfig_BasicV2
from src.config_options.option_def import MyProgramArgs
from src.utilts import get_padding_one_more_or_same

logger = logging.getLogger(__name__)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class FCN_Block2d(nn.Module):
    def __init__(
        self,
        inplane: int,
        outplane: int,
        kernel: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int],
        bias: bool,
        batchnorm: bool,
    ) -> None:
        super().__init__()
        layers = []
        layers += [nn.Conv2d(inplane, outplane, kernel, stride, padding, bias=bias)]
        if batchnorm:
            layers += [nn.BatchNorm2d(outplane)]
        layers += [nn.ReLU(inplace=True)]
        self.fcnblock = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fcnblock(x)
        return x


class FCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        chan_list: list[int],
        ker_list: list[int | tuple[int, int]],
        padding_list: list[int | tuple[int, int]],
        stride_list: list[int | tuple[int, int]],
    ) -> None:
        super().__init__()
        layers = []
        for i in range(len(chan_list)):
            if i == 0:
                inplane = in_channels
            else:
                inplane = chan_list[i - 1]

            layers += [
                FCN_Block2d(
                    inplane=inplane,
                    outplane=chan_list[i],
                    kernel=ker_list[i],
                    stride=stride_list[i],
                    padding=padding_list[i],
                    bias=True,
                    batchnorm=True,
                ),
            ]

        self.num_layers = len(layers)
        self.model = nn.Sequential(*layers)
        self.out_channel = chan_list[-1]

    def forward(self, x):
        x = self.model(x)
        return x


class BasicV2(LightningBaseModule):
    def __init__(self, args: MyProgramArgs) -> None:
        super().__init__(args)
        self.args = args
        config: ModelConfig_BasicV2 = self.args.modelConfig
        chan_list = [config.chan_1, config.chan_2, config.chan_3]
        ker_list = [config.ker_1, config.ker_2, config.ker_3]
        stride_list = [config.stride_1, config.stride_2, config.stride_3]
        padding_list = [get_padding_one_more_or_same(k) for k in ker_list]
        logger.debug(
            f"chan_list {chan_list}\n"
            f"ker_list {ker_list}\n"
            f"stride_list {stride_list}\n"
            f"padding_list {padding_list}",
        )
        self.fcn = FCN(
            config.in_channels,
            chan_list=chan_list,
            ker_list=ker_list,
            padding_list=padding_list,
            stride_list=stride_list,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(config.dropout),
            spectral_norm(nn.Linear(self.fcn.out_channel, config.nclass)),
        )
        if args.modelBaseConfig.label_mode == "multilabel":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif args.modelBaseConfig.label_mode == "multiclass":
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=args.modelBaseConfig.label_smoothing,
            )

    def loss(self, predictions, batch):
        pred = predictions["pred"]
        target = batch["target"]
        # return F.binary_cross_entropy_with_logits(output, target)
        return self.loss_fn(pred, target)

    def forward(self, batch):
        x = batch["input"]
        x = rearrange(x, "b (v t)  -> b 1 v t", v=60)
        x = self.fcn(x)
        x = self.pool(x)
        x = self.classifier(x)

        # if self.training:
        predictions = {}
        if self.args.modelBaseConfig.label_mode == "multilabel":
            predictions["output"] = torch.round(F.sigmoid(x))
            predictions["pred"] = torch.round(F.sigmoid(x))
        else:
            predictions["output"] = torch.max(x, dim=1)[1]
            predictions["pred"] = x
        predictions["loss"] = self.loss(predictions, batch)
        return predictions


def test_basicV2():
    from deepspeed.profiling.flops_profiler import get_model_profile

    from src.config_options import OptionManager
    from src.config_options.model_configs import ModelConfig_BasicV2

    opt = OptionManager()
    args = opt.replace_params({"modelConfig": "BasicV2"})
    args.modelConfig = ModelConfig_BasicV2(5, 5, 5, 16, 8, 8, 1, 1, 1, 0.5, 1, 2)
    model = BasicV2(args)
    flops, macs, params = get_model_profile(
        model,
        (512, 24 * 7, 1),
        as_string=False,
        print_profile=False,
    )  # flops: 3756395, macs: 1864650, params: 4537
    print(f"flops: {flops}, macs: {macs}, params: {params}")
