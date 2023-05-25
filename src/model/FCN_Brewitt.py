import torch
import torch.nn as nn
from einops import rearrange

from src.config_options.model_configs import ModelConfig_BitcnNILM
from src.config_options.option_def import MyProgramArgs
from src.base_module.base_lightning import LightningBaseModule

dilations = [2,4,8,16,32,64,128,256]

def conv1d_activation(in_c, out_c, kernel, padding, activation, dilation):
    layers = []
    layers.append(nn.Conv1d(in_c, out_c, kernel, padding=padding, dilation=dilation))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)
    

class FCN_Brewitt(LightningBaseModule):
    def __init__(self, args: MyProgramArgs):
        super().__init__(args)
        # self.conv1 = nn.Conv1d(1, 128, 9, padding='same')
        self.conv1 = conv1d_activation(1, 128, 9, 'same', 'relu', dilation=1)
        self.stack_dilated_cnn = []
        for d in dilations:
            self.stack_dilated_cnn.append(conv1d_activation(128, 128, 3, 'same', 'relu', dilation=d))
        
        self.stack_dilated_cnn = nn.Sequential(*self.stack_dilated_cnn)
        self.conv2 = conv1d_activation(128, 256, 1, 'same', 'relu', dilation=1)
        self.conv3 = conv1d_activation(256, 1, 1, 'same', None, dilation=1)
        
        
        
        
        
    def forward(self, batch):
        return