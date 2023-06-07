# Multi-Label Learning for Appliance Recognition in NILM Using Fryze-Current Decomposition and Convolutional Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base_module.base_lightning import LightningBaseModule
from src.config_options.option_def import MyProgramArgs
from src.config_options.model_configs import ModelConfig_CNN_LSTM

class CNN_Anthony(LightningBaseModule):
    def __init__(self, args: MyProgramArgs):
        super().__init__(args)
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, 5, 2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, 2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            
            nn.Dropout(0.25),
            nn.Linear(128, 512),
            nn.BatchNorm1d(),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(),
            nn.ReLU(),
            
            nn.Dropout(0.25),
            nn.Linear(1024, 4),
            nn.Softmax()
        )