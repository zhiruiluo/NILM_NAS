# Non-intrusive load decomposition based on CNN–LSTM hybrid deep learning model
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base_module.base_lightning import LightningBaseModule
from src.config_options.option_def import MyProgramArgs
from src.config_options.model_configs import ModelConfig_CNN_LSTM
from einops import rearrange
from .multilabel_head import MultilabelLinearFocal

def lstm(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0,bidirectional=False):
    return nn.LSTM(input_size,hidden_size,num_layers,bias,batch_first,dropout,bidirectional)

class CNN_subnet(nn.Module):
    def __init__(self, in_chan: int, out_features=32) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_chan, out_channels=64, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Flatten(1),
            nn.LazyLinear(out_features),
        )
        
    def forward(self, x):
        return self.cnn(x)

class LSTM_subnet(nn.Module):
    def __init__(self, input_size, hidden_size, out_features=32) -> None:
        super().__init__()
        self.bilstm = lstm(input_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.leaky_relu = nn.LeakyReLU()
        self.lstm = lstm(hidden_size, hidden_size, batch_first=True)
        self.lazy_linear = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features)
        )
        
    def forward(self, x):
        output, (hn, cn) = self.bilstm(x)
        output = rearrange(output, "b t (d h) -> b t h d", d=2)
        output = torch.sum(output, dim=3)
        _, (hn, cn) = self.lstm(output)
        print(hn[-1].shape)
        return self.lazy_linear(hn[-1])


class CNN_LSTM(LightningBaseModule):
    def __init__(self, args: MyProgramArgs):
        super().__init__(args)
        config: ModelConfig_CNN_LSTM = args.modelConfig
        out_features = config.out_features
        self.cnn_sub = CNN_subnet(config.in_chan, out_features)
        self.lstm_sub = LSTM_subnet(input_size=config.in_chan, hidden_size=64, out_features=out_features)
        self.flatten = nn.Flatten()
        self.fc = MultilabelLinearFocal(out_features*2, config.nclass)
            
    def forward(self, batch):
        x = batch['input']
        target = batch['target']
        cnn_out = self.cnn_sub(rearrange(x, "b t f -> b f t"))
        lstm_hidden = self.lstm_sub(x)
        print(cnn_out.shape, lstm_hidden.shape)
        out = torch.concatenate((cnn_out,lstm_hidden), dim=1)
        print(out.shape)
        batch['feature'] = self.flatten(out)
        predictions = self.fc(batch)
        
        return predictions
    
def test_cnn_lstm():
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.replace_params(
        {"modelConfig": "CNN_LSTM", "modelBaseConfig.label_mode": "multilabel",'modelConfig.in_chan': 1,
         "modelConfig.nclass": 4},
    )
    model = CNN_LSTM(args)
    model.eval()
    batch={'input': torch.randn(32, 300, 1),'target': torch.empty((32, 4), dtype=torch.long).random_(2),}
    out = model(batch)
    print(out['output'].shape)