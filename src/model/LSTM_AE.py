# Multi-label LSTM autoencoder for non-intrusive appliance load monitoring
import torch.nn as nn
import torch
import torch.nn.functional as F
from src.base_module.base_lightning import LightningBaseModule
from src.config_options.option_def import MyProgramArgs
from src.config_options.model_configs import ModelConfig_LSTM_AE
from einops import rearrange
from .multilabel_head import *

import logging
logger = logging.getLogger(__name__)

def lstm(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0,bidirectional=False):
    return nn.LSTM(input_size,hidden_size,num_layers,bias,batch_first,dropout,bidirectional)

class Encoder(nn.Module):

    def __init__(self, input_size=4096, hidden_size=1024, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=0.1, bidirectional=False)
        self.relu = nn.ReLU()
        
        
    def forward(self, x):
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, (hidden, cell) = self.lstm(x)
        return (hidden, cell)
    
class Decoder(nn.Module):

    def __init__(self, input_size=4096, hidden_size=1024, output_size=1, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.1, bidirectional=False)
        
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x, hidden):
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        output, (hidden, cell) = self.lstm(x, hidden)  
        prediction = self.fc(output)
        
        return prediction, (hidden, cell)
    
class LSTM_AE(LightningBaseModule):
    def __init__(self, args: MyProgramArgs):
        super().__init__(args)
        config: ModelConfig_LSTM_AE = args.modelConfig
        # self.encoder = lstm(input_size=config.in_chan, hidden_size=128, batch_first=True, num_layers=2, dropout=0.1)
        # self.decoder = lstm(input_size=128, hidden_size=128, batch_first=True, num_layers=2, dropout=0.1)
        input_size = config.in_chan
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        output_size = input_size
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.reconstruct_decoder = Decoder(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(in_features=hidden_size, out_features=config.nclass)
        # )
        if args.modelBaseConfig.label_mode == "multilabel":
            if config.head_type == 'CE':
                self.linear = MultilabelLinear(hidden_size, config.nclass)
            elif config.head_type == 'Focal':
                self.linear = MultilabelLinearFocal(hidden_size, config.nclass)
            elif config.head_type == 'SBCE':
                self.linear = SharedBCELinear(hidden_size, config.nclass)
            elif config.head_type == 'Mask':
                self.linear = MultilabelLinearMask(hidden_size, config.nclass)
            elif config.head_type == 'ASL':
                self.linear = MultilabelLinearASL(hidden_size, config.nclass)
            elif config.head_type == 'MaskFocal':
                self.lienar = MultilabelLinearMaskFocal(hidden_size, config.nclass)
            else:
                raise ValueError(f'invalid head type: {config.head_type}')
        
    def reconstruction_loss(self, input, target):
        return F.mse_loss(input, target)
        
    def multilabel_loss(self, pred: torch.TensorType, target: torch.Tensor):
        return F.mse_loss(pred, target)
    
    def forward(self, batch):
        x = batch['input'] # b t f
        batch_size, sequence_length, f = x.size()
        target = batch['target']
        e_hidden = self.encoder(x)
        temp_input = torch.zeros((batch_size, 1, f), dtype=torch.float32).to(x.device)
        hidden = e_hidden
        reconstruct_output = []
        for t in range(sequence_length):
            temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
            reconstruct_output.append(temp_input)
            
        reconstruct_output = torch.cat(reconstruct_output, dim=1)
        
        reconstruct_output = torch.flip(reconstruct_output, dims=(1,))
        recon_loss = self.reconstruction_loss(reconstruct_output, x)
        batch['feature'] = e_hidden[0][-1:].squeeze(0)
        # pred = F.sigmoid(self.fc(e_hidden[0][-1].squeeze(0)))
        predictions = self.linear(batch)
        # cls_loss = preds['loss']
        predictions['loss'] = predictions['loss'] + 0.1*recon_loss
        
        # cls_loss = self.multilabel_loss(pred, target)
        
        # predictions = {}
        # predictions['output'] = torch.round(F.sigmoid(pred))
        # predictions['loss'] = cls_loss + 0.1 * recon_loss
        # predictions['pred'] = pred
        return predictions
    

def test_lstm_ae():
    from src.config_options import OptionManager

    opt = OptionManager()
    args = opt.replace_params(
        {"modelConfig": "LSTM_AE", "modelBaseConfig.label_mode": "multilabel",'modelConfig.in_chan': 1,
         "modelConfig.nclass": 4},
    )
    model = LSTM_AE(args)
    model.eval()
    batch={'input': torch.randn(32, 600, 1),'target': torch.empty((32, 4), dtype=torch.long).random_(2),}
    out = model(batch)
    print(out['output'].shape)