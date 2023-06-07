import torch.nn as nn
import torch
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class SharedBCELinear(nn.Module):
    def __init__(self, in_dim, nclass, dropout=0.5) -> None:
        super().__init__()
        self.nclass = nclass
        head = nn.Sequential(
                nn.Dropout(dropout),
                spectral_norm(nn.Linear(in_dim, nclass)),
            )
        
        self.head = head
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def loss(self, predictions, batch):
        pred = predictions["pred"]
        target = batch["target"].type(torch.float32)
        # target = F.one_hot(target).type(torch.float32)
        return self.loss_fn(pred, target)
    
    def forward(self, batch):
        x = batch['feature']
        out = self.head(x)
        print(out.shape)
        preds = {}
        preds['pred'] = out
        preds['output'] = torch.round(F.sigmoid(out))
        preds['loss'] = self.loss(preds, batch)
        return preds
    
def test_SharedBCELinear():
    m = SharedBCELinear(64, 4)
    batch = {
        "feature": torch.randn(2, 64),
        "target": torch.empty((2, 4), dtype=torch.long).random_(2),
    }
    out = m(batch)
    print(out)