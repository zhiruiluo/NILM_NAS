import torch.nn as nn
import torch
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class MultilabelLinear(nn.Module):
    def __init__(self, in_dim, nclass, dropout=0.5) -> None:
        super().__init__()
        self.nclass = nclass
        heads = nn.ModuleList()
        for i in range(nclass):
            heads.append(nn.Sequential(
                nn.Dropout(dropout),
                spectral_norm(nn.Linear(in_dim, 2)),
            ))
        
        self.heads = heads
        self.loss_fn = nn.CrossEntropyLoss()
    
    def loss(self, predictions, batch):
        pred = predictions["pred"]
        target = batch["target"].type(torch.long)
        target = F.one_hot(target, num_classes=2).type(torch.float32)
        return self.loss_fn(pred, target)
    
    def forward(self, batch):
        x = batch['feature']
        
        out = []
        for head in self.heads:
            out.append(head(x))
            
        out = torch.stack(out,dim=1)
        preds = {}
        preds['pred'] = out
        preds['output'] = torch.max(out, dim=2)[1]
        preds['loss'] = self.loss(preds, batch)
        return preds
    
def test_MultilabelLinear():
    m = MultilabelLinear(64, 4)
    batch = {
        "feature": torch.randn(2, 64),
        "target": torch.empty((2, 4), dtype=torch.long).random_(2),
    }
    out = m(batch)
    print(out)