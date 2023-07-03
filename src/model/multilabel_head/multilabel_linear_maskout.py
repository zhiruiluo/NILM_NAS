from typing import Any
import torch.nn as nn
import torch
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from ..loss.focalloss import MaskedFocal
import logging
logger = logging.getLogger(__name__)

class MaskedCrossEntropy(nn.Module):
    def __init__(self) -> Any:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, input, target, mask):
        batch, nclass, d = input.size()
        loss = self.loss_fn(input.reshape(batch*nclass, d), target.reshape(batch*nclass, d))
        return torch.mul(loss.reshape(batch, nclass), mask).mean()


class MultilabelLinearMask(nn.Module):
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
        self.loss_fn = MaskedCrossEntropy()
    
    def loss(self, predictions, batch):
        pred = predictions["pred"]
        target = batch["target"].type(torch.long)
        target = F.one_hot(target, num_classes=2).type(torch.float32)
        return self.loss_fn(pred, target, mask=batch['mask'])
    
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
    
class MultilabelLinearMaskFocal(nn.Module):
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
        self.loss_fn = MaskedFocal(2, 0.25)
    
    def loss(self, predictions, batch):
        pred = predictions["pred"]
        target = batch["target"].type(torch.long)
        # target = F.one_hot(target, num_classes=2).type(torch.float32)
        return self.loss_fn(pred, target, mask=batch['mask'])
    
    def forward(self, batch):
        x = batch['feature']
        
        out = []
        for head in self.heads:
            out.append(head(x))
            
        out = torch.stack(out,dim=2)
        preds = {}
        preds['pred'] = out
        preds['output'] = torch.max(out, dim=1)[1]
        preds['loss'] = self.loss(preds, batch)
        return preds
    
def test_MultilabelLinearMask():
    m = MultilabelLinearMaskFocal(64, 4)
    batch = {
        "feature": torch.randn(3, 64),
        # "target": torch.empty((2, 4), dtype=torch.long).random_(2),
        "target": torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,0,0]],dtype=torch.long),
        "mask": torch.tensor([[1,1,1,1],[1,1,1,0],[0,0,1,1]], dtype=torch.long)
    }
    out = m(batch)
    print(out)
    
def test_target_to_mask():
    target = torch.tensor([[0,0,1,1],[1,1,0,-1]],dtype=torch.long)
    mask = target != -1
    print(mask)
    

def test_loss_fn():
    # Example usage
    criterion = MaskedCrossEntropy()
    input = torch.randn(2, 3, requires_grad=True)
    target = torch.tensor([
            [1., 2., 1.],
            [1., 1., 0.],
        ])
    mask = torch.tensor([[1], [0]])

    output = criterion(input, target, mask)
    print(output)
    output.backward()
    print(input)