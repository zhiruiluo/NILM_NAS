import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from enum import Enum

class ReductionEnum(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2
    
def get_reduction_enum(reduction: str) -> ReductionEnum:
    if reduction == 'mean':
        return ReductionEnum.MEAN
    elif reduction == 'sum':
        return ReductionEnum.SUM
    elif reduction == 'none':
        return ReductionEnum.NONE
    raise ValueError()

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = get_reduction_enum(reduction)

    def forward(self, input, target):
        batch = input.size(0)
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduction == ReductionEnum.MEAN:
            return loss.mean()
        elif self.reduction == ReductionEnum.SUM:
            return loss.sum()
        elif self.reduction == ReductionEnum.NONE:
            return loss.reshape(batch, -1)
        
# class MaskedFocal(nn.Module):
#     def __init__(self, gamma=0, alpha=None, reduction='mean'):
#         super().__init__()
#         self.reduction = get_reduction_enum(reduction)
#         self.loss_fn = FocalLoss(gamma,alpha,reduction='none')
        
#     def forward(self, input, target, mask):
#         loss = self.loss_fn(input, target)
#         masked_loss = torch.mul(loss, mask)
#         if self.reduction == ReductionEnum.MEAN:
#             return masked_loss.mean()
#         elif self.reduction == ReductionEnum.SUM:
#             return masked_loss.sum()
#         elif self.reduction == ReductionEnum.NONE:
#             return masked_loss
        
class MaskedFocal(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = get_reduction_enum(reduction)
        
    def forward(self, input, target, mask):
        batch = input.size(0)
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        masked_loss = torch.mul(loss.reshape(batch, -1), mask)
        if self.reduction == ReductionEnum.MEAN:
            return masked_loss.mean()
        elif self.reduction == ReductionEnum.SUM:
            return masked_loss.sum()
        elif self.reduction == ReductionEnum.NONE:
            return masked_loss