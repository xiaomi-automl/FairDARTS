import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ConvSeparateLoss', 'TriSeparateLoss']
class ConvSeparateLoss(nn.modules.loss._Loss):
    """Separate the weight value between each operations using L2"""
    def __init__(self, weight=0.1, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ConvSeparateLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, input1, target1, input2):
        loss1 = F.cross_entropy(input1, target1)
        loss2 = -F.mse_loss(input2, torch.tensor(0.5, requires_grad=False).cuda())
        return loss1 + self.weight*loss2, loss1.item(), loss2.item()


class TriSeparateLoss(nn.modules.loss._Loss):
    """Separate the weight value between each operations using L1"""
    def __init__(self, weight=0.1, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(TriSeparateLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, input1, target1, input2):
        loss1 = F.cross_entropy(input1, target1)
        loss2 = -F.l1_loss(input2, torch.tensor(0.5, requires_grad=False).cuda())
        return loss1 + self.weight*loss2, loss1.item(), loss2.item()