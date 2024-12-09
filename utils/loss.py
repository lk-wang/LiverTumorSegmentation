

#
import torch
import torch.nn as nn
import torch.nn.functional as F

# dice loss  对batch维度挨个计算Dice并求和。
# view(bs, -1)转为二维，后面计算score的.sum(1)是对第二维计算的，所以bs维是一直保持的，所以是实现第一种方式。
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        #probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score



"""
二值交叉熵损失函数
"""

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
    def forward(self, pred, target):
        return self.bce_loss(pred, target)


"""
交叉熵loss+dice loss
L=L_dice + 0.5L_cross
"""
class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, targets):
        num = targets.size(0)
        smooth = 1
        #probs = F.sigmoid(logits)
        m1 = pred.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)

        l_dice = 1 - score.sum() / num
        l_cross=self.bce_loss(pred, targets)
        l_all=l_dice + 0.5*l_cross

        return l_all


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.3, beta=0.7, gamma=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        tversky_score = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky_score = (1 - tversky_score)**gamma
                       
        return FocalTversky_score




















