# -*- coding: utf-8 -*-
"""
Created on Sat April 12 2020
@author: Hao Lu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['L1Loss', 'L2Loss', 'MLL1Loss', 'MaskedMLL1Loss']


class L1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, outputs, targets):
        loss = F.l1_loss(outputs, targets, reduction=self.reduction)
        return loss


class L2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L2Loss, self).__init__()
        self.reduction = reduction

    def forward(self, outputs, targets):
        loss = F.mse_loss(outputs, targets, reduction=self.reduction)
        return loss


# multi-level L1 loss for tasselnetv3
class MLL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MLL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, outputs, targets):
        loss = 0.5 * F.l1_loss(outputs[0], targets[0], reduction=self.reduction) + \
                0.5 * F.l1_loss(outputs[1], targets[1], reduction=self.reduction)
        return loss


class MaskedMLL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MaskedMLL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, outputs, targets, masks):
        bs, _, h0, w0 = outputs[0].size()
        _, _, h1, w1 = outputs[1].size()
        
        # hard mask
        masks0 = masks[0]
        masks0[masks0 > 0] = 1
        masks0 = 1 - masks0
        masks1 = masks[1]
        masks1[masks1 > 0] = 1
        masks1 = 1 - masks1

        diff0 = (outputs[0] - targets[0]) * masks0
        loss0 = torch.abs(diff0)
        loss0 = loss0.sum(dim=2).sum(dim=2) / masks0.sum(dim=2).sum(dim=2)
        loss0 = loss0.sum() / bs

        diff1 = (outputs[1] - targets[1]) * masks1
        loss1 = torch.abs(diff1)
        loss1 = loss1.sum(dim=2).sum(dim=2) / masks1.sum(dim=2).sum(dim=2)
        loss1 = loss1.sum() / bs

        loss = loss0 + loss1
        return loss


class MTLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MTLoss, self).__init__()
        self.reduction = reduction
        # self.nll_loss = nn.NLLLoss(reduction=reduction)
        self.bce_loss = nn.BCELoss(reduction=reduction)
    
    def forward(self, outputs, targets, masks):
        
        masks0 = targets[0] > 0
        masks1 = targets[1] > 0
        masks0, masks1 = masks0.float(), masks1.float()

        diff0 = (outputs[0] - targets[0]) * masks0
        loss0 = torch.abs(diff0)
        loss0 = loss0.sum(dim=2).sum(dim=2)
        masks0 = masks0.sum(dim=2).sum(dim=2)
        loss0 = loss0[masks0>0] / masks0[masks0>0]
        loss0 = loss0.sum() / len(loss0)

        diff1 = (outputs[1] - targets[1]) * masks1
        loss1 = torch.abs(diff1)
        loss1 = loss1.sum(dim=2).sum(dim=2)
        masks1 = masks1.sum(dim=2).sum(dim=2)
        loss1 = loss1[masks1>0] / masks1[masks1>0]
        loss1 = loss1.sum() / len(loss1)

        loss_count = loss0 + loss1
                
        masks = F.interpolate(masks, size=outputs[2].size()[2:])
        # masks = masks.type(torch.cuda.LongTensor)
        # loss_seg = self.nll_loss(F.log_softmax(outputs[2], dim=1), masks[:, 0, :, :])
        loss_seg = self.bce_loss(outputs[2], masks)
        return loss_count + loss_seg

class CrossEntropyLoss(nn.Module):
    def __init__(self,reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(reduction=reduction)

    def forward(self, inputs, targets):
        loss_t = self.nll_loss(F.log_softmax(inputs, dim=1), targets[:, 0, :, :])
        return loss_t
    