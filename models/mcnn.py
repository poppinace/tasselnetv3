# -*- coding: utf-8 -*-
"""
Created on Sat April 12 2020
@author: Liang Liu and Hao Lu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mcnn', 'mcnn_bn']


class MCNN(nn.Module):
    '''
    encoder implementation of multi-column CNN
    '''
    def __init__(self):
        super(MCNN, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(16, 32, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 7, padding=3),
            nn.ReLU(inplace=True)
            )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(20, 40, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(40, 20, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, 5, padding=2),
            nn.ReLU(inplace=True)
            )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(48, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.ReLU(inplace=True)
            )

        self.weight_init()
    
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat((x1,x2,x3),1)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MCNN_BN(nn.Module):
    '''
    encoder implementation of multi-column CNN
    '''
    def __init__(self):
        super(MCNN_BN, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 9, padding=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(16, 32, 7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(32, 16, 7, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 7, padding=3, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
            )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 7, padding=3, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(20, 40, 5, padding=2, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(40, 20, 5, padding=2, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, 5, padding=2, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True)
            )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(24, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(48, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
            )

        self.weight_init()
    
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat((x1,x2,x3),1)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

def mcnn(**kwargs):
    """Construct a TasselNetv2 model."""
    model = MCNN()
    return model


def mcnn_bn(**kwargs):
    """Construct a TasselNetv2 model."""
    model = MCNN_BN()
    return model