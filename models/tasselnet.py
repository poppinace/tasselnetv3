# -*- coding: utf-8 -*-
"""
Created on Sat April 12 2020
@author: Hao Lu

TasselNet: counting maize tassels in the wild via local counts regression network
Hao Lu, Zhiguo Cao, Yang Xiao, Bohan Zhuang, and Chunhua Shen
PLME 2017

TasselNetv2: in-field counting of wheat spikes with context-augmented local regression networks
Haipeng Xiong, Zhiguo Cao, Hao Lu, Simon Madec, Liang Liu, and Chunhua Shen
PLME 2019

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .imagenet_models.model_all import tf_mixnet_l

from inplace_abn import ABN
BatchNormAct2d = ABN

__all__ = ['tasselnetv2', 'tasselnetv2plus', 'tasselnetv3']

class TasselNetv2(nn.Module):
    def __init__(self):
        super(TasselNetv2, self).__init__()
        self.encoder = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 2), stride=2),
                    nn.Conv2d(16, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 2), stride=2),
                    nn.Conv2d(32, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 2), stride=2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
        )
    
        self.weight_init()
    
    def forward(self, x):
        x = self.encoder(x)
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


class TasselNetv2Plus(nn.Module):
    def __init__(self):
        super(TasselNetv2Plus, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(16, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
        )

        self.weight_init()
    
    def forward(self, x):
        x = self.encoder(x)
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
            elif isinstance(m, BatchNormAct2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class TasselNetv3(nn.Module):
    def __init__(self):
        super(TasselNetv3, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=False),
                BatchNormAct2d(16, activation="relu"),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(16, 32, 3, padding=1, bias=False),
                BatchNormAct2d(32, activation="relu"),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                BatchNormAct2d(64, activation="relu"),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                BatchNormAct2d(128, activation="relu"),
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                BatchNormAct2d(128, activation="relu")
        )

        self.weight_init()
    
    def forward(self, x):
        x = self.encoder(x)
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
            elif isinstance(m, BatchNormAct2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class TasselNetv3_VGG16(nn.Module):
    def __init__(self, use_bn=False, fix_bn=False):
        super(TasselNetv3_VGG16, self).__init__()

        encoder = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.features = self._make_layers(encoder, use_bn=use_bn)

        if fix_bn:
            self._fix_bn()

        self._weight_init()

    def forward(self, x):
        x = self.features(x)
        return x

    def _make_layers(self, cfg, in_channels=3, use_bn=False, dilation=False):
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if use_bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _weight_init(self):
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
    
    def _fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class TasselNetv3_MixNet_L(nn.Module):
    def __init__(self, pretrain):
        super(TasselNetv3_MixNet_L, self).__init__()

        ori_mixnet_l = tf_mixnet_l(pretrained=pretrain)

        self.conv_stem = ori_mixnet_l.conv_stem
        self.bn1 = ori_mixnet_l.bn1
        self.act_fn = ori_mixnet_l.act_fn
        self.blocks = ori_mixnet_l.blocks[:3]
        # self.conv_head = ori_mixnet_l.conv_head
        # self.bn2 = ori_mixnet_l.bn2

        if not pretrain:
            self._weight_init()

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        x = self.blocks(x)
        # x = self.conv_head(x)
        # x = self.bn2(x)
        # x = self.act_fn(x, inplace=True)
        return x

    def _weight_init(self):
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


def tasselnetv2(**kwargs):
    """Construct a TasselNetv2 model."""
    model = TasselNetv2()
    return model


def tasselnetv2plus(**kwargs):
    """Construct a TasselNetv2+ model."""
    model = TasselNetv2Plus()
    return model


def tasselnetv3(**kwargs):
    """Construct a TasselNetv3 model."""
    model = TasselNetv3()
    return model


def tasselnetv3_vgg16(pretrain=True, fix_bn=True, **kwargs):
    """Construct a TasselNetv3 model."""
    model = TasselNetv3_VGG16(use_bn=True, fix_bn=fix_bn)
    if pretrain:
        pretrained_model = models.vgg16_bn(pretrained=True)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # load the new state dict
        model.load_state_dict(model_dict)

    return model


def tasselnetv3_mixnet_l(pretrain=True, **kwargs):
    """Construct a TasselNetv3 model."""
    model = TasselNetv3_MixNet_L(pretrain=pretrain)
    return model
