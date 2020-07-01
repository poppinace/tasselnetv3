# -*- coding: utf-8 -*-
"""
Created on Sun April 17 2020
@author: Hao Lu

CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes
Yuhong Li, Xiaofan Zhang, and Deming Chen
CVPR 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

__all__ = ['csrnet', 'csrnet_bn']


class CSRNet(nn.Module):
    def __init__(self, use_bn=False, fix_bn=False):
        super(CSRNet, self).__init__()

        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]

        # frontend layer
        # self.features must have the name 'features' to enable to load pretrained weights
        self.features = self._make_layers(self.frontend_feat, use_bn=use_bn)

        if fix_bn:
            self._fix_bn()

        # backend layer
        self.backend = self._make_layers(self.backend_feat, in_channels=512, use_bn=use_bn, dilation=True)

        self._weight_init()


    def forward(self, x):
        x = self.features(x)
        x = self.backend(x)
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
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation = d_rate)
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


def csrnet(pretrain=True, **kwargs):
    model = CSRNet(use_bn=False)
    if pretrain:
        pretrained_model = models.vgg16(pretrained=True)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # load the new state dict
        model.load_state_dict(model_dict)

    return model


def csrnet_bn(pretrain=True, fix_bn=True, **kwargs):
    model = CSRNet(use_bn=True, fix_bn=fix_bn)
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