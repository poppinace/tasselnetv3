import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .imagenet_models.model_all import tf_mixnet_l

from inplace_abn import ABN
BatchNormAct2d = ABN

BatchNorm2d = nn.BatchNorm2d


class Seg2Count(nn.Module):
    def __init__(self, pretrain):
        super(Seg2Count, self).__init__()

        ori_mixnet_l = tf_mixnet_l(pretrained=pretrain)

        self.conv_stem = ori_mixnet_l.conv_stem
        self.bn1 = ori_mixnet_l.bn1
        self.act_fn = ori_mixnet_l.act_fn
        self.blocks_1_8 = ori_mixnet_l.blocks[:3]
        self.blocks_1_16 = ori_mixnet_l.blocks[3:5]
        self.blocks_1_32 = ori_mixnet_l.blocks[5:6]

        if not pretrain:
            self._weight_init()

    def forward(self, x):
        x_out = []
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        x = self.blocks_1_8(x)
        x_out.append(x)
        x = self.blocks_1_16(x)
        x_out.append(x)
        x = self.blocks_1_32(x)
        x_out.append(x)
        return x_out

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


def seg2count(pretrain=True, **kwargs):
    model = Seg2Count(pretrain=pretrain)
    return model