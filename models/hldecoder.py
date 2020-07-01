# -*- coding: utf-8 -*-
"""
Created on Sun May 3 2020
@author: Hao Lu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .carafe.carafe import CARAFEPack

BatchNorm2d = nn.BatchNorm2d

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, padding=0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_3x3_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, padding=1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def dilated_conv_3x3_bn(inp, oup, padding, dilation):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, padding=padding, dilation=dilation, bias=False),
        BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class _ASPPModule(nn.Module):
    def __init__(self, inp, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        if kernel_size == 1:
            self.atrous_conv = nn.Sequential(
                nn.Conv2d(inp, planes, kernel_size=1, stride=1, padding=padding, dilation=dilation, bias=False),
                BatchNorm2d(planes),
                nn.ReLU6(inplace=True)
            )
        elif kernel_size == 3:
            # we use depth-wise separable convolution to save the number of parameters
            self.atrous_conv = dilated_conv_3x3_bn(inp, planes, padding=padding, dilation=dilation)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inp, oup, output_stride=32):
        super(ASPP, self).__init__()

        if output_stride == 32:
            dilations = [1, 2, 4, 8]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inp, int(64), 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inp, int(64), 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inp, int(64), 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inp, int(64), 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inp, int(64), 1, stride=1, padding=0, bias=False),
            BatchNorm2d(int(64)),
            nn.ReLU6(inplace=True)
        )

        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(int(64)*5, oup, 1, stride=1, padding=0, bias=False),
            BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
        
        self.dropout = nn.Dropout(0.5)

        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='nearest')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.bottleneck_conv(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class PPM(nn.Module):
    def __init__(self, inp, oup, base_size):
        super(PPM, self).__init__()
        # PPM Module
        pool_scales=(1, 2, 3, 6)
        self.ppm_pooling = []
        self.ppm_conv = []
        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(conv_1x1_bn(inp, 64))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv_3x3_bn(inp + len(pool_scales)*64, oup)
    
    def forward(self, x):
        ppm_out = [x]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(x),
                (x.size()[2], x.size()[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        ppm_out = self.ppm_last_conv(ppm_out)
        return ppm_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Seg2CountDecoder(nn.Module):
    def __init__(self, decoder_dim, base_size):
        super(Seg2CountDecoder, self).__init__()

        self.conv_1_8 = conv_1x1_bn(56, decoder_dim)
        self.conv_1_16 = conv_1x1_bn(160, decoder_dim)

        # # counting decoder
        self.conv_1_32 = conv_1x1_bn(264, decoder_dim)
        self.carafe_count_32_to_16 = CARAFEPack(decoder_dim, compressed_channels=int(decoder_dim/2))
        self.carafe_count_16_to_8 = CARAFEPack(decoder_dim, compressed_channels=int(decoder_dim/2))
        self.conv_count = conv_3x3_bn(decoder_dim, decoder_dim)

        # segmentation decoder
        self.aspp = ASPP(264, decoder_dim)
        self.carafe_seg_32_to_16 = CARAFEPack(decoder_dim, compressed_channels=int(decoder_dim/2))
        self.carafe_seg_16_to_8 = CARAFEPack(decoder_dim, compressed_channels=int(decoder_dim/2))
        # self.ppm = PPM(264, decoder_dim, base_size)
        self.conv_seg = conv_3x3_bn(decoder_dim, decoder_dim)

        self._weight_init()

    def forward(self, x):
        x32, x16, x8 = x[-1], x[-2], x[-3]

        # x_count = x8
        
        x16 = self.conv_1_16(x16)
        x8 = self.conv_1_8(x8)

        # ASPP + CARAFE
        # counting branch
        x_count = self.conv_1_32(x32)
        x_count = self.carafe_count_32_to_16(x_count) + x16
        x_count = self.carafe_count_16_to_8(x_count) + x8
        x_count = self.conv_count(x_count)

        # segmentation branch
        x_seg = self.aspp(x32)
        x_seg = self.carafe_seg_32_to_16(x_seg) + x16
        x_seg = self.carafe_seg_16_to_8(x_seg) + x8
        x_seg = self.conv_seg(x_seg)

        # # PPM + Bilinear
        # # counting branch
        # x_count = self.conv_1_32(x32)
        # x_count = nn.functional.interpolate(
        #     x_count,
        #     (x16.size()[2], x16.size()[3]),
        #     mode='bilinear', align_corners=False
        #     ) 
        # x_count += x16
        # x_count = nn.functional.interpolate(
        #     x_count,
        #     (x8.size()[2], x8.size()[3]),
        #     mode='bilinear', align_corners=False
        #     )
        # x_count += x8
        # x_count = self.conv_count(x_count)

        # # segmentation branch
        # x_seg = self.ppm(x32)
        # x_seg = nn.functional.interpolate(
        #     x_seg,
        #     (x16.size()[2], x16.size()[3]),
        #     mode='bilinear', align_corners=False
        #     ) 
        # x_seg += x16
        # x_seg = nn.functional.interpolate(
        #     x_seg,
        #     (x8.size()[2], x8.size()[3]),
        #     mode='bilinear', align_corners=False
        #     )
        # x_seg += x8
        # x_seg = self.conv_seg(x_seg)

        return x_count, x_seg

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


def seg2count_decoder(decoder_dim, base_size, **kwargs):
    decoder = Seg2CountDecoder(decoder_dim, base_size, **kwargs)
    return decoder
