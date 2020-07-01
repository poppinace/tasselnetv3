# -*- coding: utf-8 -*-
"""
Created on Sun April 12 2020
@author: Hao Lu, Liang Liu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2

if __name__ != "__main__":
    from . import tasselnet, mcnn, bcnet, csrnet, seg2count
else:
    import tasselnet, mcnn, bcnet, csrnet, seg2count
from . import hldecoder

from inplace_abn import ABN
BatchNormAct2d = ABN


class BilinearUpsampling(nn.Module):
    def __init__(self,  scale_factor):
        super(BilinearUpsampling, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        

class Encoder(nn.Module):
    def __init__(self, encoder='tasselnetv2plus', pretrain=True, fix_bn=True):
        super(Encoder, self).__init__()
        if encoder == 'tasselnetv2':
            self.encoder = tasselnet.__dict__['tasselnetv2']()
        elif encoder == 'tasselnetv2plus':
            self.encoder = tasselnet.__dict__['tasselnetv2plus']()
        elif encoder == 'tasselnetv3':
            self.encoder = tasselnet.__dict__['tasselnetv3']()
        elif encoder == 'tasselnetv3_vgg16':
            self.encoder = tasselnet.__dict__['tasselnetv3_vgg16'](pretrain, fix_bn)
        elif encoder == 'tasselnetv3_mixnet_l':
            self.encoder = tasselnet.__dict__['tasselnetv3_mixnet_l'](pretrain)
        elif encoder == 'mcnn':
            self.encoder = mcnn.__dict__['mcnn']()
        elif encoder == 'mcnn_bn':
            self.encoder = mcnn.__dict__['mcnn_bn']()
        elif encoder == 'bcnet':
            self.encoder = bcnet.__dict__['bcnet'](pretrain)
        elif encoder == 'bcnet_bn':
            self.encoder = bcnet.__dict__['bcnet_bn'](pretrain, fix_bn)
        elif encoder == 'csrnet':
            self.encoder = csrnet.__dict__['csrnet'](pretrain)
        elif encoder == 'csrnet_bn':
            self.encoder = csrnet.__dict__['csrnet_bn'](pretrain, fix_bn)
        elif encoder == 'seg2count':
            self.encoder = seg2count.__dict__['seg2count'](pretrain)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder='no_decoder', decoder_dim=64, base_size=256):
        super(Decoder, self).__init__()
        self.decoder_name = decoder
        if decoder == 'no_decoder':
            pass
        elif decoder == 'seg2count_decoder':
            self.decoder = hldecoder.__dict__['seg2count_decoder'](decoder_dim, base_size)

    def forward(self, x):
        return x if self.decoder_name == 'no_decoder' else self.decoder(x)


# ------------------------------------------------------------------------
# Counter
# ------------------------------------------------------------------------
class Counter(nn.Module):
    def __init__(
        self, 
        counter='count_regressor', 
        block_size=64, 
        output_stride=8,
        counter_dim=128,
        step_log=0.1, 
        start_log=-2,
        num_class=60
        ):
        super(Counter, self).__init__()
        self.counter_type = counter
        if counter == 'count_regressor_fc':
            k = int(block_size / output_stride)
            avg_pool_stride = int(output_stride / output_stride)
            self.counter = nn.Sequential(
                nn.Conv2d(counter_dim, counter_dim, (k, k), bias=False),
                nn.BatchNorm2d(counter_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(counter_dim, counter_dim, 1, bias=False),
                nn.BatchNorm2d(counter_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(counter_dim, 1, 1)
            )
        elif counter == 'count_regressor':
            k = int(block_size / output_stride)
            avg_pool_stride = int(output_stride / output_stride)
            self.counter = nn.Sequential(
                nn.AvgPool2d((k, k), stride=avg_pool_stride),
                nn.Conv2d(counter_dim, counter_dim, 1, bias=False),
                nn.BatchNorm2d(counter_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(counter_dim, 1, 1)
            )
        elif counter == 'count_unfolding_regressor':
            self.k = int(block_size / output_stride)
            avg_pool_stride = int(output_stride / output_stride)
            self.counter = nn.Sequential(
                nn.AvgPool2d((self.k, self.k), stride=avg_pool_stride),
                nn.Conv2d(counter_dim, counter_dim, 1, bias=False),
                # BatchNormAct2d(counter_dim, activation="relu"),
                nn.BatchNorm2d(counter_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(counter_dim, 1, 1)
            )
        elif counter == 'count_unfolding_regressor_segmenter':
            self.k = int(block_size / output_stride)
            avg_pool_stride = int(output_stride / output_stride)
            self.counter = nn.Sequential(
                nn.AvgPool2d((self.k, self.k), stride=avg_pool_stride),
                nn.Conv2d(counter_dim, counter_dim, 1, bias=False),
                nn.BatchNorm2d(counter_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(counter_dim, 1, 1)
            )
            self.segmenter = nn.Sequential(
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid()
            )

        elif counter == 'density_map_regressor':
            self.counter = nn.Conv2d(counter_dim, 1, 1)
        elif counter == 'count_interval_classifier':
            self.k = int(block_size / output_stride)
            avg_pool_stride = int(output_stride / output_stride)
            self.cls2reg = np.zeros(num_class)
            for i in range(1, num_class):
                lower = 0 if i==1 else np.exp((i-2) * step_log + start_log)
                upper = np.exp((i-1) * step_log + start_log)
                self.cls2reg[i] = (lower+upper) / 2            
            self.cls2reg = torch.FloatTensor(self.cls2reg).cuda()
            self.avg_pool = nn.AvgPool2d((self.k, self.k), stride=avg_pool_stride)
            self.counter = nn.Conv2d(counter_dim, num_class, 1)
        else:
            raise NotImplementedError

        self.weight_init()

    def forward(self, x):
        if self.counter_type == 'count_unfolding_regressor':
            # local count unfolding
            bs, _, h, w = x.size()
            y = F.unfold(x.mean(dim=1).view(bs, 1, h, w), kernel_size=self.k)
            p = F.softmax(y, dim=1)

            x = self.counter(x)

            # spatially divide the count
            x_kxk = x * p.view(bs, self.k**2, x.size()[2], x.size()[3])

            if self.training:
                x_kxk = x_kxk.view(bs, self.k**2, -1)
                x_kxk = F.fold(x_kxk, (h,w), kernel_size=self.k)
                return x, x_kxk
            else:
                return x_kxk
        elif self.counter_type == 'count_unfolding_regressor_segmenter':
            x_count, x_seg = x[0], x[1]

            x_seg = self.segmenter(x_seg)

            # from PIL import Image
            # import matplotlib.pyplot as plt
            # file_name = 'seg.JPG'
            # x_seg_map = x_seg.squeeze().cpu().detach().numpy()
            # cmap = plt.cm.get_cmap('Reds')
            # x_seg_map = cmap(x_seg_map / (x_seg_map.max() + 1e-12)) * 255.
            # Image.fromarray(x_seg_map.astype(np.uint8)).save('./outputs/'+'segmap_'+file_name.replace('.JPG', '.png'))

            # -------------------------
            # MIDDLE FUSION
            x_count = x_count * x_seg
            # -------------------------

            # local count unfolding
            bs, _, h, w = x_count.size()
            y = F.unfold(x_count.mean(dim=1).view(bs, 1, h, w), kernel_size=self.k)
            p = F.softmax(y, dim=1)

            x_count = self.counter(x_count)

            # spatially divide the count
            x_kxk = x_count * p.view(bs, self.k**2, x_count.size()[2], x_count.size()[3])

            if self.training:
                x_kxk = x_kxk.view(bs, self.k**2, -1)
                x_kxk = F.fold(x_kxk, (h,w), kernel_size=self.k)
                return x_count, x_kxk, x_seg
            else:
                return x_kxk, x_seg

        elif self.counter_type == 'count_interval_classifier': 
            x = self.counter(self.avg_pool(x))
            return x if self.training else self.cls2reg[x.data.max(1)[1]].unsqueeze(1)
        else:
            x = self.counter(x)
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


# ------------------------------------------------------------------------
# Normalizer
# ------------------------------------------------------------------------
class Normalizer:
    @staticmethod
    def cpu_normalizer(x, imh, imw, bk_sz, os):
    # deprecated, slow implementation, use gpu_normalizer instead
        def dense_sample2d(x, sx, stride):
            (h,w) = x.shape[:2]
            idx_img = np.zeros((h,w),dtype=float)
            
            th = [i for i in range(0, h-sx+1, stride)]
            tw = [j for j in range(0, w-sx+1, stride)]
            norm_vec = np.zeros(len(th)*len(tw))

            for i in th:
                for j in tw:
                    idx_img[i:i+sx,j:j+sx] = idx_img[i:i+sx,j:j+sx]+1
            idx_img = 1/idx_img
            idx_img = idx_img/sx/sx
            idx = 0
            for i in th:
                for j in tw:
                    norm_vec[idx] =idx_img[i:i+sx,j:j+sx].sum()
                    idx+=1
            return norm_vec
        # CPU normalization
        bs = x.size()[0]
        normx = np.zeros((imh, imw))
        norm_vec = dense_sample2d(normx, bk_sz, os).astype(np.float32)
        x = x.cpu().detach().numpy().reshape(bs, -1) * norm_vec
        return x
    
    @staticmethod
    def gpu_normalizer(x, imh, imw, bk_sz, os):
        _, _, h, w = x.size()            
        accm = torch.cuda.FloatTensor(1, bk_sz*bk_sz, h*w).fill_(1)           
        accm = F.fold(accm, (imh, imw), kernel_size=bk_sz, stride=os)
        accm = 1 / accm
        accm /= bk_sz**2
        accm = F.unfold(accm, kernel_size=bk_sz, stride=os).sum(1).view(1, 1, h, w)
        x *= accm
        return x.squeeze().cpu().detach().numpy()
    
    @staticmethod
    def gpu_normalizer_v3(x_kxk, imh, imw, bk_sz, os):
    # designed for tasselnetv3
        h, w = int(imh/os), int(imw/os)
        k = int(bk_sz / os)
        x_kxk = x_kxk.view(x_kxk.size()[0], k**2, -1)
        accm = torch.cuda.FloatTensor(x_kxk.size()).fill_(1)
        accm = F.fold(accm, (h,w), kernel_size=k)
        x_kxk = F.fold(x_kxk, (h,w), kernel_size=k)
        x_kxk /= accm
        return x_kxk.squeeze().cpu().numpy()

    @staticmethod
    def count_seg_normalizer(x, imh, imw, bk_sz, os):
    # designed for seg2count
        x_kxk, x_seg = x[0], x[1]
        h, w = int(imh/os), int(imw/os)
        k = int(bk_sz / os)
        x_kxk = x_kxk.view(x_kxk.size()[0], k**2, -1)
        accm = torch.cuda.FloatTensor(x_kxk.size()).fill_(1)
        accm = F.fold(accm, (h,w), kernel_size=k)
        x_kxk = F.fold(x_kxk, (h,w), kernel_size=k)
        x_kxk /= accm
        x_kxk = x_kxk.squeeze().cpu().numpy()

        x_seg = x_seg[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)
        x_seg = cv2.resize(x_seg, (imw, imh), interpolation=cv2.INTER_CUBIC)
        x_seg = (x_seg > 0.5).astype(np.uint8)

        return x_kxk, x_seg
    
    @staticmethod
    def no_normalizer(x_kxk, imh, imw, bk_sz, os):
        return x_kxk.squeeze().cpu().numpy()


# ------------------------------------------------------------------------
# Visualizer
# ------------------------------------------------------------------------
class Visualizer:
    @staticmethod
    def pixel_averaging_visualizer(x, im, bk_sz, os):
        x = x.reshape(-1)
        imH, imW = im.shape[2:4]
        cntMap = np.zeros((imH, imW), dtype=float)
        norMap = np.zeros((imH, imW), dtype=float)
        
        H = np.arange(0, imH - bk_sz + 1, os)
        W = np.arange(0, imW - bk_sz + 1, os)
        cnt = 0
        for h in H:
            for w in W:
                pixel_cnt = x[cnt] / bk_sz / bk_sz
                cntMap[h:h+bk_sz, w:w+bk_sz] += pixel_cnt
                norMap[h:h+bk_sz, w:w+bk_sz] += np.ones((bk_sz, bk_sz))
                cnt += 1
        return cntMap / (norMap + 1e-12)
    
    # designed for tasselnetv3
    @staticmethod
    def direct_averaging_visualizer(x, im, bk_sz, os):
        x = x.reshape(-1)
        imH, imW = im.shape[2:4]
        cntMap = np.zeros((imH, imW), dtype=float)
        
        H = np.arange(0, imH - os + 1, os)
        W = np.arange(0, imW - os + 1, os)
        cnt = 0
        for h in H:
            for w in W:
                pixel_cnt = x[cnt] / os / os
                cntMap[h:h+os, w:w+os] += pixel_cnt
                cnt += 1
        return cntMap
    
    @staticmethod
    def densitymap_upsampling_visualizer(x, im, bk_sz, os):
        imH, imW = im.shape[2:4]
        x = cv2.resize(x, (imW,imH), interpolation=cv2.INTER_CUBIC)
        return x


# ------------------------------------------------------------------------
# Ground Truth Generator
# ------------------------------------------------------------------------
class GTGenerator:
    @staticmethod
    def count_map_generator(targets, output_stride, target_filter, **kwargs):
        targets = F.conv2d(targets, target_filter, stride=output_stride)
        return targets
    
    @staticmethod
    def class_map_generator(targets, output_stride, target_filter, start_log, step_log, num_class, **kwargs):
        targets = F.conv2d(targets, target_filter, stride=output_stride)
        class_0 = targets == 0
        targets[class_0] = 1
        label = torch.floor((torch.log(targets)-start_log)/step_log+2).clamp(min=1, max=num_class-1)
        label[class_0] = 0 
        targets=label.detach().long()
        return targets
    
    @staticmethod
    def density_map_generator(targets, gt_downsampling_rate, **kwargs):
        h, w = targets.shape[2:]
        nh = int(h / gt_downsampling_rate) # h, w are divisible by output_stride
        nw = int(w / gt_downsampling_rate)
        targets = F.interpolate(targets, size=(nh, nw), mode='bicubic', align_corners=False) * (gt_downsampling_rate ** 2)
        return targets
    
    @staticmethod
    def count_map_generator_v3(targets, block_size, output_stride, target_filter, target_filter_os, **kwargs):
        targets_bs = F.conv2d(targets, target_filter, stride=output_stride)
        # targets_os should be generated with overlaps
        h, w = targets.size()[2:]
        kernel_size = int(block_size / output_stride)
        targets_os = F.conv2d(targets, target_filter_os, stride=output_stride)
        targets_os = F.unfold(targets_os, kernel_size=kernel_size)
        targets_os = F.fold(targets_os, (int(h/output_stride), int(w/output_stride)), kernel_size=kernel_size)
        return targets_bs, targets_os


class CountingModels(nn.Module):
    def __init__(
        self, 
        encoder='tasselnetv2plus',
        decoder=None,
        counter='count_regressor',
        normalizer='gpu_normalizer',
        generator='count_map_generator',
        visualizer='pixel_averaging_visualizer',
        base_size=256,
        block_size=64, 
        output_stride=8,
        counter_dim=128,
        num_class=60,
        step_log=0.1, 
        start_log=-2,
        pretrain=True,
        fix_bn=True
        ):
        super(CountingModels, self).__init__()
        self.block_size = block_size
        self.output_stride = output_stride

        # build encoder
        self.encoder = Encoder(
            encoder=encoder,
            pretrain=pretrain,
            fix_bn=fix_bn
        )
        # build decoder
        self.decoder = Decoder(
            decoder=decoder,
            base_size=base_size
            )
        # build counter
        self.counter = Counter(
            counter=counter, 
            block_size=block_size, 
            output_stride=output_stride,
            counter_dim=counter_dim,
            start_log=start_log, 
            step_log=step_log, 
            num_class=num_class
        )
        # build normalizer
        self.normalizer = Normalizer.__dict__[normalizer].__func__
        # build ground truth generator
        self.generator = GTGenerator.__dict__[generator].__func__
        # build visualizer
        self.visualizer = Visualizer.__dict__[visualizer].__func__

    def forward(self, x, is_normalize=True):
        imh, imw = x.size()[2:]
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.counter(x)
        if is_normalize:
            x = self.normalizer(x, imh, imw, self.block_size, self.output_stride)
        return x


if __name__ == "__main__":

    from time import time
    # from torch_receptive_field import receptive_field

    bk_sz, os = 64, 8
    imH, imW = 256, 256
    net = CountingModels(
        encoder='tasselnetv3_mixnet_l',
        decoder='no_decoder',
        counter='count_unfolding_regressor',
        normalizer='gpu_normalizer_v3',
        visualizer='direct_averaging_visualizer',
        block_size=bk_sz, 
        output_stride=os,
        counter_dim=64,
        pretrain=True,
        fix_bn=False
    ).cuda()
    # receptive_field(net, input_size=(3, 256, 256))
    with torch.no_grad():
        net.eval()
        x = torch.randn(1, 3, imH, imW).cuda()
        y = net(x)
        print(y.shape)


        # from modelsummary import get_model_summary
        # dump_x = torch.randn(1, 3, imH, imW).cuda()
        # print(get_model_summary(net, dump_x))

    
    # import numpy as np

    # with torch.no_grad():
    #     frame_rate = np.zeros((100, 1))
    #     infer_time = np.zeros((100, 1))
    #     norm_time = np.zeros((100, 1))
        
    #     # normx = np.zeros((912, 1216))
    #     # norm_vec = dense_sample2d(normx, 64, 8).astype(np.float32)

    #     # for i in range(100):
    #     #     x = torch.randn(1, 3, 912, 1216).cuda()
    #     #     torch.cuda.synchronize()
    #     #     start = time()
    #     #     y = net(x)
    #     #     # y = (y.cpu().detach().numpy().reshape(-1) * norm_vec).sum()
    #     #     torch.cuda.synchronize()
    #     #     end = time()
    #     #     running_frame_rate = 1 * float(1 / (end - start))
    #     #     frame_rate[i] = running_frame_rate

    #     for i in range(100):
    #         x = torch.randn(1, 3, imH, imW).cuda()
    #         torch.cuda.synchronize()
    #         start = time()
    #         # inference
    #         y = net(x)

    #         torch.cuda.synchronize()
    #         end1 = time()
    #         # normalization
    #         # y = cpu_normalizer(y, x.size()[2], x.size()[3], insz=insz, os=os)
    #         # y = gpu_normalizer(y, x.size()[2], x.size()[3], insz=insz, os=os)

    #         torch.cuda.synchronize()
    #         end2 = time()

    #         infer_time[i] = end1 - start
    #         norm_time[i] = end2 - end1

    #         running_frame_rate = 1 * float(1 / (end2 - start))
    #         frame_rate[i] = running_frame_rate
    #     print(np.mean(infer_time) * 1000)
    #     print(np.mean(norm_time) * 1000)
    #     print(np.mean(frame_rate))