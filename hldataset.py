# -*- coding: utf-8 -*-
"""
Created on Sat April 12 2020
@author: Hao Lu
"""

import random
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

import dataloader as dl


class RandomCrop(object):
    def __init__(self, random_crop, output_size):
        assert isinstance(output_size, (int, tuple))
        self.random_crop = random_crop
        self.output_size = output_size

    def __call__(self, sample):

        image, target, mask, gtcount = sample['image'], sample['target'], sample['mask'], sample['gtcount']
        if self.random_crop:
            h, w = image.shape[:2]

            shorter_side = min(self.output_size[0], self.output_size[1])
            if h < shorter_side or w < shorter_side:
                r = shorter_side / min(h, w)
                nh = int(np.ceil(h * r))
                nw = int(np.ceil(w * r))
                image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
                target_new = cv2.resize(target, (nw, nh), interpolation = cv2.INTER_CUBIC)
                target = target_new * (target.sum() / (target_new.sum() + 1e-10))
                h, w = image.shape[:2]

            if isinstance(self.output_size, tuple):
                new_h = min(self.output_size[0], h)
                new_w = min(self.output_size[1], w)
                assert (new_h, new_w) == self.output_size
            else:
                crop_size = min(self.output_size, h, w)
                assert crop_size == self.output_size
                new_h = new_w = crop_size

            if gtcount > 0:
                target_mask = target > 0
                ch, cw = int(np.ceil(new_h / 2)), int(np.ceil(new_w / 2))
                mask_center = np.zeros((h, w), dtype=np.uint8)
                mask_center[ch:h-ch+1, cw:w-cw+1] = 1
                target_mask = (target_mask & mask_center)
                idh, idw = np.where(target_mask == 1)
                if len(idh) != 0:
                    ids = random.choice(range(len(idh)))
                    hc, wc = idh[ids], idw[ids]
                    top, left = hc-ch, wc-cw
                else:
                    top = np.random.randint(0, h-new_h+1)
                    left = np.random.randint(0, w-new_w+1)
            else:
                top = np.random.randint(0, h-new_h+1)
                left = np.random.randint(0, w-new_w+1)

            image = image[top:top+new_h, left:left+new_w, :]
            target = target[top:top+new_h, left:left+new_w]
            mask = mask[top:top+new_h, left:left+new_w]

        return {'image': image, 'target': target, 'mask': mask, 'gtcount': gtcount}


class RandomFlip(object):
    def __init__(self, random_flip=True):
        self.random_flip = random_flip

    def __call__(self, sample):
        image, target, mask, gtcount = sample['image'], sample['target'], sample['mask'], sample['gtcount']
        if self.random_flip:
            do_mirror = np.random.randint(2)
            if do_mirror:
                image = cv2.flip(image, 1)
                target = cv2.flip(target, 1)
                mask = cv2.flip(mask, 1)
        return {'image': image, 'target': target, 'mask': mask, 'gtcount': gtcount}


class Normalize(object):
    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, target, mask, gtcount = sample['image'], sample['target'], sample['mask'], sample['gtcount']
        image = image.astype('float32')

        # pixel normalization
        image = (self.scale * image - self.mean) / self.std

        image, target, mask = image.astype('float32'), target.astype('float32'), mask.astype('float32')

        return {'image': image, 'target': target, 'mask': mask, 'gtcount': gtcount}


class ZeroPadding(object):
    def __init__(self, psize=32):
        self.psize = psize

    def __call__(self, sample):
        psize =  self.psize

        image, target, mask, gtcount = sample['image'], sample['target'], sample['mask'], sample['gtcount']
        h, w = image.size()[-2:]
        ph, pw = (psize-h%psize),(psize-w%psize)
        # print(ph,pw)

        (pl, pr) = (pw//2, pw-pw//2) if pw != psize else (0, 0)
        (pt, pb) = (ph//2, ph-ph//2) if ph != psize else (0, 0)
        if (ph!=psize) or (pw!=psize):
            tmp_pad = [pl, pr, pt, pb]
            # print(tmp_pad)
            image = F.pad(image, tmp_pad)
            target = F.pad(target, tmp_pad)
            mask = F.pad(mask, tmp_pad)

        return {'image': image, 'target': target, 'mask': mask, 'gtcount': gtcount}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        image, target, mask, gtcount = sample['image'], sample['target'], sample['mask'], sample['gtcount']
        image = image.transpose((2, 0, 1))
        target = np.expand_dims(target, axis=2)
        target = target.transpose((2, 0, 1))
        mask = np.expand_dims(mask, axis=2)
        mask = mask.transpose((2, 0, 1))
        image, target, mask = torch.from_numpy(image), torch.from_numpy(target), torch.from_numpy(mask)
        return {'image': image, 'target': target, 'mask': mask, 'gtcount': gtcount}


dataloader_list = {
    # plant counting
    'mtc': dl.MaizeTasselDataset,
    'mtc-uav': dl.MaizeTasselUAVDataset,
    'wec': dl.WhearEarDataset,
    'shc': dl.SorghumHeadDataset,
    'rsc': dl.RiceSeedingDataset,
    'mkc': dl.MazieKernelDataset,
    'msc': dl.MaizeSeedingDataset,
    # crowd counting
    'sht': dl.ShanghaiTechDataset,
    # crystal counting
    'pcc': dl.PhenocrystDataset,
    'puecc': dl.PhenocrystUniformEllipseDataset,
    'pbecc': dl.PhenocrystBivariateEllipseDataset
}


if __name__=='__main__':

    dataset = dl.PhenocrystUniformEllipseDataset(
        data_dir='../data/phenocryst_counting_dataset', 
        data_list='../data/phenocryst_counting_dataset/train.txt',
        mask_dir = './',
        ratio=1, 
        sigma=5,
        scaling=1,
        transform=transforms.Compose([
            ToTensor()]
        )
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(len(dataloader))
    mean = 0.
    std = 0.
    for i, data in enumerate(dataloader, 0):
        images, targets = data['image'], data['target']
        bs = images.size(0)
        images = images.view(bs, images.size(1), -1).float()
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        print(images.size())
        print(i) 
    mean /= len(dataloader)
    std /= len(dataloader)
    print(mean/255.)
    print(std/255.)