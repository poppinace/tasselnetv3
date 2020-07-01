# -*- coding: utf-8 -*-
"""
Created on Sat April 12 2020
@author: Hao Lu
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xml.etree.ElementTree as ET
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import morphology

from torch.utils.data import Dataset


def read_image(x):
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:  # grayscale
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr


class WhearEarDataset(Dataset):
    def __init__(self, data_dir, mask_dir, data_list, ratio, sigma, scaling, preload=True, transform=None):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.ratio = ratio
        self.sigma = sigma
        self.scaling = scaling
        self.preload = preload
        self.transform = transform
        
        # store images and generate ground truths
        self.image_list = []
        self.images = {}
        self.targets = {}
        self.gtcounts = {}
        self.dotimages = {}
        self.masks = {}

    def parsexml(self, xml):
        tree = ET.parse(xml)
        root = tree.getroot()
        bbs = []
        for bb in root.iter('bndbox'):
            xmin = int(bb.find('xmin').text)
            ymin = int(bb.find('ymin').text)
            xmax = int(bb.find('xmax').text)
            ymax = int(bb.find('ymax').text)
            bbs.append([xmin, ymin, xmax, ymax])
        return bbs

    def bbs2points(self, bbs):    
        points = []
        for bb in bbs:
            x1, y1, x2, y2 = [float(b) for b in bb]
            x, y = np.round((x1+x2)/2).astype(np.int32), np.round((y1+y2)/2).astype(np.int32)
            points.append([x, y])
        return points

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        self.image_list.append(file_name[0])
        if file_name[0] in self.images:
            image = self.images[file_name[0]]
            target = self.targets[file_name[0]]
            gtcount = self.gtcounts[file_name[0]]
            mask = self.masks[file_name[0]]
        else:
            image = read_image(self.data_dir+file_name[0])
            bbs = self.parsexml(self.data_dir+file_name[1])
            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            # dotimage = image.copy()
            if bbs is not None:
                gtcount = len(bbs)
                pts = self.bbs2points(bbs)
                for pt in pts:
                    pt[0], pt[1] = int(pt[0] * self.ratio), int(pt[1] * self.ratio)
                    target[pt[1], pt[0]] = 1
                    # cv2.circle(dotimage, (pt[0], pt[1]), 6, (255, 0, 0), -1)
            else:
                gtcount = 0
            dotimage = target.copy().astype(np.uint8)
            target = gaussian_filter(target, self.sigma * self.ratio)

            mask = morphology.distance_transform_edt(dotimage==0) <= 32
            mask = mask.astype(np.uint8)

            # cmap = plt.cm.get_cmap('jet')
            # target_show = target / (target.max() + 1e-12)
            # target_show = cmap(target_show) * 255.
            # target_show = 0.5 * image + 0.5 * target_show[:, :, 0:3]
            # plt.imshow(target_show.astype(np.uint8))
            # plt.show()
            # print(target.sum())

            # plt.imshow(dotimage.astype(np.uint8))
            # plt.show()

            # save dotimages for visualization
            if file_name[0] not in self.dotimages:
                self.dotimages.update({file_name[0]:dotimage})
            
            if self.preload:
                self.images.update({file_name[0]:image})
                self.targets.update({file_name[0]:target})
                self.masks.update({file_name[0]:mask})
                self.gtcounts.update({file_name[0]:gtcount})

        sample = {
            'image': image, 
            'target': target, 
            'mask': mask,
            'gtcount': gtcount
        }

        if self.transform:
            sample = self.transform(sample)

        return sample