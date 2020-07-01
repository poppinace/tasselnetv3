# -*- coding: utf-8 -*-
"""
Created on Sun April 13 2020
@author: Hao Lu
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import morphology

from torch.utils.data import Dataset

def read_image(x):
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:  # grayscale
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr


class RiceSeedingDataset(Dataset):
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
            image = read_image(os.path.join(self.data_dir, file_name[0]))
            annotation=sio.loadmat(os.path.join(self.data_dir, file_name[1]))
            annotation= annotation['annPoints'][:]
            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            # dotimage = image.copy()
            if annotation is not None:
                pts = annotation
                gtcount = pts.shape[0]
                for pt in pts:
                    x, y = int(np.floor(pt[0] * self.ratio)), int(np.floor(pt[1] * self.ratio))
                    x, y = x - 1, y - 1
                    if x >= w or y >= h:
                        continue
                    target[y, x] = 1
                    # cv2.circle(dotimage, (x, y), 6, (255, 0, 0), -1)
            else:
                gtcount = 0
            dotimage = target.copy().astype(np.uint8)
            target = gaussian_filter(target, self.sigma * self.ratio) * self.scaling
            
            mask = morphology.distance_transform_edt(dotimage==0) <= 16
            mask = mask.astype(np.uint8)

            # plt.imshow(target, cmap=cm.jet)
            # plt.show()
            # print(target.sum())

            # if not os.path.exists('./outputs'):
            #     os.mkdir('./outputs')
            # _, image_name = os.path.split(file_name[0])
            # cmap = cm.get_cmap('jet')
            # target_map = cmap(target / (target.max() + 1e-12)) * 255.
            # image = 0.5 * image + 0.5 * target_map[:, :, 0:3]
            # Image.fromarray(image.astype(np.uint8)).save('./outputs/'+image_name)

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