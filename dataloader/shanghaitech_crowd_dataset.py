# -*- coding: utf-8 -*-
"""
Created on Mon April 13 2020
@author: Hao Lu
"""

import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

from torch.utils.data import Dataset


def read_image(x):
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:  # grayscale
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr


class ShanghaiTechDataset(Dataset):
    def __init__(self, data_dir, data_list, ratio, sigma, scaling, preload=True, transform=None):
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

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        self.image_list.append(file_name[0])
        if file_name[0] in self.images:
            image = self.images[file_name[0]]
            target = self.targets[file_name[0]]
            gtcount = self.gtcounts[file_name[0]]
        else:
            image = read_image(self.data_dir+file_name[0])
            annotation = sio.loadmat(self.data_dir+file_name[1])
            h, w = image.shape[:2]
            r = 288. / min(h, w) if min(h, w) < 288 else self.ratio
            nh = int(np.ceil(h * r))
            nw = int(np.ceil(w * r))
            image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            dotimage = image.copy()
            if annotation['image_info'][0][0][0][0][0] is not None:
                pts = annotation['image_info'][0][0][0][0][0]
                gtcount = pts.shape[0]
                for pt in pts:
                    x, y = int(np.floor(pt[0] * r)), int(np.floor(pt[1] * r))
                    x, y = x - 1, y - 1
                    if x >= w or y >= h:
                        continue
                    target[y, x] = 1
                    cv2.circle(dotimage, (x, y), 3, (255, 0, 0), -1)
            else:
                gtcount = 0
            target = gaussian_filter(target, self.sigma) * self.scaling

            # plt.imshow(target, cmap=cm.jet)
            # plt.show()
            # print(target.sum())

            # save dotimages for visualization
            if file_name[0] not in self.dotimages:
                self.dotimages.update({file_name[0]:dotimage})
            
            if self.preload:
                self.images.update({file_name[0]:image})
                self.targets.update({file_name[0]:target})
                self.gtcounts.update({file_name[0]:gtcount})

        sample = {
            'image': image, 
            'target': target, 
            'gtcount': gtcount
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

