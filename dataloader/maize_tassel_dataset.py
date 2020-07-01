# -*- coding: utf-8 -*-
"""
Created on Sat April 12 2020
@author: Hao Lu
"""

import json
import os
import cv2
import scipy.io as sio
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import morphology

from torch.utils.data import Dataset


def read_image(x):
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:  # grayscale
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr


class MaizeTasselDataset(Dataset):
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

    def bbs2points(self, bbs):    
        points = []
        for bb in bbs:
            x1, y1, w, h = [float(b) for b in bb]
            x2, y2 = x1+w-1, y1+h-1
            x, y = np.round((x1+x2)/2).astype(np.int32), np.round((y1+y2)/2).astype(np.int32)
            points.append([x, y])
        return points

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        self.image_list.append(file_name[0])
        if file_name[0] in self.images:
            image = self.images[file_name]
            target = self.targets[file_name]
            gtcount = self.gtcounts[file_name]
            mask = self.masks[file_name]
        else:
            image = read_image(self.data_dir+file_name[0])
            annotation = sio.loadmat(self.data_dir+file_name[1])
            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            # dotimage = image.copy()
            if annotation['annotation'][0][0][1] is not None:
                bbs = annotation['annotation'][0][0][1]
                gtcount = bbs.shape[0]
                pts = self.bbs2points(bbs)
                for pt in pts:
                    pt[0], pt[1] = int(pt[0] * self.ratio), int(pt[1] * self.ratio)
                    target[pt[1], pt[0]] = 1
                    # cv2.circle(dotimage, (pt[0], pt[1]), 3, (255, 0, 0), -1)
            else:
                gtcount = 0
            dotimage = target.copy().astype(np.uint8)
            target = gaussian_filter(target, self.sigma * self.ratio) * self.scaling
            
            mask = morphology.distance_transform_edt(dotimage==0) <= 32
            mask = mask.astype(np.int)

            # plt.imshow(target, cmap=cm.jet)
            # plt.show()
            # print(target.sum())

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


class MaizeTasselUAVDataset(Dataset):
    def __init__(self, data_dir, mask_dir, data_list, ratio, sigma, scaling, preload=True, transform=None):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.ratio = ratio
        self.sigma = sigma
        self.scaling = scaling
        self.preload = preload
        self.transform = transform
        
        self.annotations, self.image_list = self.parse_csv(self.data_list)
        
        self.images = {}
        self.targets = {}
        self.gtcounts = {}
        self.dotimages = {}
        self.masks = {}

    def parse_csv(self, data_list):
        annotations = {}
        image_list = []
        for file_path in data_list:
            image_path = file_path[0]
            _, name = os.path.split(image_path)
            image_list.append(name)

            csv_path = self.data_dir+file_path[1]
            lists = pd.read_csv(csv_path, sep=',', header=None).values
            lists = lists[1:]
            for entry in lists:
                image_name = entry[0]
                point = json.loads(entry[5])
                pt = [point['cx'], point['cy']] if point else None
                if os.path.splitext(image_name)[1] != '.JPG':
                    continue
                if image_name not in annotations:
                    annotations[image_name] = []
                    annotations[image_name].append(pt)
                else:
                    annotations[image_name].append(pt)
        return annotations, image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        file_path = self.data_list[idx][0]
        _, file_name = os.path.split(file_path)

        if file_name in self.images:
            image = self.images[file_name]
            target = self.targets[file_name]
            gtcount = self.gtcounts[file_name]
            mask = self.masks[file_name]
        else:
            image = read_image(self.data_dir+file_path)
            pts = self.annotations[file_name]
            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            # dotimage = image.copy()
            pts = [pt for pt in pts if pt is not None]
            if pts[0] is not None:
                gtcount = len(pts)
                for pt in pts:
                    w, h = np.array(pt[0]) * self.ratio, np.array(pt[1]) * self.ratio
                    w, h = int(w), int(h)
                    target[h, w] = 1
                    # cv2.circle(dotimage, (w, h), 5, (255, 0, 0), -1)
            else:
                gtcount = 0
            dotimage = target.copy().astype(np.uint8)
            target = gaussian_filter(target, self.sigma * self.ratio) * self.scaling
            
            mask = morphology.distance_transform_edt(dotimage==0) <= 32
            mask = mask.astype(np.uint8)
            # Image.fromarray((mask * 255).astype(np.uint8)).save('./outputs/'+'mask_'+file_name.replace('.JPG', '.png'))
            
            # _, ext = os.path.splitext(file_name)
            # mask_path = os.path.join(self.mask_dir, file_name.replace(ext, '.png'))
            # if os.path.isfile(mask_path):
            #     mask = read_image(mask_path)
            #     mask = mask[:, :, 0] / 255.
            # else:
            #     mask = np.zeros((nh, nw), dtype=np.float32)

            # plt.imshow(target, cmap=cm.jet)
            # plt.show()
            # print(target.sum())
            
            # cmap = plt.cm.get_cmap('jet')
            # target_map = cmap(target / (target.max() + 1e-12)) * 255.
            # image = 0.5 * image + 0.5 * target_map[:, :, 0:3]
            # Image.fromarray(image.astype(np.uint8)).save('./outputs/'+file_name)
            
            # Image.fromarray(dotimage*255).save('./outputs/'+file_name)
            
            # save dotimages for visualization
            if file_name not in self.dotimages:
                self.dotimages.update({file_name:dotimage})
            
            if self.preload:
                self.images.update({file_name:image})
                self.targets.update({file_name:target})
                self.masks.update({file_name:mask})
                self.gtcounts.update({file_name:gtcount})

        sample = {
            'image': image, 
            'target': target, 
            'mask': mask,
            'gtcount': gtcount
        }

        if self.transform:
            sample = self.transform(sample)

        return sample