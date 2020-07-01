# -*- coding: utf-8 -*-
"""
Created on Tun April 14 2020
@author: Hao Lu
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label, regionprops

from torch.utils.data import Dataset
import torch


def read_image(x):
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:  # grayscale
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr

    
class PhenocrystDataset(Dataset):
    def __init__(self, data_dir, data_list, ratio, sigma, scaling, preload=True, transform=None):
        self.data_dir = data_dir
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
                polyline = json.loads(entry[5])
                pt = [polyline['all_points_x'], polyline['all_points_y']] if polyline else None
                if os.path.splitext(image_name)[1] != '.jpg':
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
        else:
            image = read_image(self.data_dir+file_path)
            pts = self.annotations[file_name]
            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            dotimage = image.copy()
            if pts[0] is not None:
                gtcount = len(pts)
                for pt in pts:
                    pt[0], pt[1] = np.array(pt[0]) * self.ratio, np.array(pt[1]) * self.ratio
                    pt[0], pt[1] = int(np.mean(pt[0])), int(np.mean(pt[1]))
                    target[pt[1], pt[0]] = 1
                    cv2.circle(dotimage, (pt[0], pt[1]), 3, (255, 0, 0), -1)
            else:
                gtcount = 0
            target = gaussian_filter(target, self.sigma * self.ratio) * self.scaling

            # plt.imshow(target, cmap=cm.jet)
            # plt.show()
            # print(target.sum())
            
            # cmap = plt.cm.get_cmap('jet')
            # target_map = cmap(target / (target.max() + 1e-12)) * 255.
            # image = 0.5 * image + 0.5 * target_map[:, :, 0:3]
            # Image.fromarray(image.astype(np.uint8)).save('./outputs/'+file_name)
            
            # save dotimages for visualization
            if file_name not in self.dotimages:
                self.dotimages.update({file_name:dotimage})
            
            if self.preload:
                self.images.update({file_name:image})
                self.targets.update({file_name:target})
                self.gtcounts.update({file_name:gtcount})

        sample = {
            'image': image, 
            'target': target, 
            'gtcount': gtcount
        }


        if self.transform:
            sample = self.transform(sample)

        return sample

    
class PhenocrystUniformEllipseDataset(Dataset):
   def __init__(self, data_dir, mask_dir, data_list, ratio, sigma, scaling, preload=True, transform=None):
        self.data_dir = data_dir
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
        self.masks = {}
        self.dotimages = {}

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
                polyline = json.loads(entry[5])
                pt = [polyline['all_points_x'], polyline['all_points_y']] if polyline else None
                if os.path.splitext(image_name)[1] != '.jpg':
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
            dotimage = image.copy()
                        
            target_line = np.zeros((nh, nw), dtype=np.float32)  
            target_mask = np.zeros((nh, nw), dtype=np.float32)
            
            if pts[0] is not None:                
                gtcount = len(pts)
                for pt in pts:
                    pt = np.array(pt)
                    pt[0], pt[1] = (pt[0]) * self.ratio, (pt[1]) * self.ratio
                    x, y = int(np.mean(pt[0])), int(np.mean(pt[1]))   
                    for i in range(0,pt.shape[1]):
                        cv2.line(target_line, (x,y), (pt[0,i], pt[1,i]), (1),2)                
                    target_mask[y,x] = 1
                    cv2.circle(dotimage, (x, y), 3, (255, 0, 0), -1)
            else:
                gtcount = 0
                
            regions = label(target_line, background=0) 
            regions = regionprops(regions)
            
            for props in regions:
                y0, x0 = props.centroid
                major, minor = props.major_axis_length * 0.5, props.minor_axis_length * 0.5
                
                minr, minc, maxr, maxc = props.bbox      
                target_t = np.zeros((2 * (maxr - minr), 2 * (maxc - minc)), dtype = np.float32)            
                cv2.ellipse(target_t, 
                            (int(x0 - minc + (maxc - minc) / 2), int(y0 - minr + (maxr - minr) / 2)),
                            (int(major), int( minor)),
                            (-props.orientation / 3.141592653 * 180), 
                            0,    #start angle
                            360,  #end angle
                            1,    #filling value
                            -1)   #-1 means fill the ellipse area with filling value
                                
                min_ori_r = max(0, int(minr - (maxr - minr) / 2))
                max_ori_r = min(nh, int(maxr + (maxr - minr) / 2))
                min_ori_c = max(0, int(minc - (maxc - minc) / 2))
                max_ori_c = min(nw, int(maxc + (maxc - minc) / 2))
                
                error_min_r = abs(int(minr - (maxr - minr) / 2) - min_ori_r)
                error_min_c = abs(int(minc - (maxc - minc) / 2) - min_ori_c)
                min_target_r = error_min_r
                min_target_c = error_min_c
                max_target_r = min_target_r + (max_ori_r - min_ori_r)
                max_target_c = min_target_c + (max_ori_c - min_ori_c)
    
                number_point=(target_t[min_target_r:max_target_r, min_target_c:max_target_c]).sum()
                
                mask_t = np.zeros((nh, nw))
                for cood in props.coords:                
                    if cood[1] >= 0 and cood[1] < nw and cood[0] >= 0 and cood[0] < nh:
                        mask_t[cood[0], cood[1]] = 1
                mask_t = mask_t[min_ori_r:max_ori_r, min_ori_c:max_ori_c]
                number_region = (target_mask[min_ori_r:max_ori_r, min_ori_c:max_ori_c] * (mask_t)).sum()
                target[min_ori_r:max_ori_r, min_ori_c:max_ori_c] += (target_t[min_target_r:max_target_r, min_target_c:max_target_c] * (number_region / number_point))
            
            # # visualize density map
            # cmap = plt.cm.get_cmap('jet')
            # # target_map = cmap(target / (target.max() + 1e-12)) * 255.
            # target_map = cmap((target > 0).astype(np.float32)) * 255.
            # image = 0.5 * image + 0.5 * target_map[:, :, 0:3]
            # Image.fromarray(image.astype(np.uint8)).save('./outputs/'+file_name)
            
            mask = (target > 0).astype(np.uint8)

            # scaling ground truth
            target = target * self.scaling
       
            if file_name not in self.dotimages:
                self.dotimages.update({file_name:dotimage})
            
            if self.preload:
                self.images.update({file_name:image})
                self.targets.update({file_name:target})
                self.gtcounts.update({file_name:gtcount})
                self.masks.update({file_name:mask})

        sample = {
            'image': image, 
            'target': target, 
            'gtcount': gtcount,
            'mask': mask
        }


        if self.transform:
            sample = self.transform(sample)

        return sample


class PhenocrystBivariateEllipseDataset(Dataset):
    def __init__(self, data_dir, data_list, ratio, sigma, scaling, preload=True, transform=None, show=False):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.ratio = ratio
        self.sigma = sigma
        self.scaling = scaling
        self.preload = preload
        self.transform = transform
        
        self.show = show
        self.annotations, self.image_list = self.parse_csv(self.data_list)
        
        self.images = {}
        self.targets = {}
        self.gtcounts = {}
        self.dotimages = {}
    
    def multivariate_gaussian(self, pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.
    
        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.
        """
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(np.negative(fac) / 2) / N

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
                polyline = json.loads(entry[5])
                pt = [polyline['all_points_x'], polyline['all_points_y']] if polyline else None
                if os.path.splitext(image_name)[1] != '.jpg':
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
        else:
            image = read_image(self.data_dir+file_path)
            pts = self.annotations[file_name]
            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            dotimage = image.copy()
                        
            target_line = np.zeros((nh, nw), dtype=np.float32)  
            target_mask = np.zeros((nh, nw), dtype=np.float32)
            target_mask_ind = np.zeros((nh, nw), dtype=np.int16)            
            ellipset_mask = np.zeros((nh, nw), dtype=np.float32)
            
            id_region = 0
            if pts[0] is not None:                
                gtcount = len(pts)
                for pt in pts:
                    pt = np.array(pt)
                    pt[0], pt[1] = (pt[0]) * self.ratio, (pt[1]) * self.ratio
                    x, y = int(np.mean(pt[0])), int(np.mean(pt[1]))   
                    for i in range(0,pt.shape[1]):
                        cv2.line(target_line, (x,y), (pt[0,i], pt[1,i]), (1), 2)                
                    target_mask[y,x] = 1
                    target_mask_ind[y,x] = id_region
                    cv2.circle(dotimage, (x, y), 6, (255, 0, 0), -1)
                    id_region += 1
            else:
                gtcount = 0
                
            regions = label(target_line, background=0) 
            regions = regionprops(regions)
            
            for props in regions:
                y0, x0 = props.centroid
            
                major0, minor0 = int(props.major_axis_length * 0.5), int(props.minor_axis_length * 0.5)
                                
                x_t = np.linspace(0, 4 * major0, 4 * major0+1)
                y_t = np.linspace(0, 4 * major0, 4 * major0+1)
                x_t, y_t = np.meshgrid(x_t, y_t)
                pos = np.empty(x_t.shape + (2,))
                pos[:, :, 0] = x_t
                pos[:, :, 1] = y_t
                
                mu = np.array([2 * major0, 2 * major0])                
                sigma = np.array([[self.sigma * self.ratio * major0, 0], [0, self.sigma * self.ratio * minor0]])
                target_t = self.multivariate_gaussian(pos, mu, sigma)
                matRotate = cv2.getRotationMatrix2D((2 * major0, 2 * major0),  int(props.orientation / 3.141592653 * 180), 1)              
                target_t = cv2.warpAffine(target_t, matRotate, (4 * major0 + 1, 4 * major0 + 1))
                
                target_ell_mask = np.zeros((4 * major0 + 1, 4 * major0 + 1))
                cv2.ellipse(target_ell_mask, 
                            (int(2 * major0), int(2 * major0)),
                            (int(major0), int(minor0)),
                            (-props.orientation / 3.141592653 * 180), 
                            0,    #start angle
                            360,  #end angle
                            1,    #filling value
                            -1)   #-1 means fill the ellipse area with filling value

                
                if self.show:
                    target_ell_mask_show = np.zeros((4 * major0 + 1, 4 * major0 + 1))
                    cv2.ellipse(target_ell_mask_show, 
                                (int(2 * major0), int(2 * major0)),
                                (int(major0), int(minor0)),
                                (-props.orientation / 3.141592653 * 180), 
                                0,    #start angle
                                360,  #end angle
                                1,    #filling value
                                1)   #-1 means fill the ellipse area with filling value
                            
                min_ori_r = max(0, int(y0 - 2 * major0))
                max_ori_r = min(nh, int(y0 + 2 * major0))
                min_ori_c = max(0, int(x0 - 2 * major0))
                max_ori_c = min(nw, int(x0 + 2 * major0))
                
                error_min_r = abs(int(y0 - 2 * major0) - min_ori_r)
                error_min_c = abs(int(x0 - 2 * major0) - min_ori_c)
                min_target_r = error_min_r
                min_target_c = error_min_c
                max_target_r = min_target_r + (max_ori_r - min_ori_r)
                max_target_c = min_target_c + (max_ori_c - min_ori_c)
                    
                mask_t = np.zeros((nh, nw))
                for cood in props.coords:                
                    if cood[1] >= 0 and cood[1] < nw and cood[0] >= 0 and cood[0] < nh:
                        mask_t[cood[0], cood[1]] = 1
                mask_t = mask_t[min_ori_r:max_ori_r, min_ori_c:max_ori_c]
                number_region = (target_mask[min_ori_r:max_ori_r, min_ori_c:max_ori_c] * (mask_t)).sum()
                
                if number_region>1:
                    idx_position = np.nonzero(target_mask[min_ori_r:max_ori_r, min_ori_c:max_ori_c] * (mask_t))
                    idx_position = np.array(idx_position).T
                    
                    for position_each in list(idx_position):
                        target_mask_ind_t=target_mask_ind[min_ori_r:max_ori_r, min_ori_c:max_ori_c]
                        ind = target_mask_ind_t[position_each[0], position_each[1]]                        
                        target_t_multi_each = np.zeros((4 * major0, 4 * major0), dtype = np.float32) 
                         
                        pt = np.array(pts[ind]).copy()
                        pt[0], pt[1] = (pt[0]) * self.ratio, (pt[1]) * self.ratio
                        x, y = int(np.mean(pt[0])), int(np.mean(pt[1]))
                        for i in range(0, pt.shape[1]):
                            cv2.line(target_t_multi_each, (x - min_ori_c, y - min_ori_r), (pt[0,i] - min_ori_c, pt[1,i] - min_ori_r), (255), 1)
                        
                        regions_each = label(target_t_multi_each, background=0) 
                        regions_each = regionprops(regions_each)

                        for props_each in regions_each:
                            y, x = props_each.centroid
                            major, minor = int(props.major_axis_length * 0.5), int(props.minor_axis_length * 0.5)
                           
                            x_t = np.linspace(0, 4 * major, 4 * major + 1)
                            y_t = np.linspace(0, 4 * major, 4 * major + 1)
                            x_t, y_t = np.meshgrid(x_t, y_t)
                            pos = np.empty(x_t.shape + (2,))
                            pos[:, :, 0] = x_t
                            pos[:, :, 1] = y_t
                
                            mu = np.array([2 * major, 2 * major])
                            sigma = np.array([[self.sigma * self.ratio * major, 0], [0, self.sigma * self.ratio * minor]])
                            target_t_multi = self.multivariate_gaussian(pos, mu, sigma)
                            matRotate = cv2.getRotationMatrix2D((2 * major, 2 * major),  int(-props.orientation / 3.141592653 * 180), 1)
                            target_t_multi = cv2.warpAffine(target_t_multi, matRotate, (4 * major + 1, 4 * major + 1))
                            
                            target_ell_mask_multi = np.zeros((4 * major + 1, 4 * major + 1))               
                            cv2.ellipse(target_ell_mask_multi, 
                                        (2*int(major), 2*int(major)),
                                        (int(major), int(minor)),
                                        (-props.orientation / 3.141592653 * 180), 
                                        0,    #start angle
                                        360,  #end angle
                                        1,    #filling value
                                        -1)   #-1 means fill the ellipse area with filling value
                            
                            if self.show:       
                                cv2.ellipse(ellipset_mask, 
                                            (int(x + min_ori_c), int(y + min_ori_r)),
                                            (int(major), int(minor)),
                                            (-props.orientation / 3.141592653 * 180), 
                                            0,    #start angle
                                            360,  #end angle
                                            1,    #filling value
                                            1)   #-1 means fill the ellipse area with filling value
                            
                            min_ori_r_multi = max(0, int(y - 2 * major + min_ori_r))
                            max_ori_r_multi = min(nh, int(y + 2 * major + min_ori_r))
                            min_ori_c_multi = max(0, int(x - 2 * major + min_ori_c))
                            max_ori_c_multi = min(nw, int(x + 2 * major + min_ori_c))
                            
                            error_min_r_multi = abs(int(y - 2 * major + min_ori_r) - min_ori_r_multi)
                            error_min_c_multi = abs(int(x - 2 * major + min_ori_c) - min_ori_c_multi)
                            min_target_r_multi = error_min_r_multi
                            min_target_c_multi = error_min_c_multi
                            max_target_r_multi = min_target_r_multi + (max_ori_r_multi - min_ori_r_multi)
                            max_target_c_multi = min_target_c_multi + (max_ori_c_multi - min_ori_c_multi)
                
                            target_t_multi=target_t_multi * target_ell_mask_multi
                            target_t_multi=target_t_multi / (target_t_multi.sum() +1e-12)
                            
                            target[min_ori_r_multi:max_ori_r_multi, min_ori_c_multi:max_ori_c_multi] += target_t_multi [min_target_r_multi:max_target_r_multi, min_target_c_multi:max_target_c_multi]                      
                           
                            if self.show:
                                cv2.rectangle(ellipset_mask, (int(min_ori_c_multi),int(min_ori_r_multi)), (int(max_ori_c_multi),int(max_ori_r_multi)),1,1)
                else:
                    target_t=target_t * target_ell_mask
                    target_t=target_t / (target_t.sum() +1e-12)
                    target[min_ori_r:max_ori_r, min_ori_c:max_ori_c] += target_t[min_target_r:max_target_r, min_target_c:max_target_c] 
                    if self.show:
                        ellipset_mask[min_ori_r:max_ori_r, min_ori_c:max_ori_c] += (target_ell_mask_show[min_target_r:max_target_r, min_target_c:max_target_c])  
                        cv2.rectangle(ellipset_mask, (min_ori_c,min_ori_r), (max_ori_c,max_ori_r),1,1)
            
            # scaling ground truth
            target = target * self.scaling

            if self.show:
                cmap = plt.cm.get_cmap('jet')
                target_map=target.copy()
                # target_map[target_map>0]=255
                target_map = cmap(target_map / (target.max() + 1e-12)) * 255
                line_map = cmap(target_line.copy())*255
                mask_map = cmap(ellipset_mask.copy())*255
                image = 0.5 * image + 0.5 * target_map[:, :, 0:3]
                    # 0.5 + line_map[:, :, 0:3] + \
                    # 0.5 * mask_map[:, :, 0:3] + \
                    # 0.5 * target_map[:, :, 0:3]
                image[image>255]=255
                Image.fromarray(image.astype(np.uint8)).save('./outputs/'+file_name)
                print(target.sum()-target_mask.sum(), idx)


            if file_name not in self.dotimages:
                self.dotimages.update({file_name:dotimage})
            
            if self.preload:
                self.images.update({file_name:image})
                self.targets.update({file_name:target})
                self.gtcounts.update({file_name:gtcount})

        sample = {
            'image': image, 
            'target': target, 
            'gtcount': gtcount
        }


        if self.transform:
            sample = self.transform(sample)

        return sample
