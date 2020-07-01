# -*- coding: utf-8 -*-
"""
Created on Sat April 12 2020
@author: hao lu
"""

import os
import scipy
import numpy as np
import math
import re
from PIL import Image
import functools
import cv2 as cv
from scipy.ndimage import gaussian_filter, morphology
from skimage.measure import label, regionprops
from sklearn import linear_model
import matplotlib.pyplot as plt
import torch


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))
    

REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):

    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret


def compute_mae(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mae = np.mean(np.abs(diff))
    return mae


def compute_mse(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mse = np.sqrt(np.mean((diff ** 2)))
    return mse


def compute_relerr(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    diff = diff[gt > 0]
    gt = gt[gt > 0]
    if (diff is not None) and (gt is not None):
        rmae = np.mean(np.abs(diff) / gt) * 100
        rmse = np.sqrt(np.mean(diff**2 / gt**2)) * 100
    else:
        rmae = 0
        rmse = 0
    return rmae, rmse


def rsquared(pd, gt):
    """ Return R^2 where x and y are array-like."""
    pd, gt = np.array(pd), np.array(gt)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pd, gt)
    return r_value**2


def plot_r2(pd, gt, cfg):

    r2 = rsquared(pd,gt)

    pd, gt = np.array(pd).reshape(-1, 1), np.array(gt).reshape(-1, 1)

    upper = int(max(max(pd), max(gt)))
    ww=list(range(0, upper+50, 50))
    LR=linear_model.LinearRegression()
    LR.fit(gt,pd)
    predictions = LR.predict(gt)
    
    # print(LR.coef_,LR.intercept_)
    # print("R^2={0:.4f}".format(r2))
    
    plt.figure(dpi=300) 
    plt.ylim(0, max(ww))
    plt.xlim(0, max(ww))
    _, dataset_name = os.path.split(cfg.DIR.dataset)
    plt.scatter(gt, pd, color='red', marker = '.')
    plt.plot(gt, predictions, color='black',)
    plt.plot(ww, ww, color='black',  linestyle=':')
    plt.annotate(r"$R^2={0:.4f}$".format(r2), (50, int(0.75*upper)), xycoords='data',
            xytext=(50, int(0.75*upper)), fontsize=12
            )
    plt.title(dataset_name, fontsize=14)
    plt.ylabel('Inferred Count', fontsize=12)
    plt.xlabel('Manual Count', fontsize=12)
    #
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.DIR.snapshot, dataset_name+'.png'))
    # plt.show()


def save_figure(output_save, output_seg, mask_image, mask, gt, pd, epoch_result_dir, epoch_map_dir, image_name, ext):
    
    # save map
    Image.fromarray(output_save.astype(np.uint8)).save(
                    os.path.join(epoch_map_dir, image_name.replace(ext, '.png'))
                    )

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(mask_image.astype(np.uint8))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('ground truth=%4.2f'%(gt), fontsize=10)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(output_save.astype(np.uint8))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title('inferred count=%4.2f'%(pd), fontsize=10)
    plt.savefig(os.path.join(epoch_result_dir, image_name.replace(ext, '.png')), bbox_inches='tight', dpi = 600)
    plt.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(mask_image.astype(np.uint8))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow((output_seg * 255).astype(np.uint8))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    plt.savefig(os.path.join(epoch_result_dir, 'seg_'+image_name.replace(ext, '.png')), bbox_inches='tight', dpi = 600)
    plt.close()


def hist(pd, gt, nclass):
    def fast_hist(a, b, n):
        # a for gt
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n**2).reshape(n, n)
    return fast_hist(gt.flatten(), pd.flatten(), nclass)


def compute_iou(cm):
    # iou = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm))
    # miou = np.nanmean(iou)
    iou = np.zeros(cm.shape[0])
    num = np.diag(cm)
    den = cm.sum(1) + cm.sum(0) - np.diag(cm)
    idx = den > 0
    iou[idx] = num[idx] / den[idx]
    miou = np.mean(iou)
    return iou, miou


def save_checkpoint(state, snapshot_dir, filename='model_ckpt.pth.tar'):
    torch.save(state, '{}/{}'.format(snapshot_dir, filename))


def plot_learning_curves(net, dir_to_save):
    # plot learning curves
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(net.train_loss['epoch_loss'], label='train loss', color='tab:blue')
    ax1.legend(loc = 'upper right')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(net.val_loss['epoch_loss'], label='val mae', color='tab:orange')
    ax2.legend(loc = 'upper right')
    # ax2.set_ylim((0,50))
    fig.savefig(os.path.join(dir_to_save, 'learning_curves.png'), bbox_inches='tight', dpi = 300)
    plt.close()


def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    new_x = cv.resize(x, dsize=(w,h), interpolation=cv.INTER_CUBIC)

    return new_x

    