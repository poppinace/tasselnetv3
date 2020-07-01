# -*- coding: utf-8 -*-
"""
Created on Sat April 12 2020
@author: Hao Lu
"""

import os
import argparse
from time import time

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from skimage import measure
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from scipy.ndimage import morphology

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.backends.cudnn as cudnn

from config import cfg
from models import CountingModels
import hlloss
from hldataset import dataloader_list, RandomCrop, RandomFlip, ZeroPadding, Normalize, ToTensor
from utils import *

# prevent dataloader deadlock, uncomment if deadlock occurs
# cv.setNumThreads(0)
cudnn.enabled = True


def read_image(x):
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:  # grayscale
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr


def train(net, train_loader, criterion, optimizer, epoch, cfg):
    # switch to 'train' mode
    net.train()

    # uncomment the following line if the training images don't have the same size
    cudnn.benchmark = True

    if cfg.TRAIN.batch_size == 1:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    running_loss = 0.0
    avg_frame_rate = 0.0
    bk_sz = cfg.MODEL.block_size
    ostride = cfg.MODEL.output_stride
    target_filter = torch.cuda.FloatTensor(1, 1, bk_sz, bk_sz).fill_(1)
    target_filter_ostride = torch.cuda.FloatTensor(1, 1, ostride, ostride).fill_(1)
    for i, sample in enumerate(train_loader):
        torch.cuda.synchronize()
        start = time()

        inputs, targets, masks = sample['image'], sample['target'], sample['mask']
        inputs, targets, masks = inputs.cuda(), targets.cuda(), masks.cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(inputs, is_normalize=False)
        # generate targets
        targets = net.module.generator(
            targets=targets, 
            block_size=bk_sz,
            output_stride=ostride, 
            target_filter=target_filter,
            target_filter_os=target_filter_ostride,
            start_log=cfg.MODEL.start_log,   
            step_log=cfg.MODEL.step_log, 
            num_class=cfg.MODEL.num_class,
            gt_downsampling_rate=cfg.DATASET.gt_downsampling_rate   
            )
        # compute loss
        loss = criterion(outputs, targets, masks)
        # backward + optimize
        loss.backward()
        optimizer.step()
        # collect and print statistics
        running_loss += loss.item()

        torch.cuda.synchronize()
        end = time()

        running_frame_rate = cfg.TRAIN.batch_size * float(1 / (end - start))
        avg_frame_rate = (avg_frame_rate*i + running_frame_rate)/(i+1)
        if i % cfg.TRAIN.disp_iter == cfg.TRAIN.disp_iter-1:
            print('epoch: %d, train: %d/%d, '
                  'loss: %.5f, frame: %.2fHz/%.2fHz' % (
                      epoch,
                      i+1,
                      len(train_loader),
                      running_loss / (i+1),
                      running_frame_rate,
                      avg_frame_rate
                  ))
    net.train_loss['epoch_loss'].append(running_loss / (i+1))


def validate(net, valset, val_loader, criterion, epoch, cfg):
    # switch to 'eval' mode
    net.eval()
    cudnn.benchmark = False
    
    image_list = valset.image_list
    
    if cfg.VAL.visualization:
        epoch_result_dir = os.path.join(cfg.DIR.result, str(epoch))
        if not os.path.exists(epoch_result_dir):
            os.makedirs(epoch_result_dir)
        epoch_map_dir = os.path.join(epoch_result_dir, 'map')
        if not os.path.exists(epoch_map_dir):
            os.makedirs(epoch_map_dir)
        cmap = plt.cm.get_cmap('jet')

    pdcounts = []
    gtcounts = []
    pdcounts_fg = []
    gtcounts_fg = []
    pdcounts_bg = []
    gtcounts_bg = []
    cm = np.zeros((2, 2), dtype=int)
    bk_sz = cfg.MODEL.block_size
    ostride = cfg.MODEL.output_stride # os conflicts with module 'os'
    with torch.no_grad():
        avg_frame_rate = 0.0
        for i, sample in enumerate(val_loader):
            torch.cuda.synchronize()
            start = time()

            image, mask, gtcount = sample['image'], sample['mask'], sample['gtcount']
            # inference
            if cfg.VAL.visualization:
                output_save = net(image.cuda(), is_normalize=cfg.VAL.normalization_before_visualization)
                if cfg.VAL.normalization_before_visualization:
                    output = output_save[0]
                    output_seg = output_save[1]
                    output_save = output
                else:
                    output = net.module.normalizer(output_save, image.size()[2], image.size()[3], bk_sz, ostride)
                    output_save = output_save.squeeze().cpu().numpy()
            else:
                output = net(image.cuda(), is_normalize=True)
                output_count, output_seg = output[0], output[1]
                output = output_count

            # postprocessing
            output = np.clip(output, 0, None)
            output_hr = net.module.visualizer(output, image, bk_sz, ostride)

            mask = mask.squeeze().numpy()
            mask_idx = mask < 2 # Ignore every class index larger than the number of classes
            cm += hist(output_seg[mask_idx], mask[mask_idx], 2)
            _, miou = compute_iou(cm)
            
            # ----------------------------------------------------------------------------
            # # LATE FUSION
            # # mask dilation 
            # output_seg = morphology.distance_transform_edt(output_seg==0) <= int(bk_sz/4)
            # # masking prediction
            # output_hr = output_hr * output_seg
            # ----------------------------------------------------------------------------
            
            # compute fg and bg errors
            output_fg, output_bg = output_hr * mask, output_hr * (1 - mask)
            pdcount = output_hr.sum() / cfg.DATASET.scaling
            gtcount = float(gtcount.numpy())
            pdcount_fg = output_fg.sum() / cfg.DATASET.scaling
            gtcount_fg = gtcount
            pdcount_bg = output_bg.sum() / cfg.DATASET.scaling
            gtcount_bg = 0

            pdcounts.append(pdcount)
            gtcounts.append(gtcount)
            pdcounts_fg.append(pdcount_fg)
            gtcounts_fg.append(gtcount_fg)
            pdcounts_bg.append(pdcount_bg)
            gtcounts_bg.append(gtcount_bg)

            if cfg.VAL.visualization:
                _, image_name = os.path.split(image_list[i])
                _, ext = os.path.splitext(image_name)
                output_save = np.clip(output_save, 0, None)
                output_save = net.module.visualizer(output_save, image, bk_sz, ostride)
                output_save = output_save / (output_save.max() + 1e-12)
                output_save = cmap(output_save) * 255.
                # image composition
                image = read_image(os.path.join(valset.data_dir, valset.data_list[i][0]))
                nh, nw = output_save.shape[:2]
                image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
                output_save = 0.5 * image + 0.5 * output_save[:, :, 0:3]
                mask_image = 0.5 * image + 0.5 * np.expand_dims(mask, axis=2) * 255.

                save_figure(output_save, output_seg, mask_image, mask, gtcount, pdcount, epoch_result_dir, epoch_map_dir, image_name, ext)

            # compute mae and mse
            mae = compute_mae(pdcounts, gtcounts)
            mse = compute_mse(pdcounts, gtcounts)
            rmae, rmse = compute_relerr(pdcounts, gtcounts)

            mae_fg = compute_mae(pdcounts_fg, gtcounts_fg)
            mse_fg = compute_mse(pdcounts_fg, gtcounts_fg)
            rmae_fg, rmse_fg = compute_relerr(pdcounts_fg, gtcounts_fg)

            mae_bg = compute_mae(pdcounts_bg, gtcounts_bg)
            mse_bg = compute_mse(pdcounts_bg, gtcounts_bg)

            torch.cuda.synchronize()
            end = time()
            
            running_frame_rate = 1 * float(1 / (end - start))
            avg_frame_rate = (avg_frame_rate*i + running_frame_rate)/(i+1)
            if i % cfg.VAL.disp_iter == cfg.VAL.disp_iter - 1:
                print(
                    'epoch: {0}, test: {1}/{2}, pd: {3:.2f}, gt:{4:.2f}, mae: {5:.2f}, mse: {6:.2f}, rmae: {7:.2f}%, rmse: {8:.2f}%, miou: {9:.2f}, frame: {10:.2f}Hz/{11:.2f}Hz'
                    .format(epoch, i+1, len(val_loader), pdcount, gtcount, mae, mse, rmae, rmse, miou, running_frame_rate, avg_frame_rate)
                    )
            start = time()
    iou, miou = compute_iou(cm)
    r2 = rsquared(pdcounts, gtcounts)
    print('epoch: {0}, mae: {1:.2f}, mse: {2:.2f}, rmae: {3:.2f}%, rmse: {4:.2f}%, r2: {5:.4f}, iou_fg: {6:.2f}, iou_bg: {7:.2f}, miou: {8:.2f}'.format(epoch, mae, mse, rmae, rmse, r2, iou[1], iou[0], miou))
    print('epoch: {0}, mae_fg: {1:.2f}, mse_fg: {2:.2f}, rmae_fg: {3:.2f}%, rmse_fg: {4:.2f}%'.format(epoch, mae_fg, mse_fg, rmae_fg, rmse_fg))
    print('epoch: {0}, mae_bg: {1:.2f}, mse_bg: {2:.2f}%'.format(epoch, mae_bg, mse_bg))
    # write to files        
    with open(os.path.join(cfg.DIR.snapshot, cfg.DIR.exp+'.txt'), 'a') as f:
        print('epoch: {0}, mae: {1:.2f}, mse: {2:.2f}, rmae: {3:.2f}%, rmse: {4:.2f}%, r2: {5:.4f}, iou_fg: {6:.2f}, iou_bg: {7:.2f}, miou: {8:.2f}'.format(epoch, mae, mse, rmae, rmse, r2, iou[1], iou[0], miou), file=f)
        print('epoch: {0}, mae_fg: {1:.2f}, mse_fg: {2:.2f}, rmae_fg: {3:.2f}%, rmse_fg: {4:.2f}%'.format(epoch, mae_fg, mse_fg, rmae_fg, rmse_fg), file=f)
        print('epoch: {0}, mae_bg: {1:.2f}, mse_bg: {2:.2f}\n'.format(epoch, mae_bg, mse_bg), file=f)
    with open(os.path.join(cfg.DIR.snapshot, 'counts.txt'), 'a') as f:
        for pd, gt in zip(pdcounts, gtcounts):
            print('{0} {1}'.format(pd, gt), file=f)
    # save stats
    net.val_loss['epoch_loss'].append(mae)
    net.measure['mae'].append(mae)
    net.measure['mse'].append(mse)
    net.measure['rmae'].append(rmae)
    net.measure['rmse'].append(rmse)
    net.measure['r2'].append(r2)
    net.measure['miou'].append(miou)

    return pdcounts, gtcounts


def generate_mask(net, dataset, dataloader, criterion, epoch, cfg):
    # switch to 'eval' mode
    net.eval()
    cudnn.benchmark = False
    
    image_list = dataset.image_list

    bk_sz = cfg.MODEL.block_size
    ostride = cfg.MODEL.output_stride # os is contradict with module 'os'
    target_filter_ostride = torch.cuda.FloatTensor(1, 1, ostride, ostride).fill_(1)
    with torch.no_grad():
        avg_frame_rate = 0.0
        for i, sample in enumerate(dataloader):
            torch.cuda.synchronize()
            start = time()

            _, image_name = os.path.split(image_list[i])
            _, ext = os.path.splitext(image_name)

            image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
            image, target = image.cuda(), target.cuda()
            pred = net(image, is_normalize=cfg.VAL.normalization_before_visualization)
            pred = np.clip(pred, 0, None)
            pdcount = pred.sum() / cfg.DATASET.scaling
            pred = net.module.visualizer(pred, image, bk_sz, ostride)

            # find and label mask
            mask = pred > pred.mean() 
            mask_label = measure.label(mask)
            label_all, inverse = np.unique(mask_label, return_inverse=True)

            region_count = []
            for l in label_all:
                if l == 0:
                    continue
                region_count.append(round(pred[mask_label == l].sum(), 2))


            # filter tiny mask
            regions = measure.regionprops(mask_label)
            bbs = [props.bbox for props in regions]
            # area = [props.area for props in regions]
            # area_idx = np.where(np.array(area) == 64)
            # label_all[area_idx[0] + 1] = 0
            count_idx = np.where(np.array(region_count) < 0.1)
            label_all[count_idx[0] + 1] = 0

            target = F.conv2d(target, target_filter_ostride, stride=ostride)
            target = net.module.visualizer(target.squeeze().cpu().numpy(), image, bk_sz, ostride)
            target = target > target.mean()
            # dotimage = dataset.dotimages[image_list[i]]
            # # Euclidean distance transform
            # dotimage = morphology.distance_transform_edt(dotimage==0) <= int(cfg.DATASET.sigma * cfg.DATASET.resize_ratio)
            # find vaild regions
            label_valid = np.unique(mask_label[target])
            # find invalid regions
            label_mask = np.array([l if l not in label_valid else 0 for l in label_all])
            mask = label_mask[inverse].reshape(mask.shape)
            mask = mask > 0
            mask = np.array(mask).astype(np.uint8)
            h, w = mask.shape

            # visualize connected regions
            dotimage_label = measure.label(target)
            cmap = plt.cm.get_cmap('nipy_spectral')
            mask_composition = 0.5 * cmap(dotimage_label)[:, :, 0:3] + 0.5 * np.expand_dims(mask, axis=2)
            image = read_image(dataset.data_dir+dataset.data_list[i][0])
            nh, nw = image.shape[:2]
            ratio = nh / h
            mask_up = cv2.resize(mask, (nw, nh), interpolation = cv2.INTER_NEAREST)
            image_composition = 0.5 * image + 0.5 * np.expand_dims(mask_up, axis=2) * 255.

            # Image.fromarray((cmap(dotimage_label) * 255).astype(np.uint8)).save(
            #         os.path.join(cfg.DIR.mask, 'dot_label_'+image_name.replace(ext, '.png'))
            #         )
            # Image.fromarray((cmap(mask_label) * 255).astype(np.uint8)).save(
            #         os.path.join(cfg.DIR.mask, 'mask_label_'+image_name.replace(ext, '.png'))
            #         )
            # Image.fromarray((mask_composition * 255.).astype(np.uint8)).save(
            #         os.path.join(cfg.DIR.mask, 'composition_'+image_name.replace(ext, '.png'))
            #         )
            # Image.fromarray((mask * 255.).astype(np.uint8)).save(
            #         os.path.join(cfg.DIR.mask, image_name.replace(ext, '.png'))
            #         )
            # Image.fromarray(image_composition.astype(np.uint8)).save(
            #         os.path.join(cfg.DIR.mask, image_name.replace(ext, '.JPG'))
            #         )

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(image_composition.astype(np.uint8))
            for l in label_mask:
                if l == 0:
                    continue
                y, x = int(bbs[l-1][0] * ratio), int(bbs[l-1][1] * ratio)
                plt.text(x, y, str(region_count[l-1]), color='y', fontsize=3)
            plt.title('pd=%4.2f, gt=%4.2f'%(pdcount, gtcount), fontsize=8)
            plt.savefig(os.path.join(cfg.DIR.mask, 'image_'+image_name.replace(ext, '.png')), bbox_inches='tight', dpi = 600)
            plt.close()

            torch.cuda.synchronize()
            end = time()
            
            running_frame_rate = 1 * float(1 / (end - start))
            avg_frame_rate = (avg_frame_rate*i + running_frame_rate)/(i+1)
            if i % cfg.VAL.disp_iter == cfg.VAL.disp_iter - 1:
                print(
                    'epoch: {0}, generate mask: {1}/{2}, frame: {3:.2f}Hz/{4:.2f}Hz'
                    .format(epoch, i+1, len(dataloader), running_frame_rate, avg_frame_rate)
                )
            start = time()


def main(cfg, gpus):

    # instantiate model
    net = CountingModels(
        encoder=cfg.MODEL.encoder,
        decoder=cfg.MODEL.decoder,
        counter=cfg.MODEL.counter,
        normalizer=cfg.MODEL.normalizer,
        generator=cfg.MODEL.generator,
        visualizer=cfg.MODEL.visualizer,
        base_size=cfg.DATASET.crop_size[0],
        block_size=cfg.MODEL.block_size,
        output_stride=cfg.MODEL.output_stride,
        counter_dim=cfg.MODEL.counter_dim,        
        num_class=cfg.MODEL.num_class,
        step_log=cfg.MODEL.step_log, 
        start_log=cfg.MODEL.start_log,
        pretrain=cfg.MODEL.pretrain,
        fix_bn=cfg.MODEL.fix_bn
        )

    net = nn.DataParallel(net)
    net.cuda()

    # with torch.no_grad():
    #     from models.modelsummary import get_model_summary
    #     dump_x = torch.randn(1, 3, 320, 320).cuda()
    #     print(get_model_summary(net, dump_x))

    R = [512, 768, 1024, 1280, 1536, 1792, 2048]
    with torch.no_grad():
        net.eval()
        for r in R:
            frame_rate = np.zeros((100, 1))
            for i in range(100):
                x = torch.randn(1, 3, r, r).cuda()
                torch.cuda.synchronize()
                start = time()
                y = net(x, is_normalize=True)
                torch.cuda.synchronize()
                end = time()
                running_frame_rate = 1 * float(1 / (end - start))
                frame_rate[i] = running_frame_rate
            print(np.mean(frame_rate))
    
    # filter parameters
    pretrained_params = []
    learning_params = []
    for p in net.named_parameters():
        if 'encoder' in p[0]:
            pretrained_params.append(p[1])
        else:
            learning_params.append(p[1])

    # define loss function and optimizer
    criterion = hlloss.__dict__[cfg.TRAIN.loss](reduction=cfg.TRAIN.loss_reduction)

    if cfg.TRAIN.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            [
                {'params': learning_params},
                {'params': pretrained_params, 'lr': cfg.TRAIN.lr_encoder / cfg.TRAIN.encoder_multiplier},
            ],
            lr=cfg.TRAIN.lr_encoder,
            momentum=cfg.TRAIN.momentum,
            weight_decay=cfg.TRAIN.weight_decay
        )
    elif cfg.TRAIN.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            [
                {'params': learning_params},
                {'params': pretrained_params, 'lr': cfg.TRAIN.lr_encoder / cfg.TRAIN.encoder_multiplier},
            ],
            lr=cfg.TRAIN.lr_encoder
        )
    else:
        raise NotImplementedError

    # restore parameters
    start_epoch = 0
    net.train_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.val_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.measure = {
        'mae': [],
        'mse': [],
        'rmae': [],
        'rmse': [],
        'r2': [],
        'miou': []
    }
    restore_dir = cfg.VAL.checkpoint if cfg.VAL.evaluate_only else cfg.TRAIN.checkpoint
    if restore_dir is not None:
        if os.path.isfile(restore_dir):
            checkpoint = torch.load(restore_dir)
            net.load_state_dict(checkpoint['state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'train_loss' in checkpoint:
                net.train_loss = checkpoint['train_loss']
            if 'val_loss' in checkpoint:
                net.val_loss = checkpoint['val_loss']
            if 'measure' in checkpoint:
                net.measure = checkpoint['measure']
            print("==> load checkpoint '{}' (epoch {})"
                  .format(restore_dir, start_epoch))
        else:
            print("==> no checkpoint found at '{}'".format(restore_dir))

    # instantiate dataset
    dataset = dataloader_list[cfg.DATASET.name]

    # define transform
    transform_train = [
        RandomCrop(cfg.DATASET.random_crop, cfg.DATASET.crop_size),
        RandomFlip(cfg.DATASET.random_flip),
        Normalize(
            cfg.DATASET.image_scale, 
            np.array(cfg.DATASET.image_mean).reshape((1, 1, 3)), 
            np.array(cfg.DATASET.image_std).reshape((1, 1, 3)) 
        ),
        ToTensor(),
        ZeroPadding(cfg.DATASET.padding_constant)
    ]
    transform_val = [
        Normalize(
            cfg.DATASET.image_scale, 
            np.array(cfg.DATASET.image_mean).reshape((1, 1, 3)), 
            np.array(cfg.DATASET.image_std).reshape((1, 1, 3)) 
        ),
        ToTensor(),
        ZeroPadding(cfg.DATASET.padding_constant)
    ]
    composed_transform_train = transforms.Compose(transform_train)
    composed_transform_val = transforms.Compose(transform_val)

    # define dataset loader
    trainset = dataset(
        data_dir=cfg.DIR.dataset,
        mask_dir=cfg.DIR.mask,
        data_list=cfg.DATASET.list_train,
        ratio=cfg.DATASET.resize_ratio,
        sigma=cfg.DATASET.sigma,
        scaling=cfg.DATASET.scaling,
        preload=cfg.DATASET.preload,
        transform=composed_transform_train
    )
    train_loader = DataLoader(
        trainset,
        batch_size=cfg.TRAIN.batch_size,
        shuffle=True,
        num_workers=cfg.TRAIN.num_workers,
        pin_memory=True,
        drop_last=True
    )
    valset = dataset(
        data_dir=cfg.DIR.dataset,
        mask_dir=cfg.DIR.mask,
        data_list=cfg.DATASET.list_val,
        ratio=cfg.DATASET.resize_ratio,
        sigma=cfg.DATASET.sigma,
        scaling=cfg.DATASET.scaling,
        preload=cfg.DATASET.preload,
        transform=composed_transform_val
    )
    val_loader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.TRAIN.num_workers,
        pin_memory=True
    )

    if cfg.TRAIN.generate_mask:
        print('generate training masks...')
        if not os.path.exists(cfg.DIR.mask):
            os.makedirs(cfg.DIR.mask)
        # generate training masks
        trainset.transform = composed_transform_val
        mask_loader = DataLoader(
            trainset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.TRAIN.num_workers,
            pin_memory=True
        )
        generate_mask(net, trainset, mask_loader, criterion, start_epoch, cfg)
        return

    print('alchemy start...')
    if cfg.VAL.evaluate_only:
        _, _ = validate(net, valset, val_loader, criterion, start_epoch, cfg)
        return
    
    best_mae, best_mse, best_rmae, best_rmse, best_r2, best_miou = 1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0, 0
    resume_epoch = -1 if start_epoch == 0 else start_epoch
    scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.milestones, gamma=0.1, last_epoch=resume_epoch)
    for epoch in range(start_epoch, cfg.TRAIN.num_epochs):
        # train
        train(net, train_loader, criterion, optimizer, epoch+1, cfg)
        if epoch % cfg.VAL.val_epoch == cfg.VAL.val_epoch - 1:
            # val
            pd, gt = validate(net, valset, val_loader, criterion, epoch+1, cfg)
            # save_checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch+1,
                'train_loss': net.train_loss,
                'val_loss': net.val_loss,
                'measure': net.measure
            }
            save_checkpoint(state, cfg.DIR.snapshot, filename='model_ckpt.pth.tar')
            if net.measure['mae'][-1] <= best_mae:
                save_checkpoint(state, cfg.DIR.snapshot, filename='model_best.pth.tar')
                np.save(cfg.DIR.snapshot+'/pd.npy', pd)
                np.save(cfg.DIR.snapshot+'/gt.npy', gt)
                plot_r2(pd, gt, cfg)
                best_mae = net.measure['mae'][-1]
                best_mse = net.measure['mse'][-1]
                best_rmae = net.measure['rmae'][-1]
                best_rmse = net.measure['rmse'][-1]
                best_r2 = net.measure['r2'][-1]
                best_miou = net.measure['miou'][-1]
            print(cfg.DIR.exp+' epoch {} finished!'.format(epoch+1))
            print('best mae: {0:.2f}, best mse: {1:.2f}, best_rmae: {2:.2f}, best_rmse: {3:.2f}, best_r2: {4:.4f}, best_miou: {5:.2f}'
                  .format(best_mae, best_mse, best_rmae, best_rmse, best_r2, best_miou))
            plot_learning_curves(net, cfg.DIR.snapshot)
        scheduler.step()
        
    print('Experiments with '+cfg.DIR.exp+' done!')
    with open(os.path.join(cfg.DIR.snapshot, cfg.DIR.exp+'.txt'), 'a') as f:
        print(
            'best mae: {0:.2f}, best mse: {1:.2f}, best_rmae: {2:.2f}, best_rmse: {3:.2f}, best_r2: {4:.4f}, best_miou: {5:.2f}'
            .format(best_mae, best_mse, best_rmae, best_rmse, best_r2, best_miou),
            file=f
        )
        print(
            'overall best mae: {0:.2f}, overall best mse: {1:.2f}, overall best_rmae: {2:.2f}, overall best_rmse: {3:.2f}, overall best_r2: {4:.4f}'
            .format(min(net.measure['mae']), min(net.measure['mse']), min(net.measure['rmae']), min(net.measure['rmse']), max(net.measure['r2'])),
            file=f
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SMART Object Counting in PyTorch"
    )
    parser.add_argument(
        "--cfg",
        default="config/mtc-uav-seg2count.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # seeding for reproducbility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)
    np.random.seed(cfg.TRAIN.seed)

    cfg.DIR.snapshot = os.path.join(cfg.DIR.snapshot, cfg.DATASET.name.lower(), cfg.DIR.exp)
    if not os.path.exists(cfg.DIR.snapshot):
        os.makedirs(cfg.DIR.snapshot)

    cfg.DIR.result = os.path.join(cfg.DIR.result, cfg.DATASET.name.lower(), cfg.DIR.exp)
    if not os.path.exists(cfg.DIR.result):
        os.makedirs(cfg.DIR.result)

    cfg.DIR.mask = os.path.join(cfg.DIR.result, 'masks')

    cfg.TRAIN.checkpoint = os.path.join(cfg.DIR.snapshot, cfg.TRAIN.checkpoint)
    cfg.VAL.checkpoint = os.path.join(cfg.DIR.snapshot, cfg.VAL.checkpoint)

    print("Loaded configuration file {}".format(args.cfg))
    print("Running with config:\n{}".format(cfg))
    with open(os.path.join(cfg.DIR.snapshot, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    main(cfg, gpus)