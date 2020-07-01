
import os
import glob
import random
import numpy as np

root = '../data/maize_tassels_counting_uav_dataset'
image_folder = 'images'
label_folder = 'labels'

image_path = os.path.join(root, image_folder)
image_list = glob.glob(os.path.join(image_path, '*.JPG'))

np.random.seed(2020)
rd = np.random.permutation(306)
train_idx = rd[0:200]
val_idx = rd[200:]
train_list = [image_list[i] for i in train_idx]
val_list = [image_list[i] for i in val_idx]

with open(os.path.join(root, 'train.txt'), 'w') as f:
    for image_path in train_list:
        im_path = image_path.replace(root, '')
        gt_path = im_path.replace(image_folder, label_folder).replace('.JPG', '.csv')
        f.write(im_path+'\t'+gt_path+'\n')

with open(os.path.join(root, 'val.txt'), 'w') as f:
    for image_path in val_list:
        im_path = image_path.replace(root, '')
        gt_path = im_path.replace(image_folder, label_folder).replace('.JPG', '.csv')
        f.write(im_path+'\t'+gt_path+'\n')