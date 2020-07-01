
import os
import glob
import random
import numpy as np

root = './data/maize_kernels_counting_dataset'
image_folder = 'long500'
label_folder = 'Pts'

image_path = os.path.join(root, image_folder)
image_list = glob.glob(os.path.join(image_path, '*.jpg'))

np.random.seed(2020)
rd = np.random.permutation(1000)
train_idx = rd[0:500]
val_idx = rd[500:]
train_list = [image_list[i] for i in train_idx]
val_list = [image_list[i] for i in val_idx]

with open('train.txt', 'w') as f:
    for image_path in train_list:
        im_path = image_path.replace(root, '')
        gt_path = im_path.replace(image_folder, label_folder).replace('.jpg', '.txt')
        f.write(im_path+'\t'+gt_path+'\n')

with open('val.txt', 'w') as f:
    for image_path in val_list:
        im_path = image_path.replace(root, '')
        gt_path = im_path.replace(image_folder, label_folder).replace('.jpg', '.txt')
        f.write(im_path+'\t'+gt_path+'\n')