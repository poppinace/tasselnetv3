import os
import glob
import random

root = '../data/phenocryst_counting_dataset'
image_folder = 'images'
label_folder = 'labels'
trainval = 'train'
test = 'val'

trainval_path = os.path.join(root, trainval)
with open('train.txt', 'w') as f:
    for image_path in glob.glob(os.path.join(trainval_path, image_folder, '*.jpg')):
        im_path = image_path.replace(root, '')
        gt_path = im_path.replace(image_folder, label_folder).replace('.jpg', '.csv')
        f.write(im_path+'\t'+gt_path+'\n')

test_path = os.path.join(root, test)
with open('val.txt', 'w') as f:
    for image_path in glob.glob(os.path.join(test_path, image_folder, '*.jpg')):
        im_path = image_path.replace(root, '')
        gt_path = im_path.replace(image_folder, label_folder).replace('.jpg', '.csv')
        f.write(im_path+'\t'+gt_path+'\n')

