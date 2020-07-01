import os
import glob
import random

root = './data/sorghum_head_counting_dataset'
image_folder = 'original'
label_folder = 'labeled'
trainval = 'dataset1'
test = 'dataset2'

trainval_path = os.path.join(root, image_folder)
with open('dataset1.txt', 'w') as f:
    for image_path in glob.glob(os.path.join(trainval_path, trainval, '*.jpg')):
        im_path = image_path.replace(root, '')
        gt_path = im_path.replace(image_folder, label_folder).replace('.jpg', '-hand.png')
        f.write(im_path+'\t'+gt_path+'\n')

test_path = os.path.join(root, image_folder)
with open('dataset2.txt', 'w') as f:
    for image_path in glob.glob(os.path.join(test_path, test, '*.tif')):
        im_path = image_path.replace(root, '')
        gt_path = im_path.replace(image_folder, label_folder).replace('.tif', '-hand.png')
        f.write(im_path+'\t'+gt_path+'\n')