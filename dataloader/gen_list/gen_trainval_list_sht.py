
import os
import glob

root = '/media/hao/DATA/ShanghaiTech'

dataset = 'part_A_final'
image_list = os.path.join(root, dataset, 'train_data', 'images', '*.jpg')
with open('part_A_train.txt', 'w') as f:  
    for image_path in glob.glob(image_list):
        img_path = image_path[28:]
        gt_path = img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG', 'GT_IMG')
        f.write(img_path+'\t'+gt_path+'\n')

image_list = os.path.join(root, dataset, 'test_data', 'images', '*.jpg')
with open('part_A_test.txt', 'w') as f:  
    for image_path in glob.glob(image_list):
        img_path = image_path[28:]
        gt_path = img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG', 'GT_IMG')
        f.write(img_path+'\t'+gt_path+'\n')

dataset = 'part_B_final'
image_list = os.path.join(root, dataset, 'train_data', 'images', '*.jpg')
with open('part_B_train.txt', 'w') as f:  
    for image_path in glob.glob(image_list):
        img_path = image_path[28:]
        gt_path = img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG', 'GT_IMG')
        f.write(img_path+'\t'+gt_path+'\n')

image_list = os.path.join(root, dataset, 'test_data', 'images', '*.jpg')
with open('part_B_test.txt', 'w') as f:  
    for image_path in glob.glob(image_list):
        img_path = image_path[28:]
        gt_path = img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG', 'GT_IMG')
        f.write(img_path+'\t'+gt_path+'\n')