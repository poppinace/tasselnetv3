# TasselNetV3
by [Hao Lu](https://sites.google.com/site/poppinace/), Liang Liu


## Installation
The code has been tested on Python 3.7.4 and PyTorch 1.2.0. Please follow the
official instructions to configure your environment. See other required packages
in `requirements.txt`.

## Prepare Your Data
**Maize Tassels Counting**
* Download the Maize Tassels Counting (MTC) dataset from: [BaiduYun (1.64
  GB)](https://pan.baidu.com/s/1F7eiW3TDsQ-EFg8fQjjvzw) (code: m8rj) or [Google Drive (1.8
  GB)](https://drive.google.com/open?id=1IyGpYMS_6eClco2zpHKzW5QDUuZqfVFJ)
* Unzip the dataset and move it into the `./data` folder, the path structure should look like this:
````
$./data/maize_tassels_counting_dataset
├──── trainval
│    ├──── images
│    └──── labels
├──── test
│    ├──── images
│    └──── labels
├──── train.txt
├──── test.txt
````

## Training
Run the following command to train TasselNetv2+ on the MTC dataset:

    python --cfg config/mtc-tasselnetv2plus.yaml

* Setting `VAL.evaluate_only=False` and `VAL.visualization=False`
* Use `CUDA_VISIBLE_DEVICES` trick if you have multiple GPUs

### Tips and Tricks
> If you find some useful tricks and tips, please share it here.
- (**Hao Lu**) Do not fix bn when training with pretrained models (batch_size=16 tested)
- (**Hao Lu**) Scale the ground truth by x10 for density-map-based methods when L2 Loss is used (reduction='mean')

## Inference
Once the training is finished, run the same command above with `VAL.evaluate_only=True` for inference.    
* Setting `VAL.visualization=True` to output visualizations. Visualizations are saved in the path `./results/<dataset>/<exp>/<epoch>`.


## Benchmark Results

### *Plant Counting*
#### Maize Tassels Counting
| Method        | Venue, Year           | Pretrained    | #Param.   | MAE   | MSE    | rMAE  | R<sup>2</sup> | Model             |
| :--:          | :--:                  | :--:          | :--:      | :--:  | :--:   | :--:  | :--:          | :--:              |
| CSRNet        | CVPR 2018             | VGG16         | 16.3M     | 9.43  | 14.43  | 100.65| 0.7573        | [One Drive (116MB)](https://1drv.ms/u/s!ArQcMEHVq8YvjzpBqep4UEg8nI7c?e=kyuBvq) | 
| TasselNetv2   | PLME 2019             | No            | 525K      | 5.42  | 9.21   | 31.94 | 0.8923        | [Baidu Yun (2MB)](https://pan.baidu.com/s/1_FiACr0fEiBjsPrpJ44yLg) (code: hrhi) |
| TasselNetv2+  | TBD                   | No            | 262K      | 5.41  | 9.31   | 37.65 | 0.8937        | [Baidu Yun (2MB)](https://pan.baidu.com/s/1pIj-elQ5YnFRkT8GuxYw9Q) (code: hbnx) |
| BCNet-BN      | TCSVT 2019            | VGG16         | 14.8M     | 5.11  | 9.58   | 27.84 | 0.8749        | [Baidu Yun (105MB)](https://pan.baidu.com/s/1xZIMbpn9i58jDZW4tivMFg) (code: mnys) |


#### Maize Tassels Counting (UAV)
| Method        | Venue, Year           | Resolution  | Pretrained    | #Param.   | MAE   | MSE    | rMAE  | R<sup>2</sup> | Model |
| :--:          | :--:                  | :--:        | :--:          | :--:      | :--:  | :--:   | :--:  | :--:          | :--:  |
| TasselNetv2+  | TBD                   | 1/8         | No            | 262K      | 27.08 | 38.38  | 14.61  | 0.8958       | - |
| TasselNetv2+  | TBD                   | 1/4         | No            | 262K      | 16.43 | 25.79  | 9.67  | 0.9515        | [Baidu Yun (2MB)](https://pan.baidu.com/s/1QJ7WRZqQKT_hUmmKZMeCAw) (code: 68dn) |
| CSRNet        | CVPR 2018             | 1/4         | VGG16         | 16.3M     | 14.38 | 20.52  | 9.56  | 0.9704        | [One Drive (116MB)](https://1drv.ms/u/s!ArQcMEHVq8YvjzZIK1Aqy06LbsSo?e=5MgZp2)
| BCNet-BN      | TCSVT 2019            | 1/4         | VGG16         | 14.8M     | 14.37 | 21.37  | 8.75  | 0.9659        | [Baidu Yun (105MB)](https://pan.baidu.com/s/1cGztl4x3Ey0ARMGPtwZtdw) (code: t81t) |
