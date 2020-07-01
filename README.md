# SMART Object Counting Framework in Pytorch
by [Hao Lu](https://sites.google.com/site/poppinace/), Liang Liu


## Installation
The code has been tested on Python 3.7.4 and PyTorch 1.2.0. Please follow the
official instructions to configure your environment. See other required packages
in `requirements.txt` (pending).

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

**ShanghaiTech Crowd Counting**
* Download the ShanghaiTech dataset from: [BaiduYun (1.64
  GB)](https://pan.baidu.com/s/16lHrFG7aWxr8UKJHk6oh4A) (code: sir7)

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

### *Crowd Counting*
#### ShanghaiTech Part A
| Method        | Venue, Year           | Pretrained    | #Param.   | MAE   | MSE    | rMAE  | R<sup>2</sup> | Model             |
| :--:          | :--:                  | :--:          | :--:      | :--:  | :--:   | :--:  | :--:          | :--:              |
| TasselNetv2   | PLME 2019             | No            | 525K      | 89.75 | 151.14 | 23.74 | 0.8189        | [Baidu Yun (10MB)](https://pan.baidu.com/s/1_Vk_GkVapNwtPShj-L7Ikg) (code: f8iz) |
| TasselNetv2+  | TBD                   | No            | 262K      | 91.84 | 155.48 | 25.51 | 0.8076        | [Baidu Yun (2MB)](https://pan.baidu.com/s/1Zzah1E7RYlBBUAF3zk_0rw) (code: aeaq) |
| CSRNet-BN     | CVPR 2018             | VGG16         | 16.3M     | 63.80 | 102.01 | 16.09 | 0.9169        | [One Drive (116MB)](https://1drv.ms/u/s!ArQcMEHVq8YvjzgJLWXdnihEY9Ga?e=sv1MSL) |
| BCNet         | TCSVT 2019            | VGG16         | 14.8M     | 64.96 | 106.25 | 16.31 | 0.9108        | |
| BCNet-BN      | TCSVT 2019            | VGG16         | 14.8M     | 61.98 | 103.62 | 13.96 | 0.9149        | [Baidu Yun (105MB)](https://pan.baidu.com/s/1FGcIbF0aniNCBV6rldWNPA) (code: e8jp) |


#### ShanghaiTech Part B
| Method        | Venue, Year           | Pretrained    | #Param.   | MAE   | MSE    | rMAE  | R<sup>2</sup> | Model             |
| :--:          | :--:                  | :--:          | :--:      | :--:  | :--:   | :--:  | :--:          | :--:              |
| TasselNetv2   | PLME 2019             | No            | 525K      | 16.13 | 26.63  | 13.21 | 0.9506        | [Baidu Yun (10MB)](https://pan.baidu.com/s/1HlWCjl9Hc_U65S0iEH95DA) (code: matr) |
| TasselNetv2+  | TBD                   | No            | 262K      | 14.37 | 26.07  | 11.41 | 0.9355        | [Baidu Yun (2MB)](https://pan.baidu.com/s/1Mx9doqRqtzvcvcTvxAuUJw) (code: s2pn) |
| CSRNet-BN     | CVPR 2018             | VGG16         | 16.3M     | 7.96  | 13.97  | 7.06  | 0.9797        | [One Drive (116MB)](https://1drv.ms/u/s!ArQcMEHVq8YvjzdZ6gE3I2ZMo7zO?e=1lzuLs) |
| BCNet-BN      | TCSVT 2019            | VGG16         | 14.8M     | 8.61  | 15.88  | 7.91  | 0.9763        | [Baidu Yun (105MB)](https://pan.baidu.com/s/1U1WU7IxJwUsF9OfG0IFRIA) (code: vaer)|


### *Geologic Counting*
#### Phenocryst Counting
| Method        | Venue, Year           | Pretrained    | #Param.   | MAE   | MSE    | rMAE  | R<sup>2</sup> |
| :--:          | :--:                  | :--:          | :--:      | :--:  | :--:   | :--:  | :--:          |
| Gaussian Kernel |
| MCNN          | CVPR 2016             | No            | 134K      | 70.15 | 108.20 | 67.62 | 0.7083        |
| TasselNetv2   | PLME 2019             | No            | 525K      | 62.62 | 98.75  | 37.57 | 0.7544        |
| TasselNetv2+  | TBD                   | No            | 262K      | 64.84 | 102.06 | 40.10 | 0.7364        |
| CSRNet        | CVPR 2018             | VGG16         | 16.3M     | 61.20 | 101.28 | 53.53 | 0.7323        |
| BCNet         | TCSVT 2019            | VGG16         | 14.8M     | 54.24 | 90.24  | 34.99 | 0.7952        |
| BCNet-BN      | TCSVT 2019            | VGG16         | 14.8M     | 46.90 | 76.83  | 29.58 | 0.8484        |
| Uniform Kernel |
| MCNN          | CVPR 2016             | No            | 134K      | 76.08 | 117.74 | 78.80 | 0.6623        |
| TasselNetv2+  | TBD                   | No            | 262K      | 64.49 | 100.92 | 38.84 | 0.7447        |
| CSRNet        | CVPR 2018             | VGG16         | 16.3M     | 48.96 | 82.12  | 39.69 | 0.8253        |
| BCNet-BN      | TCSVT 2019            | VGG16         | 14.8M     | 51.51 | 82.82  | 29.49 | 0.8358        |
| Bivariate Gaussian Kernel |
| MCNN          | CVPR 2016             | No            | 134K      | 73.17 | 115.36 | 67.21 | 0.6874        |
| TasselNetv2+  | TBD                   | No            | 262K      | 65.04 | 102.81 | 41.23 | 0.7320        |
| CSRNet        | CVPR 2018             | VGG16         | 16.3M     | 48.08 | 79.39  | 36.50 | 0.8365        |
| BCNet-BN      | TCSVT 2019            | VGG16         | 14.8M     | 48.57 | 77.71  | 28.91 | 0.8453        |