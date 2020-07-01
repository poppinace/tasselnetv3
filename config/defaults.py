from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Directory
# -----------------------------------------------------------------------------
_C.DIR = CN()
_C.DIR.dataset = "maize_tassels_counting"
_C.DIR.exp = "dataset_model_inputblocksize_outputstride_resizeratio_cropsize_batchsize_epoch"
_C.DIR.snapshot = "./snapshots"
_C.DIR.result = "./results"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.name = "mtc"
_C.DATASET.list_train = "./data/training.odgt"
_C.DATASET.list_val = "./data/validation.odgt"
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.crop_size = (256, 256)
# image scale
_C.DATASET.image_scale = 0.00392156862745098 # 1./255
# image mean
_C.DATASET.image_mean = (0.485, 0.456, 0.406)
# image std
_C.DATASET.image_std = (0.229, 0.224, 0.225)
# gaussin kernel size
_C.DATASET.sigma = 24
# resizing ratio
_C.DATASET.resize_ratio = 0.125
# maximum input image size of long edge
_C.DATASET.img_max_size = 2048
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
# downsampling rate of the segmentation label
_C.DATASET.gt_downsampling_rate = 8
# scale the ground truth density map
_C.DATASET.scaling = 1
# preload dataset into the memory to speed up training
_C.DATASET.preload = True
# randomly horizontally flip images when training
_C.DATASET.random_flip = True
# randomly cropping images when training
_C.DATASET.random_crop = True

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.encoder = "tasselnetv2"
# architecture of net_decoder
_C.MODEL.decoder = ""
# architecture of net_counter
_C.MODEL.counter = "count_regressor"
# architecture of net_normalizer
_C.MODEL.normalizer = 'gpu_normalizer'
# ground truth generator
_C.MODEL.generator = "count_map"
# type of visualizer (architecture specific)
_C.MODEL.visualizer = 'pixel_averaging'
# weights to finetune net_encoder
_C.MODEL.block_size = 64
# weights to finetune net_decoder
_C.MODEL.output_stride = 8
# dim of counter
_C.MODEL.counter_dim = 128
# number of classes
_C.MODEL.num_class = 8
# step of classification
_C.MODEL.step_log = 0.1
# start of log
_C.MODEL.start_log = -2
# use pretrained model
_C.MODEL.pretrain = True
# fix bn params, only under finetuning
_C.MODEL.fix_bn = False

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# restore training from a checkpoint
_C.TRAIN.checkpoint = "model_ckpt.pth.tar"
# loss function
_C.TRAIN.loss = "L1Loss"
# loss reduction
_C.TRAIN.loss_reduction = "mean"
# batch size
_C.TRAIN.batch_size = 16
# epochs to train for
_C.TRAIN.num_epochs = 20
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000
# optimizer and learning rate
_C.TRAIN.optimizer = "SGD"
_C.TRAIN.lr_encoder = 0.01
_C.TRAIN.lr_decoder = 0.01
_C.TRAIN.encoder_multiplier = 1
# milestone
_C.TRAIN.milestones = (200, 400)
# momentum
_C.TRAIN.momentum = 0.95
# weights regularizer
_C.TRAIN.weight_decay = 5e-4
# number of data loading workers
_C.TRAIN.num_workers = 0
# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = 2020
# generate training masks
_C.TRAIN.generate_mask = False

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# the checkpoint to evaluate on
_C.VAL.checkpoint = "model_best.pth.tar"
# currently only supports 1
_C.VAL.batch_size = 1
# frequency to display
_C.VAL.disp_iter = 10
# frequency to validate
_C.VAL.val_epoch = 10
# evaluate_only
_C.VAL.evaluate_only = False
# output visualization during validation
_C.VAL.visualization = False
_C.VAL.normalization_before_visualization = False