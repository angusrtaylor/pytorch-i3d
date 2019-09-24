# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ""
_C.LOG_DIR = "log"
_C.MODEL_DIR = "checkpoints"
_C.GPUS = (0,1,2,3)
_C.WORKERS = 16
_C.PRINT_FREQ = 50
_C.EVAL_FREQ = 5
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.LOG_CONFIG = "logging.conf"
_C.SEED = 42

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Dataset
_C.DATASET = CN()
_C.DATASET.NAME = 'hmdb51'
_C.DATASET.TRAIN = "/home/anta/pytorch-i3d/data/train_rgb.list"
_C.DATASET.TEST = "/home/anta/pytorch-i3d/data/test_rgb.list"
_C.DATASET.NUM_CLASSES = 51
_C.DATASET.FILENAMES = "frame{:04d}.jpg"

# NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = "i3d_flow"
_C.MODEL.ARCH = ""
_C.MODEL.PRETRAINED = True

# Train
_C.TRAIN = CN()
_C.TRAIN.INPUT_SIZE = 224
_C.TRAIN.RESIZE_RANGE_MIN = 256
_C.TRAIN.RESIZE_RANGE_MAX = 320
_C.TRAIN.INPUT_MEAN = [0.485, 0.456, 0.406]
_C.TRAIN.INPUT_STD = [0.229, 0.224, 0.225]
_C.TRAIN.SAMPLE_FRAMES = 64
_C.TRAIN.MODALITY = "flow"
_C.TRAIN.BATCH_SIZE = 24
_C.TRAIN.MAX_EPOCHS = 100

# Test
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 32

def update_config(cfg, options=None, config_file=None):
    cfg.defrost()

    if config_file:
        cfg.merge_from_file(config_file)

    if options:
        cfg.merge_from_list(options)

    cfg.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(_C, file=f)

