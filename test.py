import os
import time
import sys
import numpy as np

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from videotransforms import (
    Stack, GroupNormalize, ToTorchFormatTensor, GroupScale, GroupCenterCrop
)

from models.pytorch_i3d import InceptionI3d

from dataset import I3DDataSet
from default import _C as config
from default import update_config

# to work with vscode debugger https://github.com/joblib/joblib/issues/864
import multiprocessing
multiprocessing.set_start_method('spawn', True)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_model(modality, state_dict_file):

    channels = 3 if modality == "RGB" else 2
    model = InceptionI3d(config.DATASET.NUM_CLASSES, in_channels=channels)
    state_dict = torch.load(state_dict_file)
    model.load_state_dict(state_dict)

    model = torch.nn.DataParallel(model).cuda()

    return model


def test(test_loader, modality, state_dict_file):

    model = load_model(modality, state_dict_file)

    model.eval()

    target_list = []
    predictions_list = []
    with torch.no_grad():
        end = time.time()
        for step, (input, target) in enumerate(test_loader):
            #print("step", step)
            target_list.append(target)
            input = input.cuda(non_blocking=True)

            # compute output
            output = model(input)
            output = torch.mean(output, dim=2)
            predictions_list.append(output)
    
    targets = torch.cat(target_list)
    predictions = torch.cat(predictions_list)
    return targets, predictions


def run(*options, cfg=None):

    update_config(config, options=options, config_file=cfg)

    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(seed=config.SEED)

    # Setup Augmentation/Transformation pipeline
    input_size = config.TRAIN.INPUT_SIZE
    resize_range_min = config.TRAIN.RESIZE_RANGE_MIN

    test_augmentation = transforms.Compose(
        [
            GroupScale(resize_range_min),
            GroupCenterCrop(input_size)
        ]
    )

    # Data-parallel
    devices_lst = list(range(torch.cuda.device_count()))
    print("Devices {}".format(devices_lst))

    if (config.TEST.MODALITY == "RGB") or (config.TEST.MODALITY == "both"):

        rgb_loader = torch.utils.data.DataLoader(
            I3DDataSet(
                data_root=config.DATASET.DIR,
                split=config.DATASET.SPLIT,
                sample_frames=config.TRAIN.SAMPLE_FRAMES,
                modality="RGB",
                image_tmpl=config.DATASET.FILENAMES,
                train_mode=False,
                transform=torchvision.transforms.Compose([
                        test_augmentation,
                        Stack(),
                        ToTorchFormatTensor(),
                        GroupNormalize(0, 0),
                    ])
            ),
            batch_size=config.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True
        )

        rgb_model_file = 'checkpoints/i3d_rgb_split1_epoch199.pt'
        if not os.path.exists(rgb_model_file):
            raise FileNotFoundError(rgb_model_file, " does not exist")

        print("scoring with rgb model")
        targets, rgb_predictions = test(
            rgb_loader,
            "RGB",
            rgb_model_file
        )
        
        targets = targets.cuda(non_blocking=True)
        rgb_top1_accuracy = accuracy(rgb_predictions, targets, topk=(1, ))
        print("rgb top1 accuracy: ", rgb_top1_accuracy[0].cpu().numpy().tolist())
    
    if (config.TEST.MODALITY == "flow") or (config.TEST.MODALITY == "both"):

        flow_loader = torch.utils.data.DataLoader(
            I3DDataSet(
                data_root=config.DATASET.DIR,
                split=config.DATASET.SPLIT,
                sample_frames=config.TRAIN.SAMPLE_FRAMES,
                modality="flow",
                image_tmpl=config.DATASET.FILENAMES,
                train_mode=False,
                transform=torchvision.transforms.Compose([
                        test_augmentation,
                        Stack(),
                        ToTorchFormatTensor(),
                        GroupNormalize(0, 0),
                    ])
            ),
            batch_size=config.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True
        )

        flow_model_file = 'checkpoints/i3d_flow_split1_epoch199.pt'
        if not os.path.exists(flow_model_file):
            raise FileNotFoundError(flow_model_file, " does not exist")

        print("scoring with flow model")
        targets, flow_predictions = test(
            flow_loader,
            "flow",
            flow_model_file
        )

        targets = targets.cuda(non_blocking=True)
        flow_top1_accuracy = accuracy(flow_predictions, targets, topk=(1, ))
        print("flow top1 accuracy: ", flow_top1_accuracy[0].cpu().numpy().tolist())

    if config.TEST.MODALITY == "both":
        predictions = torch.stack([rgb_predictions, flow_predictions])
        predictions_mean = torch.mean(predictions, dim=0)
        top1accuracy = accuracy(predictions_mean, targets, topk=(1, ))
        print("combined top1 accuracy: ", top1accuracy[0].cpu().numpy().tolist())


if __name__ == "__main__":
    fire.Fire(run)