import os
import time
import sys
import numpy as np

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from opt_ranger import Ranger
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision
from torchvision import datasets, transforms

from videotransforms import (
    GroupRandomResizeCrop, GroupRandomHorizontalFlip, GroupColorJitter,
    GroupScale, GroupCenterCrop, GroupNormalize, Stack, ToTorchFormatTensor
)
from models.pytorch_i3d import InceptionI3d

from metrics import accuracy, AverageMeter

from dataset import I3DDataSet 
from tensorboardX import SummaryWriter

from default import _C as config
from default import update_config

# to work with vscode debugger https://github.com/joblib/joblib/issues/864
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def train(train_loader, model, criterion, optimizer, epoch, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for step, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        output = torch.mean(output, dim=2)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % config.PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
        
        if writer:
            writer.add_scalar('train/loss', losses.avg, epoch+1)
            writer.add_scalar('train/top1', top1.avg, epoch+1)
            writer.add_scalar('train/top5', top5.avg, epoch+1)


def validate(val_loader, model, criterion, epoch, writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for step, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            output = torch.mean(output, dim=2)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1,5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % config.PRINT_FREQ == 0:
                print(('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    step, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5)))

        print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
            .format(top1=top1, top5=top5, loss=losses)))

        if writer:
            writer.add_scalar('val/loss', losses.avg, epoch+1)
            writer.add_scalar('val/top1', top1.avg, epoch+1)
            writer.add_scalar('val/top5', top5.avg, epoch+1)

    return losses.avg


def run(*options, cfg=None):
    """Run training and validation of model

    Notes:
        Options can be passed in via the options argument and loaded from the cfg file
        Options loaded from default.py will be overridden by options loaded from cfg file
        Options passed in through options argument will override option loaded from cfg file
    
    Args:
        *options (str,int ,optional): Options used to overide what is loaded from the config. 
                                      To see what options are available consult default.py
        cfg (str, optional): Location of config file to load. Defaults to None.
    """
    update_config(config, options=options, config_file=cfg)

    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(seed=config.SEED)

    # Log to tensorboard
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # Setup Augmentation/Transformation pipeline
    input_size = config.TRAIN.INPUT_SIZE
    resize_range_min = config.TRAIN.RESIZE_RANGE_MIN
    resize_range_max = config.TRAIN.RESIZE_RANGE_MAX

    #is_flow = True if config.TRAIN.MODALITY == "flow" else False
    is_flow = False

    train_augmentation = transforms.Compose(
        [
            GroupRandomResizeCrop(
                [resize_range_min, resize_range_max], input_size),
            GroupRandomHorizontalFlip(is_flow=is_flow),
            #GroupColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
        ]
    )

    val_augmentation = transforms.Compose(
        [
            GroupScale(resize_range_min),
            GroupCenterCrop(input_size)
        ]
    )

    # Setup DataLoaders
    train_loader = torch.utils.data.DataLoader(
        I3DDataSet(
            data_root=config.DATASET.DIR,
            split=config.DATASET.SPLIT,
            sample_frames=config.TRAIN.SAMPLE_FRAMES,
            modality=config.TRAIN.MODALITY,
            transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(),
                       ToTorchFormatTensor(),
                       GroupNormalize(0, 0),
                   ])
        ),
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        I3DDataSet(
            data_root=config.DATASET.DIR,
            split=config.DATASET.SPLIT,
            sample_frames=config.TRAIN.SAMPLE_FRAMES,
            modality=config.TRAIN.MODALITY,
            transform=torchvision.transforms.Compose([
                       val_augmentation,
                       Stack(),
                       ToTorchFormatTensor(),
                       GroupNormalize(0, 0),
                   ]),
            train_mode=False,
        ),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # Setup Model
    if config.TRAIN.MODALITY == "RGB":
        channels = 3
        checkpoint = config.MODEL.PRETRAINED_RGB
    elif config.TRAIN.MODALITY == "flow":
        channels = 2
        checkpoint = config.MODEL.PRETRAINED_FLOW
    else:
        raise ValueError("Modality must be RGB or flow")

    i3d_model = InceptionI3d(400, in_channels=channels)
    i3d_model.load_state_dict(torch.load(checkpoint))

    # Replace final FC layer to match dataset
    i3d_model.replace_logits(config.DATASET.NUM_CLASSES)

    # Data-parallel
    devices_lst = list(range(torch.cuda.device_count()))
    print("Devices {}".format(devices_lst))
    if len(devices_lst) > 1:
        i3d_model = torch.nn.DataParallel(i3d_model).cuda()
    else:
        raise Exception('Get more GPUs')

    # Optimiser
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
       i3d_model.parameters(), 
       lr=1.0,
       momentum=0.9, 
       weight_decay=0.0000001
    )
    # optimizer = Ranger(i3d_model.parameters())
    # optimizer = optim.Adam(i3d_model.parameters(), lr=0.0001)

    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 50], gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=2,
        verbose=True,
        threshold=1e-4,
        min_lr=1e-4
    )
    
    # clr = cyclical_lr(step_size, min_lr=0.0001, max_lr=0.1, mode='triangular')
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, [clr])
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1, mode='triangular2', step_size_up=25)

    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    
    for epoch in range(config.TRAIN.MAX_EPOCHS):

        # train for one epoch
        train(train_loader,
            i3d_model,
            criterion,
            optimizer,
            epoch,
            writer
        )

        # scheduler.step()

        # evaluate on validation set
        if (epoch + 1) % config.EVAL_FREQ == 0 or epoch == config.TRAIN.MAX_EPOCHS - 1:
            val_loss = validate(val_loader, i3d_model, criterion, epoch, writer)
            scheduler.step(val_loss)
            torch.save(
                i3d_model.module.state_dict(),
                config.MODEL_DIR+'/'+config.MODEL.NAME+'_split'+str(config.DATASET.SPLIT)+'_epoch'+str(epoch).zfill(3)+'.pt')

    writer.close()

if __name__ == "__main__":
    fire.Fire(run)