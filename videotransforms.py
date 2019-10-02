import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch

# for debug
from dataset import I3DDataSet


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupColorJitter(object):
    def __init__(self, brightness=0,contrast=0,saturation=0,hue=0):
        self.worker = torchvision.transforms.ColorJitter(brightness,contrast,saturation,hue)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class GroupNormalize(object):

    def __call__(self, tensor):
        tensor = (tensor/255.)*2 - 1
        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]



class GroupRandomResizeCrop(object):
    """
    random resize image to shorter size = [256,320] (e.g.),
    and random crop image to 224[e.g.]
    p.s.: if input size > 224, resize_range should be enlarged in equal proportion
    """
    def __init__(self, resize_range, input_size, interpolation=Image.BILINEAR):
        self.resize_range = resize_range
        self.crop_worker = GroupRandomCrop(input_size)
        self.interpolation = interpolation

    def __call__(self, img_group):
        resize_size = random.randint(self.resize_range[0],self.resize_range[1])
        resize_worker = GroupScale(resize_size)
        resized_img_group = resize_worker(img_group)
        crop_img_group = self.crop_worker(resized_img_group)

        return crop_img_group


class Stack(object):

    def __call__(self, img_group):
        # if img_group[0].mode == 'L':
        #     stacked_group = np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        # else:
        #     stacked_group = np.concatenate([np.expand_dims(x, 3) for x in img_group], axis=3)
        #print(stacked_group[0].shape)
        stacked_group = np.concatenate([np.expand_dims(x, 3) for x in img_group], axis=3)
        return stacked_group


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C x D) in the range [0, 255]
    to a torch.FloatTensor of shape (C x D x H x W) in the range [0.0, 1.0] """

    def __call__(self, pic):
        img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        return img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data


if __name__ == "__main__":

    dataset = I3DDataSet(
            list_file="/home/anta/pytorch-i3d/data/train_rgb.list",
            sample_frames=64,
            modality="flow",
            image_tmpl="config.DATASET.FILENAMES",
            transform=torchvision.transforms.Compose([
                       GroupRandomResizeCrop(
                        [256, 320], 224),
                        GroupRandomHorizontalFlip(is_flow=False),
                       Stack(),
                       ToTorchFormatTensor(),
                       GroupNormalize(0, 0),
                   ])
        )

    im = dataset.__getitem__(100)[0]
    print(im)

