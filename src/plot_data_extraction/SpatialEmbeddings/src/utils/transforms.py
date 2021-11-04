"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import random

import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

import torch
import cv2


class CropRandomObject:

    def __init__(self, keys=[],object_key="instance", size=100):
        self.keys = keys
        self.object_key = object_key
        self.size = size

    def __call__(self, sample):

        object_map = np.array(sample[self.object_key], copy=False)
        h, w = object_map.shape

        unique_objects = np.unique(object_map)
        unique_objects = unique_objects[unique_objects != 0]
        
        if unique_objects.size > 0:
            random_id = np.random.choice(unique_objects, 1)

            y, x = np.where(object_map == random_id)
            ym, xm = np.mean(y), np.mean(x)
            
            i = int(np.clip(ym-self.size[1]/2, 0, h-self.size[1]))
            j = int(np.clip(xm-self.size[0]/2, 0, w-self.size[0]))

        else:
            i = random.randint(0, h - self.size[1])
            j = random.randint(0, w - self.size[0])

        for k in self.keys:
            assert(k in sample)

            sample[k] = F.crop(sample[k], i, j, self.size[1], self.size[0])

        return sample




class RandomCrop(T.RandomCrop):

    def __init__(self, keys=[], size=100):

        super().__init__(size)
        self.keys = keys

    def __call__(self, sample):

        params = None

        for k in self.keys:

            assert(k in sample)

            if params is None:
                params = self.get_params(sample[k], self.size)

            sample[k] = F.crop(sample[k], *params)

        return sample

class RandomRotation(T.RandomRotation):

    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

        if isinstance(self.resample, collections.Iterable):
            assert(len(keys) == len(self.resample))

    def __call__(self, sample):

        angle = self.get_params(self.degrees)

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            resample = self.resample
            if isinstance(resample, collections.Iterable):
                resample = resample[idx]

            sample[k] = F.rotate(sample[k], angle, resample,
                                 self.expand, self.center)

        return sample


class Resize(T.Resize):

    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

        if isinstance(self.interpolation, collections.Iterable):
            assert(len(keys) == len(self.interpolation))

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            interpolation = self.interpolation
            if isinstance(interpolation, collections.Iterable):
                interpolation = interpolation[idx]

            sample[k] = F.resize(sample[k], self.size, interpolation)

        return sample

class CustomResizePad(object):
    
    def __init__(self, keys=[], **kwargs):
        self.keys = keys
        self.max_width_height = 512
        self.is_test = kwargs.get("is_test", False)
        
    def __call__(self, sample):
        w, h = sample['image'].size
        if w > h:
            nw = self.max_width_height
            nh = int(self.max_width_height/w*h)
            dw = 0
            dh = 0 if nh==nw else np.random.randint(nw-nh)
        else:
            nh = self.max_width_height
            nw = int(self.max_width_height/h*w)
            dh = 0
            dw = 0 if nh==nw else np.random.randint(nh-nw)
            
        if self.is_test:
            dw = 0
            dh = 0
        
        for idx, k in enumerate(self.keys):
            
            assert(k in sample)
            
            canvas = Image.new(mode=sample[k].mode, size=(self.max_width_height, self.max_width_height))
            if k == "image":
                sample[k] = sample[k].resize(size=(nw, nh), resample = Image.BILINEAR)
            else:
                sample[k] = sample[k].resize(size=(nw, nh), resample = Image.NEAREST)
            canvas.paste(sample[k], box=(dw,dh))
            sample[k] = canvas
            
        return sample
    
class Normalize():
    def __init__(self, keys=[], **kwargs):
        self.keys = keys
        assert isinstance(self.keys, str), "{}".format(type(self.keys))
        self.p = kwargs.get("p", 0.5)
        
    def __call__(self, sample):
        k = self.keys

        assert(k in sample)

        target = sample[k]
        if np.random.rand() > self.p:
            target = T.ToPILImage()(target)
            target = cv2.fastNlMeansDenoisingColored(np.array(target),None,10,10,7,21)
            target = Image.fromarray(np.uint8(target))
            target = T.ToTensor()(target)

        sample[k] = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(target)
        return sample
    

    
    
class ToTensor(object):

    def __init__(self, keys=[], type="float"):

        if isinstance(type, collections.Iterable):
            assert(len(keys) == len(type))

        self.keys = keys
        self.type = type

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            sample[k] = F.to_tensor(sample[k])

            t = self.type
            if isinstance(t, collections.Iterable):
                t = t[idx]

            if t == torch.ByteTensor:
                sample[k] = sample[k]*255

            sample[k] = sample[k].type(t)

        return sample


def get_transform(transforms):
    transform_list = []

    for tr in transforms:
        name = tr['name']
        opts = tr['opts']
        transform_list.append(globals()[name](**opts))

    return T.Compose(transform_list)
