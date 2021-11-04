import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import json
import glob
import cv2
import logging
logger = logging.getLogger(__name__)

from .base import get_image_transform

class DirDataLoader(data.Dataset):
    """Load image from a directory,
    this is usually used at test time, where the labels are not available
    """

    def __init__(self, opt, **kwargs):
        super(DirDataLoader, self).__init__()
        self.image_dir = opt.image_dir
        self.height = opt.height
        self.width = opt.width

        self.image_transform = get_image_transform(
            height=self.height, width=self.width)

        pattern = '{}/*.{}'.format(self.image_dir, opt.image_ext)
        image_list = [os.path.basename(f) for f in glob.glob(pattern)]
        self.image_list = sorted(image_list)

    def __getitem__(self, index):
        file_name = self.image_list[index]
        image_id = os.path.splitext(file_name)[0]
        file_path = os.path.join(self.image_dir, file_name)
        image = cv2.imread(file_path)  # in BGR order
        image = cv2.resize(image, (self.width, self.height))

        # transform the image, and convert to Tensor
        image_t = self.image_transform(image)
        image_raw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image_t, image_raw, image_id

    def __len__(self):
        return len(self.image_list)

