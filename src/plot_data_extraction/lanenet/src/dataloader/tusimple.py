import os
import torch
import torch.utils.data as data
import json
import cv2
from ..utils.utils import get_binary_labels, get_instance_labels
import logging
logger = logging.getLogger(__name__)

from .base import get_image_transform

class TuSimpleDataLoader(data.Dataset):
    """
    Load raw images and labels
    where labels are a binary map
    and an instance semegtation map
    """

    def __init__(self, opt, split='train', return_org_image=False):
        super(TuSimpleDataLoader, self).__init__()

        self.image_dir = opt.image_dir
        self.thickness = opt.thickness
        self.height = opt.height
        self.width = opt.width
        self.max_lanes = opt.max_lanes
        self.return_org_image = return_org_image

        self.image_transform = get_image_transform(
            height=self.height, width=self.width)

        logger.info('Loading meta file: %s', opt.meta_file)

        data = json.load(open(opt.meta_file))
        self.info = data[split]

        self.image_ids = list(self.info.keys())

    def __getitem__(self, index):

        image_id = self.image_ids[index]
        file_name = self.info[image_id]['raw_file']
        file_path = os.path.join(self.image_dir, file_name)
        image = cv2.imread(file_path)  # in BGR order
        width_org = image.shape[1]
        height_org = image.shape[0]

        image = cv2.resize(image, (self.width, self.height))

        x_lanes = self.info[image_id]['lanes']
        y_samples = self.info[image_id]['h_samples']
        pts = [
            [(x, y) for(x, y) in zip(lane, y_samples) if x >= 0]
            for lane in x_lanes]

        # remove empty lines (not sure the role of these lines, but it causes
        # bugs)
        pts = [l for l in pts if len(l) > 0]

        x_rate = 1.0*self.width/width_org
        y_rate = 1.0*self.height/height_org

        pts = [[(int(round(x*x_rate)), int(round(y*y_rate)))
                for (x, y) in lane] for lane in pts]

        # get the binary segmentation image and convert it into labels,
        # that has size 2 x Height x Weight
        bin_labels = get_binary_labels(self.height, self.width, pts,
                                       thickness=self.thickness)

        # get the instance segmentation image and convert it to labels
        # that has size Max_lanes x Height x Width
        ins_labels, n_lanes = get_instance_labels(self.height, self.width, pts,
                                                  thickness=self.thickness,
                                                  max_lanes=self.max_lanes)

        # transform the image, and convert to Tensor
        image_t = self.image_transform(image)
        bin_labels = torch.Tensor(bin_labels)
        ins_labels = torch.Tensor(ins_labels)

        if self.return_org_image:
            # original image is converted to RGB to be displayed
            # not a good practice here, maybe it is better to write the
            # unnormalization transform
            image_raw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_t, bin_labels, ins_labels, n_lanes, image_raw, image_id
        else:
            return image_t, bin_labels, ins_labels, n_lanes

    def __len__(self):
        return len(self.image_ids)


class TuSimpleTestDataLoader(data.Dataset):
    """Load image and y_samples, generate x_lanes for each lane
    this is used at test time, where the full x, y labels are not available
    """
    def __init__(self, opt, split='test', return_org_image=False):
        super(TuSimpleTestDataLoader, self).__init__()

        self.image_dir = opt.image_dir
        self.thickness = opt.thickness
        self.height = opt.height
        self.width = opt.width
        self.max_lanes = opt.max_lanes

        self.image_transform = get_image_transform(
            height=self.height, width=self.width)

        test_lines = [l for l in open(opt.meta_file, 'rb')]
        logger.info('Loaded %s test images', len(test_lines))

        info = []
        for l in test_lines:
            img_info = json.loads(l)
            info.append(img_info)
        self.info = info

    def __getitem__(self, index):

        file_name = self.info[index]['raw_file']
        file_path = os.path.join(self.image_dir, file_name)
        image = cv2.imread(file_path)  # in BGR order
        width_org = image.shape[1]
        height_org = image.shape[0]

        image = cv2.resize(image, (self.width, self.height))
        # transform the image, and convert to Tensor
        image = self.image_transform(image)

        y_samples = self.info[index]['h_samples']

        # by default, pytorch don't behave properly if we return list here,
        # so convert to tensor before returning
        y_samples = torch.Tensor(y_samples)

        return image, y_samples, width_org, height_org, file_name

    def __len__(self):
        return len(self.info)
