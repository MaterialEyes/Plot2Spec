import torch
import torch.utils.data as data
import json
import cv2
from ..utils.utils import get_binary_labels, get_instance_labels
import logging
logger = logging.getLogger(__name__)

from .base import get_image_transform

class CULaneDataLoader(data.Dataset):
    """
    Load raw images and labels
    where labels are a binary map
    and an instance semegtation map
    """

    def __init__(self, opt, split='train', return_org_image=False):
        super(CULaneDataLoader, self).__init__()

        self.image_dir = opt.image_dir
        self.thickness = opt.thickness
        self.height = opt.height
        self.width = opt.width
        self.max_lanes = opt.max_lanes
        self.max_points = 40
        self.return_org_image = return_org_image

        self.image_transform = get_image_transform(
            height=self.height, width=self.width)

        logger.info('Loading meta file: %s', opt.meta_file)

        data = json.load(open(opt.meta_file))
        info = data[split]
        # filter image with no annotations

        if split in ['train', 'val']:
            num_image_before = len(info)
            info = {k:info[k] for k in info if len(info[k]['pts']) > 0}
            num_image_after = len(info)
            logger.warning('REMOVED %d/%d images that have no lanes!',
                           num_image_before - num_image_after,
                           num_image_before)

        self.info = info
        self.image_ids = list(self.info.keys())

    def __getitem__(self, index):

        image_id = self.image_ids[index]
        file_name = self.info[image_id]['raw_file']
        file_path = self.image_dir + file_name
        image = cv2.imread(file_path)  # in BGR order
        width_org = image.shape[1]
        height_org = image.shape[0]

        image = cv2.resize(image, (self.width, self.height))
        pts = self.info[image_id]['pts']

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
        new_pts = torch.zeros(self.max_lanes, self.max_points, 2)
        for i, ps in enumerate(pts):
            for j,p in enumerate(ps):
                if p[0] > 0 and p[1] > 0:
                    new_pts[i, j] = torch.Tensor(p)

        if self.return_org_image:
            # original image is converted to RGB to be displayed
            # not a good practice here, maybe it is better to write the
            # unnormalization transform
            image_raw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_t, bin_labels, ins_labels, n_lanes, new_pts, image_raw, image_id
        else:
            return image_t, bin_labels, ins_labels, n_lanes, new_pts

    def __len__(self):
        return len(self.image_ids)


class CULaneTestDataLoader(CULaneDataLoader):
    """
    Load raw images and labels
    where labels are a binary map
    and an instance semegtation map
    """

    def __init__(self, opt, split='train', return_org_image=False):
        super(CULaneTestDataLoader, self).__init__(opt, split, return_org_image)

    def __getitem__(self, index):

        image_id = self.image_ids[index]
        file_name = self.info[image_id]['raw_file']
        file_path = self.image_dir + file_name
        image = cv2.imread(file_path)  # in BGR order
        width_org = image.shape[1]
        height_org = image.shape[0]

        image = cv2.resize(image, (self.width, self.height))
        image = self.image_transform(image)

        # y_samples values are obtained from this code
        # https://github.com/XingangPan/SCNN/blob/master/tools/prob2lines/main.m
        # assuming image size is alway 1640x590
        assert height_org == 590
        y_samples = [(590-(m-1)*20)-1 for m in range(1, 19)]
        y_samples = torch.Tensor(y_samples)

        #return image, y_samples, width_org, height_org, image_id
        return image, y_samples, width_org, height_org, file_name

