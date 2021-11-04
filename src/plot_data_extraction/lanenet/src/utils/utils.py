import numpy as np
import argparse
import cv2

import logging
logger = logging.getLogger(__name__)


def get_binary_image(img, pts, thickness=5):
    """ Get the binary image

    Args:
        img: numpy array
        pts: set of lanes, each lane is a set of points

    Output:

    """
    # import pdb; pdb.set_trace()
    bin_img = np.zeros(shape=[img.shape[0], img.shape[1]], dtype=np.uint8)
    for i, lane in enumerate(pts):
        cv2.polylines(
            bin_img,
            np.int32(
                [lane]),
            isClosed=False,
            color=255,
            thickness=thickness)

    return bin_img


def get_instance_image(img, pts, thickness=5):
    """  Get the instance segmentation images,
    where each lane is annotated using a polyline with a
    different color

    Args:
            image
            pts

    Output:
            instance segmentation image
    """
    ins_img = np.zeros(shape=[img.shape[0], img.shape[1]], dtype=np.uint8)
    nlanes = len(pts)
    color_codes = list(range(0, 255, 255//(nlanes + 1)))[1:]

    for i, lane in enumerate(pts):
        cv2.polylines(
            ins_img,
            np.int32(
                [lane]),
            isClosed=False,
            color=color_codes[i],
            thickness=thickness)

    return ins_img


def get_binary_labels(height, width, pts, thickness=5):
    """ Get the binary labels. this function is similar to
    @get_binary_image, but it returns labels in 2 x H x W format
    this label will be used in the CrossEntropyLoss function.

    Args:
        img: numpy array
        pts: set of lanes, each lane is a set of points

    Output:

    """
    bin_img = np.zeros(shape=[height, width], dtype=np.uint8)
    for lane in pts:
        cv2.polylines(
            bin_img,
            np.int32(
                [lane]),
            isClosed=False,
            color=255,
            thickness=thickness)

    bin_labels = np.zeros_like(bin_img, dtype=bool)
    bin_labels[bin_img != 0] = True
    bin_labels = np.stack([~bin_labels, bin_labels]).astype(np.uint8)
    return bin_labels


def get_instance_labels(height, width, pts, thickness=5, max_lanes=5):
    """  Get the instance segmentation labels.
    this function is similar to @get_instance_image,
    but it returns label in L x H x W format

    Args:
            image
            pts

    Output:
            max Lanes x H x W, number of actual lanes
    """
    if len(pts) > max_lanes:
        #logger.warning('More than 5 lanes: %s', len(pts))
        pts = pts[:max_lanes]

    ins_labels = np.zeros(shape=[0, height, width], dtype=np.uint8)

    n_lanes = 0
    for lane in pts:
        ins_img = np.zeros(shape=[height, width], dtype=np.uint8)
        cv2.polylines(
            ins_img,
            np.int32(
                [lane]),
            isClosed=False,
            color=1,
            thickness=thickness)

        # there are some cases where the line could not be draw, such as one
        # point, we need to remove these cases
        # also, if there is overlapping among lanes, only the latest lane is
        # labeled
        if ins_img.sum() != 0:
            # comment this line because it will zero out previous lane data,
            # this leads to NaN error in computing the discriminative loss

            # ins_labels[:, ins_img != 0] = 0
            ins_labels = np.concatenate([ins_labels, ins_img[np.newaxis]])
            n_lanes += 1

    if n_lanes < max_lanes:
        n_pad_lanes = max_lanes - n_lanes
        pad_labels = np.zeros(
            shape=[
                n_pad_lanes,
                height,
                width],
            dtype=np.uint8)
        ins_labels = np.concatenate([ins_labels, pad_labels])

    return ins_labels, n_lanes


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every [lr_update] epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


def get_lane_area(binary_seg_ret, instance_seg_ret):
    """ Get possible lane area from the binary segmentation results

    Args:
        binary_seg_ret:
        instance_seg_ret:

    Returns:

    """
    idx = np.where(binary_seg_ret == 1)

    lane_embedding_feats = []
    lane_coordinate = []
    for i in range(len(idx[0])):
        lane_embedding_feats.append(instance_seg_ret[:, idx[0][i], idx[1][i]])
        lane_coordinate.append([idx[0][i], idx[1][i]])

    return np.array(
        lane_embedding_feats, np.float32), np.array(
        lane_coordinate, np.int64)


def get_lane_mask(num_clusters, labels, binary_seg_ret, lane_coordinate):
    """
    Get a masking images, where each lane is colored by a different color

    Args:
        num_clusters: number of possible lanes
        labels: lane label for each point
        binary_seg_ret:
        lane_coordinate

    Returns:
        a mask image

    """

    color_map = [(255, 0, 0),
                 (0, 255, 0),
                 (0, 0, 255),
                 (125, 125, 0),
                 (0, 125, 125),
                 (125, 0, 125),
                 (50, 100, 50),
                 (100, 50, 100)]

    # continue working on this
    if num_clusters > 8:
        cluster_sample_nums = []
        for i in range(num_clusters):
            cluster_sample_nums.append(len(np.where(labels == i)[0]))
        sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
        cluster_index = np.array(range(num_clusters))[sort_idx[0:8]]
    else:
        cluster_index = range(num_clusters)

    mask_image = np.zeros(
        shape=[
            binary_seg_ret.shape[0],
            binary_seg_ret.shape[1],
            3],
        dtype=np.uint8)

    for index, ci in enumerate(cluster_index):
        idx = np.where(labels == ci)
        coord = lane_coordinate[idx]
        coord = np.flip(coord, axis=1)
        color = color_map[index]
        coord = np.array([coord])
        cv2.polylines(
            img=mask_image,
            pts=coord,
            isClosed=False,
            color=color,
            thickness=2)

    return mask_image

def draw_lane_mask(x_preds, y_samples, img_h, img_w):
    """
    Get a masking images, where each lane is colored by a different color

    Args:
        num_clusters: number of possible lanes
        labels: lane label for each point
        binary_seg_ret:
        lane_coordinate

    Returns:
        a mask image

    """

    color_map = [(255, 0, 0),
                 (0, 255, 0),
                 (0, 0, 255),
                 (125, 125, 0),
                 (0, 125, 125),
                 (125, 0, 125),
                 (50, 100, 50),
                 (100, 50, 100)]

    mask_image = np.zeros(
        shape=[img_h, img_w, 3],
        dtype=np.uint8)

    for index,xs in enumerate(x_preds):
        coord = [[x, y] for (x, y) in zip(xs, y_samples) if x > 0 and x < img_w]
        coord = np.array([coord])
        color = color_map[index]
        cv2.polylines(
            img=mask_image,
            pts=coord,
            isClosed=False,
            color=color,
            thickness=2)

    return mask_image

def output_lanes(num_clusters, labels, bin_pred, lane_coordinate, y_samples,
                 method='polyfit'):
    """ Output lane predictions according to TuSimple Challenge's format
    Basically, provide x coordinate at each requested y coordinate

    At first a polynominal line is fitted, then its coefficients are used to
    make the predictions on new y points. The degree of the polynomial is set
    to the length of the y_samples, which could be not optimal

    Args:
        num_clusters:
        labels:
        binary_seg_ret:
        lane_coordinate:
        y_samples:

        method: if polyfit, fit a poly lines and use fit line to make predictions
                if max_prob, output x value that has max prob from bin_pred
    Returns:
        x_lanes: list of x-coordinate for each lane

    """
    if num_clusters > 8:
        cluster_sample_nums = []
        for i in range(num_clusters):
            cluster_sample_nums.append(len(np.where(labels == i)[0]))
        sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
        cluster_index = np.array(range(num_clusters))[sort_idx[0:8]]
    else:
        cluster_index = range(num_clusters)

    x_lanes = []
    for index, ci in enumerate(cluster_index):
        idx = np.where(labels == ci)
        coord = lane_coordinate[idx]
        y_coord = coord[:, 0]
        x_coord = coord[:, 1]
        if method == 'polyfit':
            z = np.polyfit(y_coord, x_coord, len(y_samples))
            predictor = np.poly1d(z)
            x_lane = [predictor(y) for y in y_samples]
        elif method == 'maxprob':
            y_samples = [int(round(y)) for y in y_samples]
            x_lane = []
            for y in y_samples:
                sel_coord = coord[coord[:, 0] == y]
                if sel_coord.size > 0:
                    sel_coord_x = sel_coord[:, 1]
                    prob_coord_x = bin_pred[y, sel_coord_x]
                    max_prob_idx = prob_coord_x.argmax()
                    max_x = sel_coord_x[max_prob_idx]
                    x_lane.append(max_x)
                else:
                    x_lane.append(-2)

        x_lanes.append(x_lane)

    return x_lanes

