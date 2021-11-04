import argparse
import os
import json
import numpy as np
import cv2
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)


def get_imageid(s, dataset='tusimple'):
    """Get image id from the raw_file path,
    e.g., clips/0313-1/8420/20.jpg'
    return: 0313-1-8420

    culane: driver_23_30frame/05151649_0422.MP4/00000.jpg
            driver_23_30frame_05151649_0422_00000
    """
    if dataset == 'tusimple':
        p = s.split('/')
        img_id = p[1] + '-' + p[2]
    elif dataset == 'culane':
        p = list(filter(None, s.split('/')))
        p[1] = p[1].split('.')[0]
        p[2] = p[2].split('.')[0]
        img_id = '_'.join(p)
    else:
        img_id = s
    return img_id

def to_json(lines):
    """Convert list of json to json format

    Return:
        imgs[img_id] = {'lanes': [], 'h_samples': [], 'raw_file': []}

    """
    imgs = {}
    for l in lines:
        img_info = json.loads(l)
        img_id = get_imageid(img_info['raw_file'])
        if img_id in imgs:
            logger.error('Duplicated key %s', img_id)
        else:
            imgs[img_id] = img_info

    return imgs

def iou(img1, img2, EPS = 1e-6):
    intersection = (img1 & img2).sum()
    union = (img1 | img2).sum()
    iou = (intersection + EPS) / (union + EPS)
    return iou

def get_merged_lanes(lane_labels, method='iou', img_h=720, img_w=1280):
    """ Merge consecutive lines into 1 lane

        labels: list of raw lane annotations (line annotations)
    """
    # getting all lanes and sorting points by its heights
    lanes = []
    for l in lane_labels:
        lane = l['poly2d'][0]['vertices']
        lane = sorted(lane, key=lambda x: x[1])
        lanes.append(lane)

    if method == 'angle':
        # maximum angle difference threshold to be considered a same line
        method_threshold = 4
        angles = []
        for lane in lanes:
            # take the start point and end point to compute the line angle
            # while this improve the accuracy in case of straing light
            # this is prone to error in case of curve (use first 2 points instead)
            x0 = lane[0][0]
            x1 = lane[1][0]
            y0 = lane[0][1]
            y1 = lane[1][1]
            # use abs here because we only care about the angle, not the orientation
            angle = np.rad2deg(np.arctan2(abs(y1 - y0), abs(x1 - x0)))
            angles.append(angle)

        # difference between two consecutive angles in this list
        diffs = [abs(j-i) for i, j in zip(angles[:-1], angles[1:])]

    elif method == 'iou':
        # maximum angle difference threshold to be considered a same line
        method_threshold = -0.1

        imgs = []
        for lane in lanes:
            # fill the lines so that IoU can be computed!
            tmp_img = np.zeros(shape=[img_h, img_w], dtype=np.uint8)
            cv2.polylines(tmp_img, np.int32([lane]), isClosed=False, color=1, thickness=20)
            imgs.append(np.int32(tmp_img))

        # IoU difference between two consecutive (masked) images
        diffs = [-iou(imgs[i], imgs[i+1]) for i, img in enumerate(imgs[:-1])]

    merge_lanes = []
    line_merged = False
    # merge lanes based on angle differences
    for i, diff in enumerate(diffs):
        if line_merged:
            line_merged = False
            continue
        this_lane = lanes[i]
        if diff < method_threshold:
            # next line will be merged
            this_lane.extend(lanes[i+1])
            line_merged = True

        this_lane = sorted(this_lane, key=lambda x: x[1])
        merge_lanes.append(this_lane)
        if i == len(diffs)-1 and not line_merged:
            this_lane = sorted(lanes[i+1], key=lambda x: x[1])
            merge_lanes.append(this_lane)

    return merge_lanes


def generate_bdd(input_dir):
    """ Generate metadata for TuSimple dataset
    """

    splits = ['train', 'val']

    out = {}
    for split in splits:
        label_file = os.path.join(input_dir, 'labels_new', \
                                  'bdd100k_labels_images_{}.json'.format(split))

        logger.info('Loading label file: %s', label_file)
        image_list = json.load(open(label_file))
        logger.info('Generating metadata for split: %s', split)
        split_imgs = {}
        for img_info in tqdm(image_list):
            img_name = img_info['name']
            img_id = os.path.splitext(img_name)[0]
            image_file = os.path.join('images', '100k', split, img_name)

            # only use parrallel lines at the moment
            lane_labels = [l for l in img_info['labels'] \
                            if l['category'] == 'lane' and \
                            l['attributes']['laneDirection'] == 'parallel']

            pts = get_merged_lanes(lane_labels)
            img_info = {
                'raw_file': image_file,
                'pts': pts
            }
            split_imgs[img_id] = img_info
        out[split] = split_imgs
    return out


def generate_culane(input_dir, image_ext='.jpg'):
    """ Generate metadata for TuSimple dataset
    """

    splits = ['train', 'val', 'test']

    out = {}
    for split in splits:
        label_file = os.path.join(input_dir, 'list', split + '.txt')
        image_list = [f.rstrip('\n') for f in open(label_file)]
        logger.info('Generating metadata for split: %s', split)
        split_imgs = {}
        for image_file in tqdm(image_list):
            img_id = get_imageid(image_file, dataset='culane')
            label_file = input_dir + image_file.replace(image_ext, '.lines.txt')
            label_lines = [[float(x) for x in f.split()] for f in open(label_file, 'r')]
            pts = [list(zip(lane[0::2], lane[1::2])) for lane in label_lines]
            img_info = {
                'raw_file': image_file,
                'pts': pts
            }
            split_imgs[img_id] = img_info
        out[split] = split_imgs
    return out

def generate_tusimple(input_dir, val_size):
    """ Generate metadata for TuSimple dataset
    """
    trainval_label_files = ['label_data_0313.json',
                            'label_data_0531.json',
                            'label_data_0601.json']

    test_label_file = 'test_tasks_0627.json'

    trainval_lines = []
    trainval_fileids = []
    n_train_images = 0
    for i, f in enumerate(trainval_label_files):
        label_file = os.path.join(input_dir, f)
        lines = [l for l in open(label_file, 'rb')]
        logger.info('Loaded %s images', len(lines))
        n_train_images += len(lines)
        trainval_lines.extend(lines)

        y = [i]*len(lines)
        trainval_fileids.extend(y)

    logger.info('Loaded %s training images', n_train_images)

    # this is to make sure val data is stratified over all annotation files
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=0)
    train_index, val_index = next(sss.split(trainval_lines, trainval_fileids))

    train_lines = [trainval_lines[i] for i in train_index]
    val_lines = [trainval_lines[i] for i in val_index]

    test_label_file = os.path.join(input_dir, test_label_file)
    test_lines = [l for l in open(test_label_file, 'rb')]
    logger.info('Loaded %s test images', len(test_lines))

    out = {}
    out['train'] = to_json(train_lines)
    out['val'] = to_json(val_lines)
    out['test'] = to_json(test_lines)

    return out

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dir',
        type=str,
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '--dataset',
        default='tusimple',
        choices=['tusimple', 'culane', 'bdd'],
        help='Name of dataset')
    parser.add_argument(
        '--output_file',
        type=str,
        help='Path to the output file'
    )
    parser.add_argument(
        '--val_size',
        type=int,
        default=0.1,
        help='percentage of training image to be used as val data'
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s: %(message)s')

    logger.info(
        'Input arguments: %s',
        json.dumps(
            vars(args),
            sort_keys=True,
            indent=4))

    start = datetime.now()

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.dataset == 'tusimple':
        out = generate_tusimple(args.input_dir, args.val_size)
    elif args.dataset == 'culane':
        out = generate_culane(args.input_dir)
    elif args.dataset == 'bdd':
        out = generate_bdd(args.input_dir)
    else:
        raise ValueError('Unknown dataset %s', args.dataset)

    json.dump(out, open(args.output_file, 'w'))
    logger.info('Saved output to %s', args.output_file)
    logger.info('Time: %s', datetime.now() - start)
