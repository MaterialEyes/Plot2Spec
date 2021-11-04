import argparse
import os
import json
from tqdm import tqdm
from datetime import datetime
import cv2

import logging
logger = logging.getLogger(__name__)

from utils import get_binary_image, get_instance_image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'meta_file',
        type=str,
        help='Path to the metadata file'
    )
    parser.add_argument(
        'image_dir',
        type=str,
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '--bin_dir',
        type=str,
        help='Path to the output directory'
    )
    parser.add_argument(
        '--ins_dir',
        type=str,
        help='Path to the output directory'
    )
    parser.add_argument(
        '--thickness',
        type=int,
        default=10,
        help='Line thickness'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        type=str,
        default=['train', 'val'],
        help='splits that have labels'
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

    if not os.path.exists(args.bin_dir):
        os.makedirs(args.bin_dir)

    if not os.path.exists(args.ins_dir):
        os.makedirs(args.ins_dir)

    logger.info('Loading meta file: %s', args.meta_file)
    data = json.load(open(args.meta_file))

    for split in args.splits:
        image_ids = data[split].keys()
        logger.info(
            'Processing %d images on split [%s]...',
            len(image_ids),
            split)
        for image_id in tqdm(image_ids):
            file_name = data[split][image_id]['raw_file']
            file_path = os.path.join(args.image_dir, file_name)

            output_file = '-'.join(file_name.split('/'))
            bin_image_path = os.path.join(args.bin_dir, output_file)
            ins_image_path = os.path.join(args.ins_dir, output_file)

            if os.path.exists(bin_image_path) and os.path.exists(
                    ins_image_path):
                logger.info('Skiped because file existed: %s!', bin_image_path)
                continue

            image = cv2.imread(file_path)
            x_lanes = data[split][image_id]['lanes']
            y_samples = data[split][image_id]['h_samples']

            pts = [
                [(x, y) for(x, y) in zip(lane, y_samples) if x >= 0]
                for lane in x_lanes]
            bin_image = get_binary_image(image, pts, thickness=args.thickness)
            ins_image = get_instance_image(
                image, pts, thickness=args.thickness)

            cv2.imwrite(bin_image_path, bin_image)
            cv2.imwrite(ins_image_path, ins_image)

    logger.info('Time: %s', datetime.now() - start)
