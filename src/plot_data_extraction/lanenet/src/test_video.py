

import argparse
import os
import json
from datetime import datetime
import cv2
import numpy as np
import time
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
import logging
import warnings
from moviepy.editor import VideoFileClip

from torch.nn.parallel.scatter_gather import gather
from models.model import LaneNet, PostProcessor, LaneClustering
from utils.utils import AverageMeter, get_lane_area, get_lane_mask, draw_lane_mask, output_lanes
from utils.parallel import DataParallelModel

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def preprocess(image, width, height):
    target_size = (width, height)
    image = cv2.resize(image, target_size)
    return image

def get_image_transform(height=256, width=512):

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    t = [transforms.ToTensor(),
         normalizer]

    transform = transforms.Compose(t)
    return transform

def test_video(model, input_file, output_file, width, height, genline_method=None):
    inclip = VideoFileClip(input_file)
    clip = inclip.fl_image(lambda image: test_image(model, image, width, height, genline_method))
    clip.write_videofile(output_file, audio=False)

def test_image(model, image, width, height, genline_method=None, show_image=False):
    """Test a model on an image and display detected lanes

    Args:
        model (LaneNet): a LaneNet model
        genline_method (default is None), else generate lanes from heat maps, using
        this method. supported methods include 'maxprob' and 'polyfit'

    Returns:
        image

    """
    model.eval()

    postprocessor = PostProcessor()
    clustering = LaneClustering()
    image_transform = get_image_transform()

    run_time = AverageMeter()
    end = time.time()

    image_pre = preprocess(image, width, height)
    image = image_transform(image_pre)
    image = image.unsqueeze(0)

    with torch.no_grad():
        images = Variable(image)

        if torch.cuda.is_available():
            images = images.cuda()

        if torch.cuda.device_count() <= 1:
            bin_preds, ins_preds = model(images)
        else:
            bin_preds, ins_preds = gather(model(images), 0, dim=0)

        # convert to probabiblity output
        bin_preds = F.softmax(bin_preds, dim=1)

        bs, height, width = images.shape[0], images.shape[2], images.shape[3]

        bin_pred = bin_preds[0].data.cpu().numpy()
        ins_img = ins_preds[0].data.cpu().numpy()
        bin_img = bin_pred.argmax(0)
        bin_img = postprocessor.process(bin_img)
        lane_embedding_feats, lane_coordinate = get_lane_area(
            bin_img, ins_img)

        if lane_embedding_feats.size > 0:
            num_clusters, labels, cluster_centers = clustering.cluster(
                lane_embedding_feats, bandwidth=1.5)

            if genline_method:
                y_samples = list(range(height//2, height, 10))
                x_preds = output_lanes(num_clusters, labels, bin_pred[1],
                                lane_coordinate, y_samples,
                                method=genline_method)
                mask_img = draw_lane_mask(x_preds, y_samples, height, width)
            else:
                mask_img = get_lane_mask(num_clusters, labels, bin_img,
                                    lane_coordinate)

            mask_img = mask_img[:, :, (2, 1, 0)]
        else:
            logger.info('No lanes found!')
            mask_img = np.zeros((bin_img.shape[0], bin_img.shape[1], 3), np.uint8)

        overlay_img = cv2.addWeighted(image_pre, 1.0, mask_img, 1.0, 0)
        fps = 1.0/(time.time() - end)
        logger.info('fps={:.3f}'.format(fps))

        if show_image:
            # import here since current plt is not installed in current docker
            import matplotlib.pyplot as plt
            plt.ion()
            plt.figure('result')
            #plt.imshow(mask_img)
            plt.imshow(overlay_img)
            plt.show()
            plt.pause(10)

        return overlay_img


def main(opt):
    logger.info('Loading model: %s', opt.model_file)

    checkpoint = torch.load(opt.model_file)

    checkpoint_opt = checkpoint['opt']

    # Load model location
    model = LaneNet(cnn_type=checkpoint_opt.cnn_type)
    model = DataParallelModel(model)

    # Update/Overwrite some test options like batch size, location to metadata
    # file
    vars(checkpoint_opt).update(vars(opt))

    logger.info('Building model...')
    model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
        model = model.cuda()

    logger.info('Start testing...')
    test_video(
        model,
        opt.input_file,
        opt.output_file,
        checkpoint_opt.width,
        checkpoint_opt.height,
        genline_method=opt.genline_method
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_file',
        type=str,
        help='path to the model file')
    parser.add_argument(
        '--input_file',
        type=str,
        help='path to the test video')
    parser.add_argument(
        '--output_file',
        type=str,
        help='path to the output video')
    parser.add_argument(
        '--genline_method',
        choices=['polyfit', 'maxprob'],
        default=None,
        help='How to get the line output from the probability map')
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='number of workers (each worker use a process to load a batch of data)')
    opt = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')

    logger.info(
        'Input arguments: %s',
        json.dumps(
            vars(opt),
            sort_keys=True,
            indent=4))

    if not os.path.isfile(opt.model_file):
        logger.info('Model file does not exist: %s', opt.model_file)

    else:
        start = datetime.now()
        main(opt)
        logger.info('Time: %s', datetime.now() - start)
