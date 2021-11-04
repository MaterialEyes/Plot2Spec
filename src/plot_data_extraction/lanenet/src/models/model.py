import numpy as np
import cv2
from sklearn.cluster import MeanShift
import torch.nn as nn
import logging

from .unet import UNet, UNetSCNN
from .deeplab import DeepLab
logger = logging.getLogger(__name__)


class LaneNet(nn.Module):

    def __init__(
            self,
            cnn_type='unet', embed_dim=4):
        """Load a LaneNet model based on the cnn_type
        """
        super(LaneNet, self).__init__()

        self.core = self.get_cnn(cnn_type, embed_dim)

    def get_cnn(self, cnn_type, embed_dim):
        """Load a LaneNet model based on the cnn_type
        """
        logger.info("===> Loading model '{}'".format(cnn_type))

        if cnn_type == 'unet':
            model = UNet(embed_dim=embed_dim)
        elif cnn_type == 'unetscnn':
            model = UNetSCNN(embed_dim=embed_dim)
        elif cnn_type == 'deeplab':
            model = DeepLab(embed_dim=embed_dim)
        else:
            raise ValueError('cnn_type unknown: %s', cnn_type)

        return model

    def forward(self, images):
        """Extract image feature vectors."""

        out = self.core(images)

        return out


class PostProcessor(object):

    def __init__(self):
        pass

    def process(self, image, kernel_size=5, minarea_threshold=10):
        """Do the post processing here. First the image is converte to grayscale.
        Then a closing operation is applied to fill empty gaps among surrounding
        pixels. After that connected component are detected where small components
        will be removed.

        Args:
            image:
            kernel_size
            minarea_threshold

        Returns:
            image: binary image

        """
        if image.dtype is not np.uint8:
            image = np.array(image, np.uint8)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # fill the pixel gap using Closing operator (dilation followed by
        # erosion)
        kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT, ksize=(
                kernel_size, kernel_size))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        ccs = cv2.connectedComponentsWithStats(
            image, connectivity=8, ltype=cv2.CV_32S)
        labels = ccs[1]
        stats = ccs[2]

        for index, stat in enumerate(stats):
            if stat[4] <= minarea_threshold:
                idx = np.where(labels == index)
                image[idx] = 0

        return image


class LaneClustering(object):

    def __init__(self):
        pass

    def cluster(self, embeddings, bandwidth=1.5):
        """Clustering pixel embedding into lanes using MeanShift

        Args:
            prediction: set of pixel embeddings
            bandwidth: bandwidth used in the RBF kernel

        Returns:
            num_clusters: number of clusters (or lanes)
            labels: lane labels for each pixel
            cluster_centers: centroids

        """
        ms = MeanShift(bandwidth, bin_seeding=True)
        try:
            ms.fit(embeddings)
        except ValueError as err:
            logger.error(err)
            return 0, [], []

        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        num_clusters = cluster_centers.shape[0]

        return num_clusters, labels, cluster_centers
