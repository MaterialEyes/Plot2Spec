import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, BatchNorm):
        super(outconv, self).__init__()
        # self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = nn.Sequential(nn.Conv2d(in_ch, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, out_ch, kernel_size=1, stride=1))

    def forward(self, x):
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim=4, num_classes=2, backbone='drn',
                 BatchNorm=SynchronizedBatchNorm2d, use_hnet=True):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.use_hnet = use_hnet
        if self.use_hnet:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hnet = nn.Linear(low_level_inplanes, 6)

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.sem_out = outconv(304, num_classes, BatchNorm)
        self.ins_out = outconv(304, embed_dim, BatchNorm)
        self._init_weight()


    def forward(self, x, low_level_feat):

        if self.use_hnet:
            pooled_feat = self.avgpool(low_level_feat)
            pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)
            hnet_pred = self.hnet(pooled_feat)

        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        # dim1 = 256 + 48 = 304
        x = torch.cat((x, low_level_feat), dim=1)
        sem = self.sem_out(x)
        ins = self.ins_out(x)

        if self.use_hnet:
            return sem, ins, hnet_pred
        else:
            return sem, ins


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(embed_dim, num_classes, backbone, BatchNorm):
    return Decoder(embed_dim, num_classes, backbone, BatchNorm)
