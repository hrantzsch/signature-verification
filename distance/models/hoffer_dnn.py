import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np


class HofferDnn(chainer.Chain):
    """
    An embedding network modelled by the example of Hoffer and Ailon's
    TripletNet. Pooling stride has been ajdusted for larger input images.

    Hoffer, E., & Ailon, N. (2014).
    Deep metric learning using Triplet network.
    arXiv preprint arXiv:1412.6622.

    http://arxiv.org/abs/1412.6622
    https://github.com/eladhoffer/TripletNet/blob/master/Models/Model.lua
    """

    def __init__(self):
        super(HofferDnn, self).__init__(
            conv1=L.Convolution2D(1, 64, 5, stride=3, pad=1),
            conv2=L.Convolution2D(64, 128, 3),
            conv3=L.Convolution2D(128, 256, 3),
            conv4=L.Convolution2D(256, 128, (2, 3)),
            conv5=L.Convolution2D(128, 128, 2),
        )

    def __call__(self, x, train=True):
        h = self.conv1(x)
        h = F.max_pooling_2d(
            F.relu(h), (2, 3), (2, 3), (0, 1))
        h = F.dropout(h, train=train)
        h = F.max_pooling_2d(
            F.relu(self.conv2(h)), 2, 2, pad=(1, 0))
        h = F.dropout(h, train=train)
        h = F.max_pooling_2d(
            F.relu(self.conv3(h)), 2)
        h = F.dropout(h, train=train)
        h = F.relu(self.conv4(h))
        return F.relu(self.conv5(h))
