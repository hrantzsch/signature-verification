import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np


class MnistWithLinear(chainer.Chain):

    def __init__(self):
        super(MnistWithLinear, self).__init__(
            dnn=MnistDnn(),
            out=L.Linear(64, 10),
        )

    def __call__(self, x):
        return self.out(self.dnn(x))


class MnistDnn(chainer.Chain):
    """
    An embedding network modelled by the example of Hoffer and Ailon's
    TripletNet. Pooling stride has been ajdusted for larger input images.
    -- Variation for MNIST

    Hoffer, E., & Ailon, N. (2014).
    Deep metric learning using Triplet network.
    arXiv preprint arXiv:1412.6622.

    http://arxiv.org/abs/1412.6622
    https://github.com/eladhoffer/TripletNet/blob/master/Models/Model.lua
    """

    def __init__(self):
        super(MnistDnn, self).__init__(
            conv1=L.Convolution2D(1, 32, 3),
            bn1=L.BatchNormalization(32),
            conv2=L.Convolution2D(32, 64, 2),
            bn2=L.BatchNormalization(64),
            conv3=L.Convolution2D(64, 128, 3),
            bn3=L.BatchNormalization(128),
            conv4=L.Convolution2D(128, 64, 2),
        )

    def __call__(self, x, train=True):

        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 2)

        h = self.bn2(self.conv2(h))
        h = F.max_pooling_2d(F.relu(h), 2)

        h = self.bn3(self.conv3(h))
        h = F.max_pooling_2d(F.relu(h), 2)

        h = F.relu(self.conv4(h))
        return h
