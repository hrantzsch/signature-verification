import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np


class MnistWithLinear(chainer.Chain):

    def __init__(self):
        super(MnistWithLinear, self).__init__(
            dnn=MnistDnn(),
            out=L.Linear(128, 128),
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
            conv2=L.Convolution2D(32, 64, 2),
            conv3=L.Convolution2D(64, 128, 3),
            conv4=L.Convolution2D(128, 128, 2),
        )

    def __call__(self, x, train=True):
        h = F.max_pooling_2d(
            F.relu(self.conv1(x)), 2)
        h = F.dropout(h, train=train)
        h = F.max_pooling_2d(
            F.relu(self.conv2(h)), 2)
        h = F.dropout(h, train=train)
        h = F.max_pooling_2d(
            F.relu(self.conv3(h)), 2)
        h = F.dropout(h, train=train)
        h = F.relu(self.conv4(h))
        return h
