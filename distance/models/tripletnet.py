import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L

import numpy as np

from .hoffer_dnn import HofferDnn
from .dnn import DnnComponent
from distance.l2_distance_squared import l2_distance_squared


class TripletNet(chainer.Chain):
    """
    A triplet network remodelling the network proposed in
    Hoffer, E., & Ailon, N. (2014). Deep metric learning using Triplet network.

    The DNN to be used can be passed to the constructor, keeping it inter-
    changeable.
    """

    def __init__(self, dnn=HofferDnn):
        super(TripletNet, self).__init__()
        # TODO accept parameters for other DNNs
        self.dnn = dnn()

    def __call__(self, x, compute_acc=False):
        """
        Forward through DNN and compute loss and accuracy.

        x is a batch of size 3n following the form:

        | anchor_1   |
        | [...]      |
        | anchor_n   |
        | positive_1 |
        | [...]      |
        | positive_n |
        | negative_1 |
        | [...]      |
        | negative_n |
        """

        # The batch is forwarded through the network as a whole and then split
        # to 3 batches of size n, which are the input for the triplet_loss

        # forward batch through deep network
        h = self.dnn(x)
        h = x

        # split to anchors, positives, and negatives
        anc, pos, neg = F.split_axis(h, 3, 0)
        n = anc.data.shape[0]

        # compute distances of anchor to positive and negative, respectively
        dist_pos = F.reshape(l2_distance_squared(anc, pos), (n, 1))
        dist_neg = F.reshape(l2_distance_squared(anc, neg), (n, 1))
        dist = F.concat((dist_pos, dist_neg))

        # compute loss:
        # calculate softmax on distances as a ratio measure
        # loss is MSE of softmax to [0, 1] vector
        sm = F.softmax(dist)
        xp = cuda.get_array_module(sm)
        zero_one = xp.array([0, 1] * n, dtype=sm.data.dtype).reshape(n, 2)

        return F.mean_squared_error(sm, chainer.Variable(zero_one))
