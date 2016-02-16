import chainer
from chainer import functions as F

from functions.mse_zero_one import mse_zero_one
from functions.sqrt import sqrt


class TripletNet(chainer.Chain):
    """
    A triplet network remodelling the network proposed in
    Hoffer, E., & Ailon, N. (2014). Deep metric learning using Triplet network.

    The DNN to be used can be passed to the constructor, keeping it inter-
    changeable.
    """

    def __init__(self, dnn):
        super(TripletNet, self).__init__(
            dnn=dnn(),
        )

    def distance(self, x):
        """Compute anchor-positive distance and anchor-negative distance on a
           batch of triplets.
           The batch is forwarded through the network as a whole and then split
           to 3 batches of size n.
        """

        # forward batch through deep network
        h = self.dnn(x)
        h = F.reshape(h, (h.data.shape[0], h.data.shape[1]))

        # split to anchors, positives, and negatives
        anc, pos, neg = F.split_axis(h, 3, 0)

        # compute distances of anchor to positive and negative, respectively
        diff_pos = anc - pos
        diff_neg = anc - neg
        dist_pos = F.expand_dims(F.batch_l2_norm_squared(diff_pos), 1)
        dist_neg = F.expand_dims(F.batch_l2_norm_squared(diff_neg), 1)

        return F.concat((dist_pos, dist_neg))

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

        dist = sqrt(self.distance(x))

        # compute loss:
        # calculate softmax on distances as a ratio measure
        # loss is MSE of softmax to [0, 1] vector
        sm = F.softmax(dist)
        self.loss = mse_zero_one(sm)
        self.accuracy = (dist.data[..., 0] < dist.data[..., 1]).sum() / len(dist)

        return self.loss
