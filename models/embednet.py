import chainer
import chainer.functions as F
import chainer.links as L

from functions.tripletloss import triplet_loss, triplet_accuracy
from functions.rint import rint
from models.embednet_dnn import DnnWithLinear
from models.alex_dnn import AlexDNN
from models.mnist_dnn import MnistDnn


class EmbedNet(chainer.Chain):
    """"""

    def __init__(self):
        super(EmbedNet, self).__init__(
            # dnn=DnnWithLinear(128),
            dnn=MnistDnn(),
            embed=L.EmbedID(128000, 64),
        )
        self._train = True

    def __call__(self, x, stop=False):
        """
        Forward through DNN, L2 normalization and embedding.
        Returns the triplet loss.

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

        # Perform L2 normalizationa and embedding
        h = F.batch_l2_norm_squared(h)
        h = self.embed(rint(h))

        # split to anchors, positives, and negatives
        anc, pos, neg = F.split_axis(h, 3, 0)
        # compute loss
        # self.loss = triplet_loss(anc, pos, neg)
        dist = (anc - pos)**2 - (anc - neg)**2
        self.dist = dist.data.sum() / len(dist.data)
        h = dist + 0.6

        import numpy as np
        zeros = chainer.Variable(np.zeros_like(h.data, dtype=np.float32))
        self.loss = F.mean_squared_error(-h, h)
        self.accuracy = ((anc - pos).data > (anc - neg).data).sum() / ((anc - pos).data > (anc - neg).data).size  # triplet_accuracy(anc, pos, neg).data

        # if stop:
        #     import pdb; pdb.set_trace()

        return self.loss
