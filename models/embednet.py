import chainer
import chainer.functions as F
import chainer.links as L

from functions.tripletloss import triplet_loss, triplet_accuracy
from models.embednet_dnn import DnnWithLinear


class EmbedNet(chainer.Chain):
    """"""

    def __init__(self, embed_size):
        super(EmbedNet, self).__init__(
            dnn=DnnWithLinear(128),
            embed=L.EmbedID(embed_size, 128),
        )
        self._train = True

    def __call__(self, x, compute_acc=False):
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
        # TODO scaling?
        h = self.embed(h)

        # split to anchors, positives, and negatives
        anc, pos, neg = F.split_axis(h, 3, 0)

        # compute loss
        self.loss = triplet_loss(anc, pos, neg)
        if compute_acc:
            self.accuracy = triplet_accuracy(anc, pos, neg)

        return self.loss

    def verify(self, x):
        """
        Forward two samples through network and embed them.
        Returns the Eucledean distance.
        """

        # forward and embed
        h = self.forward_dnn(x)
        h = self.forward_embed(h)
        a, b = F.split_axis(h, 2, 0)

        return F.mean_squared_error(a, b)
