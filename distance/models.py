import chainer
import chainer.functions as F
from chainer.utils import type_check

import numpy as np

from tripletloss import triplet_loss
from l2normalization import l2_normalization


class EmbedNet(chainer.FunctionSet):
    """New GoogLeNet of BatchNormalization version.
       Adapted from the example provided with chainer."""

    embed_size = 128000

    def __init__(self):
        super(EmbedNet, self).__init__(
            conv1=F.Convolution2D(1, 64, 7, stride=2, pad=3, nobias=True),
            norm1=F.BatchNormalization(64),
            conv2=F.Convolution2D(64, 192, 3, pad=1, nobias=True),
            norm2=F.BatchNormalization(192),
            inc3a=F.InceptionBN(192, 64, 64, 64, 64, 96, 'avg', 32),
            inc3b=F.InceptionBN(256, 64, 64, 96, 64, 96, 'avg', 64),
            inc3c=F.InceptionBN(320, 0, 128, 160, 64, 96, 'max', stride=2),
            inc4a=F.InceptionBN(576, 224, 64, 96, 96, 128, 'avg', 128),
            inc4b=F.InceptionBN(576, 192, 96, 128, 96, 128, 'avg', 128),
            inc4c=F.InceptionBN(576, 128, 128, 160, 128, 160, 'avg', 128),
            inc4d=F.InceptionBN(576, 64, 128, 192, 160, 192, 'avg', 128),
            inc4e=F.InceptionBN(576, 0, 128, 192, 192, 256, 'max', stride=2),
            inc5a=F.InceptionBN(1024, 352, 192, 320, 160, 224, 'avg', 128),
            inc5b=F.InceptionBN(1024, 352, 192, 320, 192, 224, 'max', 128),
            out=F.Linear(1024, 128),

            embed=F.EmbedID(self.embed_size, 128)  # openface uses 128 dimensions
        )

    def forward_dnn(self, x):
        """Forward a batch images through the network"""

        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(x))), 3, stride=2, pad=1)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h))), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)

        h = self.inc4a(h)
        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)
        h = self.inc4e(h)

        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), (3, 7))
        h = self.out(h)

        return h

    def embed(self, x):
        """Perform L2 normalizationa and embedding"""

        norm = l2_normalization(x, scale=100)
        return self.embed(norm)

    def forward(self, x_data, train=True):
        """
        Forward through DNN, L2 normalization and embedding.
        Returns the triplet loss.

        x_data is a batch of size 3n following the form:

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
        x = chainer.Variable(x_data, volatile=not train)
        h = self.forward_dnn(x)
        h = self.embed(h)

        # split to anchors, positives, and negatives
        anc, pos, neg = F.split_axis(h, 3, 0)

        # compute loss
        return triplet_loss(anc, pos, neg)

    def verify(self, x_data):
        """
        Forward two samples through network and embed them.
        Returns the Eucledean distance.
        """

        # Expect exactly two samples in the batch
        import pdb; pdb.set_trace()
        
        type_check.expect(
            x_data.dtype == np.float32,
            x_data.ndim >= 2,
        )

        x = chainer.Variable(x_data, volatile=True)

        # forward and embed
        h = self.forward_dnn(x)
        h = self.embed(h)
        a, b = F.split_axis(h, 2, 0)

        return F.mean_squared_error(a, b)
