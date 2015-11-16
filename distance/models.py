import chainer
import chainer.functions as F

import numpy as np

from tripletloss import triplet_loss


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

    def forward_batch(self, x, train=True):
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
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = self.out(h)

        return h

    def embed_batch(self, x):
        """Perform normalization and embed each sample in the given batch in a
           128 dimensional Euclidean space"""

        x_data = np.stack([sample.dot(sample) / sample.size * self.embed_size
                           for sample in x.data]).astype(np.int32)
        return self.embed(chainer.Variable(x_data))

    def forward(self, x_data, train=True):

        """"""

        # x_data is a batch of size 3n following the form:
        #
        # | anchor_1   |
        # | [...]      |
        # | anchor_n   |
        # | positive_1 |
        # | [...]      |
        # | positive_n |
        # | negative_1 |
        # | [...]      |
        # | negative_n |
        #
        # the batch can be forwarded through the network as a whole and split
        # afterwards to 3 batches of size n, which are the input for the
        # triplet_loss

        # forward batch through deep network
        x = chainer.Variable(x_data, volatile=not train)
        out = self.forward_batch(x, train)

        embedded = self.embed_batch(out)

        # split to anchors, positives, and negatives
        n = embedded.data.shape[0] / 3
        anc, pos, neg = [chainer.Variable(batch)
                         for batch in np.split(embedded.data, [n, n*2])]

        # compute loss
        return triplet_loss(anc, pos, neg)
