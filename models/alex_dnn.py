import chainer
import chainer.functions as F
import chainer.links as L


class AlexDNN(chainer.Chain):

    """Adjusted from AlexBN chainer example."""

    def __init__(self):
        super(AlexDNN, self).__init__(
            conv1=L.Convolution2D(1, 96, 10, stride=4, pad=1),
            bn1=L.BatchNormalization(96),
            conv2=L.Convolution2D(96, 256, (1, 2), stride=(1, 2)),
            bn2=L.BatchNormalization(256),
            conv3=L.Convolution2D(256, 384, 5, stride=2, pad=1),
            bn3=L.BatchNormalization(384),
            conv4=L.Convolution2D(384, 256, 3, pad=1),
            bn4=L.BatchNormalization(256),
            conv5=L.Convolution2D(256, 128, 3, stride=2, pad=1),
            bn5=L.BatchNormalization(128),
            conv6=L.Convolution2D(128, 64, 2),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x):
        self.clear()
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), (3, 5), stride=2)

        h = self.bn2(self.conv2(h), test=not self.train)
        h = F.relu(h)

        h = self.bn3(self.conv3(h), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2, pad=1)

        h = self.bn4(self.conv4(h), test=not self.train)
        h = F.relu(h)

        h = self.bn5(self.conv5(h), test=not self.train)
        h = F.relu(h)

        h = F.relu(self.conv6(h))
        return h
