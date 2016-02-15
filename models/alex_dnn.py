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
            conv3=L.Convolution2D(256, 256, 5, stride=2, pad=1),
            bn2=L.BatchNormalization(256),
            conv4=L.Convolution2D(256, 384, 3, pad=1),
            conv5=L.Convolution2D(384, 384, 3, stride=2, pad=1),
            bn3=L.BatchNormalization(384),
            conv6=L.Convolution2D(384, 256, 3, pad=1),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x):
        self.clear()
        h = self.conv1(x)
        h = self.bn1(F.relu(h), test=not self.train)
        h = F.max_pooling_2d(h, (3, 5), stride=2)
        h = self.conv2(h)  # additional layer adjust aspect ratio
        h = self.conv3(h)
        h = self.bn2(F.relu(h), test=not self.train)
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = self.bn3(h, test=not self.train)
        h = F.max_pooling_2d(F.relu(self.conv6(h)), 2, stride=1)
        return h
