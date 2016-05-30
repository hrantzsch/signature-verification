import chainer
import chainer.functions as F
import chainer.links as L


class VGGSmallConv(chainer.Chain):
    """The convolutional part of VGGSmall, to be plugged under FC layers for
       either classifcation or tripletloss."""
    def __init__(self):
        super(VGGSmallConv, self).__init__(
            conv1_1=L.Convolution2D(1, 96, 11, stride=3, pad=1),
            conv1_2=L.Convolution2D(96, 96, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(96, 128, 5, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_4=L.Convolution2D(512, 512, 3, stride=1, pad=1)
        )

    def __call__(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = F.max_pooling_2d(h, 2, stride=2, pad=1)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = F.max_pooling_2d(h, 2, stride=2, pad=1)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_4(h)
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)

        return h


class VGGSmall(chainer.Chain):
    def __init__(self):
        super(VGGSmall, self).__init__(
            conv=VGGSmallConv(),
            fc1=L.Linear(4096, 1024),
            fc2=L.Linear(1024, 128)
        )

    def __call__(self, x):
        h = self.conv(x)
        h = self.fc1(h)
        h = self.fc2(h)
        return h
