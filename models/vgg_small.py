import chainer
import chainer.functions as F
import chainer.links as L


class VGGSmall(chainer.Chain):

    def __init__(self):
        super(VGGSmall, self).__init__(
            conv1_1=L.Convolution2D(1, 96, 11, stride=3, pad=1),
            conv1_2=L.Convolution2D(96, 128, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 5, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_4=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc1=L.Linear(1024, 256),
            fc2=L.Linear(256, 64)
        )

    def __call__(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = F.max_pooling_2d(h, 2, stride=2, pad=1),

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = F.max_pooling_2d(h, 2, stride=2, pad=1),

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        h = F.max_pooling_2d(h, 2, stride=2, pad=0),

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_4(h)
        h = F.max_pooling_2d(h, 2, stride=2, pad=0),

        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        h = F.max_pooling_2d(h, 2, stride=2, pad=0),

        h = self.fc1(h)
        h = self.fc2(h)

        return h
