"""Train a classifier.

This script trains a classifier, that can be used as a base model for triplet-
loss training. Of the trained classifier only the convolutional layers should
be saved, while the final fully connected layers will be dropped and replaced
in the tripletloss training.
"""
import argparse
import numpy as np

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import functions as F
from chainer import links as L

from aux import helpers
from aux.labelled_loader import LabelledLoader

from tripletembedding.aux import Logger, load_snapshot

from models.vgg_small import VGGSmallConv


class VGGClf(chainer.Chain):
    """Classifying FC layers on top of conv layers"""
    def __init__(self, num_classes):
        super(VGGClf, self).__init__(
            conv=VGGSmallConv(),
            fc1=L.Linear(4096, 1024),
            fc2=L.Linear(1024, num_classes)
        )
        self.predict = False

    def __call__(self, x, t):
        h = self.conv(x)
        h = self.fc1(h)
        h = self.fc2(h)

        if not self.predict:
            self.loss = F.softmax_cross_entropy(h, t)
            self.accuracy = F.accuracy(h, t)
            return self.loss
        else:
            self.pred = F.softmax(h)
            return self.pred


if __name__ == '__main__':
    args = helpers.get_args()

    model = VGGClf(4000)  # TODO provide parameter
    xp = cuda.cupy if args.gpu >= 0 else np
    dl = LabelledLoader(xp)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        dl.use_device(args.gpu)
        model = model.to_gpu()

    optimizer = optimizers.MomentumSGD(lr=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    if args.initmodel and args.resume:
        load_snapshot(args.initmodel, args.resume, model, optimizer)
        print("Continuing from snapshot. LR: {}".format(optimizer.lr))
        logger = Logger(args, optimizer, args.out)

    logger = Logger(args, optimizer, args.out)

    for _ in range(1, args.epoch + 1):
        optimizer.new_epoch()
        model.zerograds()

        print('========\nepoch', optimizer.epoch)

        # training
        dl.create_sources(args.data, args.batchsize, 0.9)

        while True:
            data = dl.get_batch('train')
            if data is None:
                break
            t_data, x_data = data
            x = chainer.Variable(x_data)
            t = chainer.Variable(t_data)
            optimizer.update(model, x, t)
            logger.log_iteration("train", float(model.loss.data),
                                 float(model.accuracy.data), 0.0, 0.0)

        if optimizer.epoch % args.lrinterval == 0 and optimizer.lr > 0.000001:
            optimizer.lr *= 0.5
            logger.mark_lr()
            print("learning rate decreased to {}".format(optimizer.lr))
        if optimizer.epoch % args.interval == 0:
            logger.make_snapshot(model)

        # testing
        for _ in range(50):
            data = dl.get_batch('test')
            if data is None:
                break
            t_data, x_data = data
            x = chainer.Variable(x_data, volatile=True)
            t = chainer.Variable(t_data, volatile=True)
            loss = model(x, t)
            logger.log_iteration("test", float(model.loss.data),
                                 float(model.accuracy.data), 0.0, 0.0)
        logger.log_mean("test")

    # make final snapshot if not just taken one
    if optimizer.epoch % args.interval != 0:
        logger.make_snapshot(model)
