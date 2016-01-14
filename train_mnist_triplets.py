"""A script to train a triplet distance based model of MNIST data.

Very much tailored to mnist...
"""

import os
import numpy as np
import pickle
import argparse

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import computational_graph as c
from chainer import links as L

from aux import helpers
from aux.mnist_loader import MnistLoader
from aux.logger import Logger
from models.tripletnet import TripletNet
from models.mnist_dnn import MnistDnn


args = helpers.get_args()

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
xp = cuda.cupy if args.gpu >= 0 else np

dl = MnistLoader(xp)
logger = Logger(args.log)
model = TripletNet(dnn=MnistDnn)

if args.gpu >= 0:
    model = model.to_gpu()

optimizer = optimizers.SGD()
optimizer.setup(model)

train = list(range(10))
test = list(range(10))
graph_generated = False
for _ in range(1, args.epoch + 1):
    optimizer.new_epoch()
    print('epoch', optimizer.epoch)

    np.random.shuffle(train)
    np.random.shuffle(test)

    # training
    for anchor in train:
        x_data = dl.get_batch(anchor, args.batchsize)
        x = chainer.Variable(x_data)
        optimizer.update(model, x)
        logger.log_iteration("train", float(model.loss.data))

        if not graph_generated:
            helpers.write_graph(model.loss)
            graph_generated = True

    logger.log_mean("train")

    if optimizer.epoch % 25 == 0:
        optimizer.lr *= 0.5
        print("learning rate decreased to {}".format(optimizer.lr))
    if optimizer.epoch % args.interval == 0:
        logger.make_snapshot(model, optimizer, optimizer.epoch, args.out)

    # testing
    for anchor in test:
        x_data = dl.get_batch(anchor, args.batchsize, train=False)
        x = chainer.Variable(x_data)
        loss = model(x, compute_acc=True)
        logger.log_iteration("test", float(model.loss.data), float(model.accuracy))
    logger.log_mean("test")
