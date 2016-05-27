"""A script to train a triplet distance based model.

This script is used to train models that expect to be fed triplets for
training. Currently it can be used to train EmbedNet, based on the FaceNet
paper, and TripletNet, based on Hoffer, Ailon: "Deep Metric Learning Using
Triplet Network".
"""

import os
import numpy as np
import pickle
import argparse
import time

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import computational_graph as c
from chainer import links as L
from chainer import serializers

from tripletembedding.predictors import TripletNet
from tripletembedding.aux import Logger, load_snapshot
from tripletembedding.models import SmallDnn

from aux import helpers
from aux.triplet_loader import TripletLoader
from aux.mcyt_loader import McytLoader


args = helpers.get_args()
NUM_CLASSES = 99  # TODO HACK -- first class is never seen

xp = cuda.cupy if args.gpu >= 0 else np
dl = McytLoader(xp)

model = TripletNet(SmallDnn)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    dl.use_device(args.gpu)
    model = model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=0.01)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

if args.initmodel and args.resume:
    load_snapshot(args.initmodel, args.resume, model, optimizer)
    print("Continuing from snapshot. LR: {}".format(optimizer.lr))
logger = Logger(args, optimizer, args.out)

train, test = helpers.train_test_anchors(args.test, num_classes=NUM_CLASSES)

graph_generated = False
for _ in range(1, args.epoch + 1):
    time_started = time.time()

    optimizer.new_epoch()
    print('========\nepoch', optimizer.epoch)

    # training
    dl.create_source('train',
                     train, args.batchsize, args.data, skilled=args.skilled)
    dl.create_source('test',
                     test, args.batchsize, args.data, skilled=args.skilled)

    for i in range(len(train)):
        model.clean()
        x_data = dl.get_batch('train')
        x = chainer.Variable(x_data)
        optimizer.update(model, x)
        logger.log_iteration("train",
                             float(model.loss.data), float(model.accuracy),
                             float(model.mean_diff), float(model.max_diff))

        if not graph_generated:
            helpers.write_graph(model.loss)
            graph_generated = True

    logger.log_mean("train")
    print("iteration time:\t{:.3f} sec"
          .format((time.time() - time_started) / len(train)))

    if optimizer.epoch % args.lrinterval == 0 and optimizer.lr > 0.000001:
        optimizer.lr *= 0.5
        logger.mark_lr()
        print("learning rate decreased to {}".format(optimizer.lr))
    if optimizer.epoch % args.interval == 0:
        logger.make_snapshot(model)

    # testing
    for i in range(len(test)):
        x = chainer.Variable(dl.get_batch('test'))
        loss = model(x)
        logger.log_iteration("test",
                             float(model.loss.data), float(model.accuracy),
                             float(model.mean_diff), float(model.max_diff))
    logger.log_mean("test")

# make final snapshot if not just taken one
if optimizer.epoch % args.interval != 0:
    logger.make_snapshot(model)
