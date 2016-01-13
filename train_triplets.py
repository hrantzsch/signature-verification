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

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import computational_graph as c
from chainer import links as L
from chainer import serializers

from aux import helpers
from aux.data_loader import DataLoader
from aux.logger import Logger
from functions.tripletloss import triplet_loss
from models.tripletnet import TripletNet
from models.hoffer_dnn import HofferDnn
from models.embednet import EmbedNet
from models.embednet_dnn import DnnComponent, DnnWithLinear


def train_test_anchors(test_fraction, num_classes):
    t = int(num_classes * test_fraction)
    return list(range(1, num_classes+1))[:-t], list(range(1, num_classes+1))[-t:]

args = helpers.get_args()
NUM_CLASSES = 200

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
xp = cuda.cupy if args.gpu >= 0 else np


batch_triplets = args.batchsize  # batchsize will be 3 * batch_triplets
dl = DataLoader(args.data, xp)
logger = Logger(args.log)


# model setup
model = TripletNet()

if args.gpu >= 0:
    model = model.to_gpu()

optimizer = optimizers.SGD()
# optimizer = optimizers.AdaGrad(lr=0.005)
optimizer.setup(model)

if args.initmodel and args.resume:
    logger.load_snapshot(args.initmodel, args.resume, model, optimizer)
elif args.initmodel:
    print("No resume state given -- finetuning on model " + args.initmodel)
    old_model = L.Classifier(DnnWithLinear(10))  # mimic pretrained model
    serializers.load_hdf5(args.initmodel, old_model)  # load snapshot
    model.dnn.dnn.copyparams(old_model.predictor.dnn)  # copy DnnComponent's params

train, test = train_test_anchors(args.test, num_classes=NUM_CLASSES)

graph_generated = False
for _ in range(1, args.epoch + 1):
    optimizer.new_epoch()
    print('epoch', optimizer.epoch)

    # training
    np.random.shuffle(train)
    dl.prepare_triplet_provider(train, batch_triplets, NUM_CLASSES)
    for i in range(len(train)):
        x_data = dl.get_batch() / 255.0
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
    dl.prepare_triplet_provider(test, batch_triplets, NUM_CLASSES)
    for i in range(len(test)):
        x = chainer.Variable(dl.get_batch() / 255.0)
        loss = model(x, compute_acc=True)
        logger.log_iteration("test", float(model.loss.data), float(model.accuracy))
    logger.log_mean("test")
