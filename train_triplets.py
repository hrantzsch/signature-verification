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
from aux.triplet_loader import TripletLoader
from aux.logger import Logger
from aux.logger import load_snapshot
from models.tripletnet import TripletNet
from models.hoffer_dnn import HofferDnn
from models.alex_dnn import AlexDNN
# from models.embednet import EmbedNet
# from models.embednet_dnn import DnnComponent, DnnWithLinear

args = helpers.get_args()
NUM_CLASSES = 4000

xp = cuda.cupy if args.gpu >= 0 else np
dl = TripletLoader(xp)

model = TripletNet(AlexDNN)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    dl.use_device(args.gpu)
    model = model.to_gpu()

batch_triplets = args.batchsize  # batchsize will be 3 * batch_triplets

# optimizer = optimizers.SGD(lr=0.001)
# optimizer = optimizers.MomentumSGD(lr=0.01)
optimizer = optimizers.AdaGrad(lr=0.01)
optimizer.setup(model)

if args.initmodel and args.resume:
    load_snapshot(args.initmodel, args.resume, model, optimizer)
    print("Continuing from snapshot. LR: {}".format(optimizer.lr))
# elif args.initmodel:
#     print("No resume state given -- finetuning on model " + args.initmodel)
#     old_model = L.Classifier(DnnWithLinear(10))  # mimic pretrained model
#     serializers.load_hdf5(args.initmodel, old_model)  # load snapshot
#     model.dnn.dnn.copyparams(old_model.predictor.dnn)  # copy DnnComponent's params

logger = Logger(args, optimizer, args.out)

train, test = helpers.train_test_anchors(args.test, num_classes=NUM_CLASSES)

graph_generated = False
for _ in range(1, args.epoch + 1):
    optimizer.new_epoch()
    print('epoch', optimizer.epoch)

    # training
    dl.create_source('train', train, batch_triplets, args.data, skilled=args.skilled)
    dl.create_source('test', test, batch_triplets, args.data, skilled=args.skilled)

    for i in range(len(train)):
        x_data = dl.get_batch('train')
        x = chainer.Variable(x_data)
        optimizer.update(model, x)
        logger.log_iteration("train", float(model.loss.data), float(model.accuracy))

        if not graph_generated:
            helpers.write_graph(model.loss)
            graph_generated = True

    logger.log_mean("train")

    if optimizer.epoch in [2, 10, 25, 50]:
        optimizer.lr *= 0.5
        logger.mark_lr()
        print("learning rate decreased to {}".format(optimizer.lr))
    if optimizer.epoch % args.interval == 0:
        logger.make_snapshot(model)

    # testing
    for i in range(len(test)):
        x = chainer.Variable(dl.get_batch('test'))
        loss = model(x)
        logger.log_iteration("test", float(model.loss.data), float(model.accuracy))
    logger.log_mean("test")
