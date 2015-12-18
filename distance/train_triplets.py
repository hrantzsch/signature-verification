import os
import numpy as np
import pickle
import argparse

import chainer
from chainer import optimizers
from chainer import computational_graph as c
from chainer import links as L

import aux
from tripletloss import triplet_loss
from models.tripletnet import TripletNet
from models.embednet import EmbedNet
from data_loader import TripletLoader
from logger import Logger


def train_test_anchors(test_fraction, num_classes):
    t = int(num_classes * test_fraction)
    return list(range(1, num_classes+1))[:-t], list(range(1, num_classes+1))[-t:]

args = aux.get_args()

if args.gpu >= 0:
    from chainer import cuda
    cuda.check_cuda_available()
    xp = cuda.cupy
else:
    xp = np

batch_triplets = args.batchsize  # batchsize will be 3 * batch_triplets
dl = TripletLoader(args.data, xp, num_classes=4000)
logger = Logger(args.log)


# model setup
model = EmbedNet(4000*1080)
if args.gpu >= 0:
    model.to_gpu(args.gpu)

optimizer = optimizers.AdaGrad(lr=0.005)
optimizer.setup(model)

if args.initmodel and args.resume:
    logger.load_snapshot(args.initmodel, args.resume, model, optimizer)
elif args.initmodel:
    print("No resume state given -- finetuning on model " + args.initmodel)
    old_model = L.Classifier(DnnComponent())
    serializers.load_hdf5(args.initmodel, old_model)
    model.predictor.dnn.copyparams(old_model.predictor)

train, test = train_test_anchors(args.test, num_classes=dl.num_classes)

graph_generated = False
for _ in range(1, args.epoch + 1):
    optimizer.new_epoch()
    print('epoch', optimizer.epoch)

    # training
    np.random.shuffle(train)
    for i in train:
        x = chainer.Variable(dl.get_batch(i, batch_triplets))
        optimizer.update(model, x)
        logger.log_iteration("train", float(model.loss.data))

        if not graph_generated:
            aux.write_graph(model.loss)
            graph_generated = True

    logger.log_mean("train")

    if optimizer.epoch % 25 == 0:
        optimizer.lr *= 0.5
        print("learning rate decreased to {}".format(optimizer.lr))
    if optimizer.epoch % args.interval == 0:
        logger.make_snapshot(model, optimizer, optimizer.epoch, args.out)

    # testing
    for i in test:
        x = chainer.Variable(dl.get_batch(i, batch_triplets))
        loss = model(x, compute_acc=True)
        logger.log_iteration("test", float(model.loss.data), float(model.accuracy.data))
    logger.log_mean("test")
