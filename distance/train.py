import os
import numpy as np
import pickle
import argparse

import chainer
from chainer import optimizers
from chainer import computational_graph as c
from chainer import links as L

from tripletloss import triplet_loss
from models.tripletnet import TripletNet
from models.embednet import EmbedNet
from data_loader import DataLoader
from logger import Logger


def train_test_anchors(test_fraction, num_classes):
    t = int(num_classes * test_fraction)
    return list(range(1, num_classes+1))[:-t], list(range(1, num_classes+1))[-t:]


def write_graph(loss):
    with open("graph.dot", "w") as o:
        o.write(c.build_computational_graph((loss, )).dump())
    with open("graph.wo_split.dot", "w") as o:
        g = c.build_computational_graph((loss, ), remove_split=True)
        o.write(g.dump())
    print('graph generated')


parser = argparse.ArgumentParser()
parser.add_argument('data', help='Path to training data')

parser.add_argument('--batchsize', '-b', type=int, default=12,
                    help='Learning minibatch size')
parser.add_argument('--epoch', '-e', default=50, type=int,
                    help='Number of epochs to learn')
parser.add_argument('--test', '-t', default=0.1, type=float,
                    help='Fraction of samples to spare for testing (0.1)')

parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

parser.add_argument('--out', '-o', default='signdist',
                    help='Path to save model snapshots')
parser.add_argument('--log', '-l', default='signdist.log',
                    help='Log file')
parser.add_argument('--interval', '-i', default=10, type=int,
                    help='Snapshot interval in epochs')

parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')

args = parser.parse_args()


if args.gpu >= 0:
    from chainer import cuda
    cuda.check_cuda_available()
    xp = cuda.cupy
else:
    xp = np

batch_triplets = args.batchsize  # batchsize will be 3 * batch_triplets
dl = DataLoader(args.data, xp, num_classes=4000)
logger = Logger(args.log)


# model setup
model = EmbedNet()
if args.gpu >= 0:
    model.to_gpu(args.gpu)

optimizer = optimizers.AdaGrad(lr=0.005)
optimizer.setup(model)

if args.initmodel and args.resume:
    logger.load_snapshot(args.initmodel, args.resume, model, optimizer)

# clear Linear layer
model.out = L.Linear(1024, 128)
optimizer.prepare()

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
            write_graph(model.loss)
            graph_generated = True

    logger.log_mean("train")

    if optimizer.epoch % 15 == 0:
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
