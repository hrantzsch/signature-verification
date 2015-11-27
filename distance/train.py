import os
import numpy as np
import pickle
import argparse

import chainer
from chainer import optimizers, cuda, serializers
from chainer import computational_graph as c

from tripletloss import triplet_loss
from embednet import EmbedNet
from data_loader import DataLoader


def make_snapshot(model, optimizer, epoch, name):
    serializers.save_hdf5('{}_{}.model'.format(name, epoch), model)
    serializers.save_hdf5('{}_{}.state'.format(name, epoch), optimizer)
    print("snapshot created")


def train_test_anchors(test_fraction, num_classes=10):
    t = int(num_classes * test_fraction)
    return list(range(1, num_classes+1))[:t], list(range(1, num_classes+1))[-t:]


def run_batch(model, optimizer, dl, anchor_id, train=True):
    volatile = 'off' if model.train else 'on'
    x = chainer.Variable(dl.get_batch(i, batch_triplets))
    if train:
        optimizer.update(model, x)
    else:
        model(x, compute_acc=True)


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
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='signdist',
                    help='Path to save model snapshots')
parser.add_argument('--interval', '-i', default=10, type=int,
                    help='Snapshot interval in epochs')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--test', '-t', default=0.1, type=float,
                    help='Fraction of samples to spare for testing (0.1)')
args = parser.parse_args()


if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

batch_triplets = args.batchsize  # batchsize will be 3 * batch_triplets

dl = DataLoader(args.data, xp)

# model setup
model = EmbedNet()
if args.gpu >= 0:
    model.to_gpu(args.gpu)

optimizer = optimizers.SGD()
optimizer.setup(model)

if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_hdf5(args.resume, optimizer)


train, test = train_test_anchors(args.test)

graph_generated = False
for epoch in range(1, args.epoch + 1):
    print('epoch', epoch)

    # training
    np.random.shuffle(train)
    sum_loss = 0
    iteration = 0
    for i in train:
        iteration += 1
        run_batch(model, optimizer, dl, i, train=True)
        print("iteration {:04d}: loss {}".format(
            iteration, float(model.loss.data)), end='\r')
        sum_loss += float(model.loss.data)

        if not graph_generated:
            write_graph(model.loss)
            graph_generated = True

    print('train mean loss={}'.format(sum_loss / iteration))

    if epoch % args.interval == 0:
        make_snapshot(model, optimizer, epoch, args.out)

    # testing
    sum_accuracy = 0
    sum_loss = 0
    iteration = 0
    for i in test:
        iteration += 1
        run_batch(model, optimizer, dl, i, train=False)

        sum_loss += float(model.loss.data)
        # sum_accuracy += float(model.accuracy.data)
        sum_accuracy += float(model.accuracy.data)

    print('test mean loss={}, accuracy={}'.format(
        sum_loss / iteration, sum_accuracy / iteration))
