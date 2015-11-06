import argparse

import numpy as np
import h5py
import pickle

import chainer
from chainer import computational_graph as c
import chainer.functions as F
from chainer import cuda
from chainer import optimizers

from alex import Alex

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('hdf5data',
                    help='Path to data in hdf5 format')
parser.add_argument('--batchsize', '-b', type=int, default=120,
                    help='Learning minibatch size')
parser.add_argument('--epoch', '-e', default=10, type=int,
                    help='Number of epochs to learn')
parser.add_argument('--out', '-o', default='model',
                    help='Path to save model on each validation')
parser.add_argument('--snapshotInterval', '-s', default=0,
                    help='Snapshot interval in epochs (0 for no snapshots)')
args = parser.parse_args()

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# hyperparams
batchsize = args.batchsize
n_epoch = args.epoch

h5data = h5py.File(args.hdf5data, 'r')
data = h5data['data']
labels = h5data['label']

N = data.shape[0]
N_train = int(N * 0.9)
N_test = int(N - N_train)

model = Alex()

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learning loop
for epoch in range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N_train, batchsize):
        x_batch = xp.asarray(data[i:i+batchsize])
        y_batch = xp.asarray(labels[i:i+batchsize], dtype=np.int32)

        optimizer.zero_grads()
        loss, acc = model.forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        if epoch == 1 and i == 0:
            with open("graph.dot", "w") as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open("graph.wo_split.dot", "w") as o:
                g = c.build_computational_graph((loss, ),
                                                remove_split=True)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))
    if args.snapshotInterval > 0 and epoch % args.snapshotInterval == 0:
        pickle.dump(model, open(args.out + '_snap{}'.format(epoch), 'wb'), -1)

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in range(N_test, N, batchsize):
        x_batch = xp.asarray(data[i:i+batchsize])
        y_batch = xp.asarray(labels[i:i+batchsize], dtype=np.int32)

        loss, acc = model.forward(x_batch, y_batch, train=False)

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))

# Save final model
pickle.dump(model, open(args.out, 'wb'), -1)
