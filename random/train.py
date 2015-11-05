import argparse

import numpy as np
import h5py

import chainer
from chainer import computational_graph as c
import chainer.functions as F
from chainer import cuda
from chainer import optimizers

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('hdf5data',
                    help='Path to data in hdf5 format')
args = parser.parse_args()

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np


# hyperparams
batchsize = 120
n_epoch = 20

h5data = h5py.File(args.hdf5data, 'r')
data = h5data['data']
labels = h5data['label']

N = data.shape[0]
N_train = int(N * 0.9)
N_test = int(N - N_train)

model = chainer.FunctionSet(
    conv1=F.Convolution2D(1,  6, 11),
    conv2=F.Convolution2D(6, 16, 8),
    conv3=F.Convolution2D(16, 30, 6),
    conv4=F.Convolution2D(30, 50,  3),
    conv5=F.Convolution2D(50, 24,  3),
    fc6=F.Linear(2856, 3072),
    fc7=F.Linear(3072, 3072),
    fc8=F.Linear(3072, 1),
)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()


def forward(x_data, y_data, train=True):
    x = chainer.Variable(x_data, volatile=not train)
    t = chainer.Variable(y_data, volatile=not train)

    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv1(x))), 2, stride=2)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv2(h))), 2, stride=2)
    h = F.relu(model.conv3(h))
    h = F.relu(model.conv4(h))
    h = F.max_pooling_2d(F.relu(model.conv5(h)), 3, stride=2)
    h = F.dropout(F.relu(model.fc6(h)))
    h = F.dropout(F.relu(model.fc7(h)))
    h = model.fc8(h)
    return F.softmax_cross_entropy(h, t), F.accuracy(h, t)


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
        loss, acc = forward(x_batch, y_batch)
        print("Iteration {}: Loss {}; Acc {}".format(i, float(loss.data), float(acc.data)))
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

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in range(N_test, N, batchsize):
        x_batch = xp.asarray(data[i:i+batchsize])
        y_batch = xp.asarray(labels[i:i+batchsize])

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
