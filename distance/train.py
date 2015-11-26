import os
import numpy as np
import pickle
import argparse

from chainer import optimizers
from chainer import cuda
from chainer import computational_graph as c

from tripletloss import triplet_loss
from models import EmbedNet
from data_loader import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('data', help='Path to training data')
parser.add_argument('--batchsize', '-b', type=int, default=12,
                    help='Learning minibatch size')
parser.add_argument('--epoch', '-e', default=50, type=int,
                    help='Number of epochs to learn')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='model',
                    help='Path to save model snapshots')
parser.add_argument('--interval', '-i', default=10, type=int,
                    help='Snapshot interval in epochs')
parser.add_argument('--resume', '-r', default=None,
                    help='Path to snapshots to continue from')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

batch_triplets = args.batchsize  # batchsize will be 3 * batch_triplets

dl = DataLoader(args.data, xp)

# model setup
if args.resume is None:
    model = EmbedNet()
else:
    print("resuming training on model {}".format(args.resume))
    model = pickle.load(open(args.resume, "rb"))

optimizer = optimizers.SGD()
optimizer.setup(model)

if args.gpu >= 0:
    model.to_gpu(args.gpu)

graph_generated = False

for epoch in range(1, args.epoch + 1):
    print('epoch', epoch)

    # training
    anchors = list(range(1, 4001))
    np.random.shuffle(anchors)

    sum_loss = 0
    iteration = 0
    for i in anchors:
        iteration += 1
        x_batch = dl.get_batch(i, batch_triplets)

        optimizer.zero_grads()
        loss = model.forward(x_batch)
        print("iteration {:04d}: loss {}".format(iteration, float(loss.data)), end='\r')

        loss.backward()
        optimizer.update()

        if not graph_generated:
            with open("graph.dot", "w") as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open("graph.wo_split.dot", "w") as o:
                g = c.build_computational_graph((loss, ),
                                                remove_split=True)
                o.write(g.dump())
            graph_generated = True
            print('graph generated')

        sum_loss += float(loss.data)

    print('train mean loss={}'.format(sum_loss / iteration))

    if epoch % args.interval == 0:
        snapshot_name = "{}_{}.pkl".format(args.out, epoch)
        snapshot_params = "{}_{}_params.pkl".format(args.out, epoch)
        print("saving snapshot to", snapshot_name)
        pickle.dump(model, open(snapshot_name, "wb"))
        pickle.dump(model.parameters, open(snapshot_params, "wb"))


    # evaluation -- later...
    # sum_accuracy = 0
    # sum_loss = 0
    # for i in range(N_test, N, batchsize):
    #     # x_batch = xp.asarray(data[i:i + batchsize])
    #     # y_batch = xp.asarray(labels[i:i + batchsize])
    #     x_batch = np.asarray(data[i:i+batchsize])
    #     y_batch = np.asarray(labels[i:i+batchsize])
    #
    #     loss, acc = forward(x_batch, y_batch, train=False)
    #
    #     sum_loss += float(loss.data) * len(y_batch)
    #     sum_accuracy += float(acc.data) * len(y_batch)
    #
    # print('test  mean loss={}, accuracy={}'.format(
    #     sum_loss / N_test, sum_accuracy / N_test))
