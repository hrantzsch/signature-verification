import os
import numpy as np
from scipy.misc import imread
import pickle
import argparse

from chainer import optimizers
from chainer import cuda
from chainer import computational_graph as c

from tripletloss import triplet_loss
from models import EmbedNet


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


def get_signature_path(person, sign_num):
    directory = os.path.join(args.data, "{:03d}".format(person))
    if sign_num > 24:  # a forgery
        prefix = "cf"
        sign_num -= 24
    else:
        prefix = "c"
    fname = "{}-{:03d}-{:02d}.jpg".format(prefix, person, sign_num)
    return os.path.join(directory, fname)


def load_image(person, sign_num):
    path = get_signature_path(person, sign_num)
    return imread(path).astype(xp.float32)[xp.newaxis, ...]


def get_batch(anchor_id, num_triplets):
    """Make a batch using person <anchor_id> as anchor."""
    anchor_samples = list(range(1, 25))
    np.random.shuffle(anchor_samples)

    # pop anchor_sample, REMOVING it from the remaining anchor_samples
    anchor_sample = anchor_samples.pop()

    neg_ids = list(range(1, 4001))
    neg_ids.remove(anchor_id)
    # allow use of 24 signatures and 30 forgeries of the negatives
    neg_samples = [(np.random.choice(neg_ids),
                    np.random.choice(list(range(1, 55))))
                   for i in range(num_triplets)]

    # repeat anchor sample
    a = xp.array([load_image(anchor_id, anchor_sample)] * num_triplets,
                 dtype=xp.float32)
    # generate <num_triplets> p's randomly sampled from remaining anchor_samples
    p = xp.array([load_image(anchor_id, np.random.choice(anchor_samples))
                  for _ in range(num_triplets)],
                 dtype=xp.float32)
    # negative samples from remaining neg_ids
    n = xp.array([load_image(np.random.choice(neg_ids), np.random.choice(list(range(1, 55))))
                  for _ in range(num_triplets)],
                 dtype=xp.float32)
    return xp.concatenate([a, p, n])


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
    anchors = list(range(1, 3))
    np.random.shuffle(anchors)

    sum_loss = 0
    iteration = 0
    for i in anchors:
        iteration += 1
        x_batch = get_batch(i, batch_triplets)

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
        print("saving snapshot to", snapshot_name)
        pickle.dump(model, open(snapshot_name, "wb"))


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
