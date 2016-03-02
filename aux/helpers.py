"""
Various auxilary methods
"""
import argparse
import numpy as np
from chainer import computational_graph as c


def write_graph(loss):
    with open("graph.dot", "w") as o:
        o.write(c.build_computational_graph((loss, )).dump())
    with open("graph.wo_split.dot", "w") as o:
        g = c.build_computational_graph((loss, ), remove_split=True)
        o.write(g.dump())
    print('graph generated')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Path to training data')

    parser.add_argument('--batchsize', '-b', type=int, default=12,
                        help='Learning minibatch size [12]')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='Number of epochs to learn [100]')
    parser.add_argument('--test', '-t', default=0.1, type=float,
                        help='Fraction of samples to spare for testing [0.1]')
    parser.add_argument('--skilled', '-s', default=0.5, type=float,
                        help='Fraction of hard triplets within each batch [0.5]')

    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU) [-1]')

    parser.add_argument('--interval', '-i', default=10, type=int,
                        help='Snapshot interval in epochs [10]')
    parser.add_argument('--lrinterval', '-l', default=10, type=int,
                        help='Interval for halving the LR [10]')
    parser.add_argument('--out', '-o', default='',
                        help='Name for snapshots and logging')
    parser.add_argument('--weight_decay', '-d', default=0.001, type=float,
                        help='Rate of weight decay regularization')

    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')

    return parser.parse_args()


def train_test_tuples(test_fraction, num_users):
    sign_per_user = 54
    sample_per_sign = 20
    t = int(test_fraction * num_users * sign_per_user * sample_per_sign)
    data = [(user, sign, sample)
            for user in range(1, num_users + 1)
            for sign in range(1, sign_per_user + 1)
            for sample in range(1, sample_per_sign + 1)]
    np.random.shuffle(data)
    return data[:-t], data[-t:]


def train_test_anchors(test_fraction, num_classes):
    t = int(num_classes * test_fraction)
    return list(range(1, num_classes+1))[:-t], list(range(1, num_classes+1))[-t:]
