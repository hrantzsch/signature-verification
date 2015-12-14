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
