"""This script is used to evaluate the features extracted by triplet training
for the MNIST data set.

The trained DNN is used to extract features form the complete data set. After
that a simple Linear layer is trained and used for classification.

See also: Section 3.3 in
    Hoffer, E., & Ailon, N. (2014).
    Deep metric learning using Triplet network.
"""

import argparse
import numpy as np
import pickle
import os
from scipy.misc import imread

import chainer
from chainer import cuda
from chainer import links as L
from chainer import serializers
from chainer import optimizers

from models.tripletnet import TripletNet
from models.mnist_dnn import MnistDnn

parser = argparse.ArgumentParser()
parser.add_argument('data', help='Path to MNIST data. Should contain folders '
                    'train and test or files train.pkl and test.pkl.')
parser.add_argument('model', help='Trained DNN model')
parser.add_argument('--batches', '-e', default=500, type=int)
parser.add_argument('--batchsize', '-b', default=60, type=int)
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
xp = cuda.cupy if args.gpu >= 0 else np


def get_paths(directory):
    paths = []  # a tuple (label, sample path)
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        # skip files
        if not os.path.isdir(path):
            continue

        for sample in os.listdir(path):
            if '.png' not in sample:
                continue
            paths.append((f, os.path.join(path, sample)))
    return paths


def load_data(paths, save=False, fname='data.pkl'):
    # load the model
    model = TripletNet(dnn=MnistDnn)
    serializers.load_hdf5(args.model, model)
    embed = model.dnn

    data = {label: [] for label in range(10)}
    for label, path in paths:
        sample = imread(path).astype(np.float32).reshape((1, 1, 28, 28))
        sample /= 255.0
        data[int(label)].append(embed(chainer.Variable(sample)).data.reshape((1, 64,)))
    if save:
        pickle.dump(data, open(fname, 'wb'))
    return data


# load data either from images or from pkl file
# features will be a dict from class to lists of np.arrays
# {0: [ (1, 64), ... ], ... }
if 'train.pkl' in os.listdir(args.data) and \
   'test.pkl' in os.listdir(args.data):
    train_features = pickle.load(
        open(os.path.join(args.data, 'train.pkl'), 'rb'))
    test_features = pickle.load(
        open(os.path.join(args.data, 'test.pkl'), 'rb'))
else:
    train = os.path.join(args.data, 'train')
    test = os.path.join(args.data, 'test')
    if not (os.path.exists(train) and os.path.exists(test)):
        print("Erorr: Data dir should contain folders train and test.")
        raise FileNotFoundError

    # push data through DNN and save resulting features
    train_features = load_data(get_paths(train), save=True, fname='train.pkl')
    test_features = load_data(get_paths(test), save=True, fname='test.pkl')


# setup the network
model = L.Classifier(L.Linear(64, 10))
optimizer = chainer.optimizers.MomentumSGD(lr=0.001)
optimizer.setup(model)

# train
for batch_num in range(args.batches):
    # we'll just assume that we get enough of them after a while

    labels = np.random.choice(10, args.batchsize)
    choose_random = lambda l: \
        train_features[l][np.random.choice(len(train_features[l]))]
    batch = [choose_random(l) for l in labels]

    t = chainer.Variable(xp.array(labels, dtype=xp.int32))
    x = chainer.Variable(xp.array(batch, dtype=xp.float32))

    optimizer.update(model, x, t)

    print("acc: {}".format(model.accuracy.data))

    if batch_num % 2000 == 0:
        optimizer.lr /= 2
