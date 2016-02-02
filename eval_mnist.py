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
from chainer import serializers

from models.tripletnet import TripletNet
from models.mnist_dnn import MnistDnn

parser = argparse.ArgumentParser()
parser.add_argument('data', help='Path to MNIST data. Should contain folders '
                    'train and test or files train.pkl and test.pkl.')
parser.add_argument('model', help='Trained DNN model')
args = parser.parse_args()


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
