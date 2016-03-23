"""Load a model and visualize the DNNs embedding"""

import sys
import pickle
import os
from itertools import chain
from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

import chainer
from chainer import links as L
from chainer import serializers

from models.tripletnet import TripletNet
from models.mnist_dnn import MnistDnn

from mpl_toolkits.mplot3d import Axes3D


def print_usage():
    print("usage: {} embedded_data_dir".format(sys.argv[0]))


def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)


def load_data_mnist(paths, save=False, fname='mnist.pkl'):
    # load the model
    model = TripletNet(dnn=MnistDnn)
    serializers.load_hdf5(sys.argv[1], model)
    embed = model.dnn

    data = {label: [] for label in range(10)}
    for label, path in paths:
        sample = imread(path).astype(np.float32).reshape((1, 1, 28, 28))
        sample /= 255.0
        data[int(label)].append(embed(chainer.Variable(sample)).data.reshape((1, 64,)))
    if save:
        pickle.dump(data, open(fname, 'wb'))
    return data


def plot_pca(pca):
    import pdb; pdb.set_trace()
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#aaaaaa', '#ffa500', '#A52A2A']
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    ax = Axes3D(fig)
    for i in range(10):
        start = sum(map(lambda x: len(data[x]), range(i)))
        end = start + len(data[i])
        # ax.scatter(pca.Y[start:end, 0], pca.Y[start:end, 1], pca.Y[start:end, 2],
        #            marker='.', color=colors[i])
        ax.plot(pca.Y[start:end, 0], pca.Y[start:end, 1], # pca.Y[start:end, 2],
                '.', markersize=5, alpha=0.1, color=colors[i], label=i)
    plt.legend(numpoints=1)
    # plt.xlabel('x_values')
    # plt.ylabel('y_values')
    # plt.xlim([0,10])
    # plt.ylim([-4,4])
    plt.savefig('pca.png')
    plt.show()


if __name__ == '__main__':

    if not len(sys.argv) == 2:
        print_usage()
        exit(0)
    data = []
    count = 0  # limit amount of data...
    for f in os.listdir(sys.argv[1]):
        if not '.pkl' in f:
            continue
        fn = os.path.join(sys.argv[1], f)
        data.append(pickle.load(open(fn, 'rb')))
        count += 1
        if count > 1000:
            break
    data = np.concatenate(data)
    mean = data.mean(axis=0)
    cleaned = np.delete(data, np.where(mean == 0), 1)
    pca = PCA(cleaned)
    plot_pca(pca)

# MNIST sample paths
# TRAIN_PKL = '/home/hannes/Data/mnist/train_paths.pkl'
# TEST_PKL  = '/home/hannes/Data/mnist/test_paths.pkl'
# paths = flatten(pickle.load(open(TRAIN_PKL, 'rb')))
# paths_test = flatten(pickle.load(open(TRAIN_PKL, 'rb')))
# data = load_data(chain(paths, paths_test), True, "mnist_16-02-17.pkl")
