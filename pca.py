"""Load a model and visualize the DNNs embedding"""

import sys
import pickle
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


# aux
def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)


def load_data(paths, save=False, fname='data.pkl'):
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


# MNIST sample paths
TRAIN_PKL = '/home/hannes/Data/mnist/train_paths.pkl'
# TEST_PKL  = '/home/hannes/Data/mnist/test_paths.pkl'
# paths = flatten(pickle.load(open(TRAIN_PKL, 'rb')))
# data = load_data(paths, True, "mnist_16-02-08_train.pkl")

data = pickle.load(open("mnist_16-02-08_train.pkl", 'rb'))
# import pdb; pdb.set_trace()
all_samples = np.concatenate([np.stack(data[c]) for c in range(10)])
all_samples = all_samples.reshape(len(all_samples), -1)
mean = all_samples.mean(axis=0)
cleaned = np.delete(all_samples, np.where(mean == 0), 1)

# compute mean -- naive
# mean = np.zeros_like(data[0][0], dtype=np.float32)
# count = 0
# for c in data:
#     for s in data[c]:
#         mean += s
#         count += 1
# mean /= count

# forward samples through the dnn to obtain the data points
# sample_class = 0
# for sample in data[sample_class]:

mlab_pca = PCA(cleaned)

colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#aaaaaa', '#ffa500', '#A52A2A']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(10):
    start = sum(map(lambda x: len(data[x]), range(i)))
    end = start + len(data[i])
    ax.plot(mlab_pca.Y[start:end, 0], mlab_pca.Y[start:end, 1], mlab_pca.Y[start:end, 2],
            '.', markersize=5, alpha=0.3, label=i, color=colors[i])

plt.legend(numpoints=1)
# plt.xlabel('x_values')
# plt.ylabel('y_values')
# plt.xlim([0,10])
# plt.ylim([-4,4])
plt.show()
