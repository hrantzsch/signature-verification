"""
Script to train the fully connected layer used for ICFHR16 classification.
Use embedded representations of samples, or raw samples and a trained embedding
network as input.
"""

import numpy as np
import argparse

import chainer
from chainer import cuda
from chainer import serializers

from tripletembedding.predictors import TripletNet
from tripletembedding.models import SmallDnn

from aux.icfhr_loader import IcfhrLoader
from models.alex_dnn import AlexDNN
from models.new_cnn import NewCnn


from matplotlib.mlab import PCA
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('data', help='Path to sample data. Samples are expected to\
                                  be embedded if no model is provided.')

parser.add_argument('--model', '-m', help='Model to perform embedding')

parser.add_argument('--batchsize', '-b', type=int, default=12,
                    help='Learning minibatch size [12]')
parser.add_argument('--epochs', '-e', default=50, type=int,
                    help='Number of epochs to learn [50]')
parser.add_argument('--test', '-t', default=0.1, type=float,
                    help='Fraction of samples to spare for testing [0.1]')

parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU) [-1]')

parser.add_argument('--interval', '-i', default=10, type=int,
                    help='Snapshot interval in epochs [10]')
parser.add_argument('--lrinterval', '-l', default=10, type=int,
                    help='Interval for halving the LR [10]')
parser.add_argument('--out', '-o', default='',
                    help='Name for snapshots and logging')

args = parser.parse_args()


# setup
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
xp = cuda.cupy if args.gpu >= 0 else np
dl = IcfhrLoader(args.data, xp)

train_anchors = dl.anchors(train=False)

NUM_CLASSES = len(train_anchors)
NUM_SAMPLES = args.batchsize  # samples per class, must fit into GPU memory


# load embedding model if given
perform_embedding = False
if args.model:
    model = TripletNet(AlexDNN)
    serializers.load_hdf5(args.model, model)
    model.cnn.train = False
    print("Load embedding model from", args.model)
    perform_embedding = True
    if args.gpu >= 0:
        model = model.to_gpu()
else:
    print("Warning: No model given. Assuming samples are already embedded.")


def plot_pca(pca):
    colors = ['b', 'g', 'r', 'k', 'c', 'm',
              'y', '#aaaaaa', '#ffa500', '#A52A2A']
    fig = plt.figure()

    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)

    for i in range(NUM_CLASSES):
        c = colors[i % 10]
        ax.scatter(pca.Y[i*NUM_SAMPLES:(i+1)*NUM_SAMPLES, 0],
                   pca.Y[i*NUM_SAMPLES:(i+1)*NUM_SAMPLES, 1],
                #    pca.Y[i*NUM_SAMPLES:(i+1)*NUM_SAMPLES, 2],
                   marker='o', s=50, c=c, edgecolor=c, label=i+158, alpha=0.6)

    plt.legend()
    plt.savefig('pca.png')
    plt.show()


def plot_tsne(t):
    colors = ['b', 'g', 'r', 'k', 'c', 'm',
              'y', '#aaaaaa', '#ffa500', '#A52A2A']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(10):
        start = i
        end = start
        # ax.scatter(distribution[start:end, 0], distribution[start:end, 1], distribution[start:end, 2],
        #            marker='.', color=colors[i])
        ax.plot(t[start:end, 0], t[start:end, 1], '.', markersize=5, alpha=1, color=colors[i], label=i)
    # ax.plot(t[:, 0], t[:, 1], '.', markersize=5, alpha=1)

    plt.savefig('tsne.png')
    plt.show()


def get_data(anchors):
    x_data, _ = dl.get_batch_labelled(anchors, train=False)
    data = model.embed(chainer.Variable(x_data)).data
    if args.gpu >= 0:
        data = cuda.cupy.asnumpy(data)
    return data

data = []
step = 1
for c in train_anchors:
    print("getting data:\t[" + "#"*step + " "*(len(train_anchors)-step) + "]",
          end='\r')
    data.append(get_data([c for _ in range(NUM_SAMPLES)]))
    step += 1
data = np.concatenate(data).squeeze()

mean = data.mean(axis=0)
cleaned = np.delete(data, np.where(mean == 0), 1)

pca = PCA(cleaned)
plot_pca(pca)

# import tools.tsne as tsne
# t = tsne.tsne(cleaned, no_dims=2, initial_dims=64)
# plot_tsne(t)



# start = sum(map(lambda x: len(data[x]), range(i)))
# end = start + len(data[i])
# ax.scatter(pca.Y[start:end, 0], pca.Y[start:end, 1], pca.Y[start:end, 2],
#            marker='.', color=colors[i])
# ax.plot(pca.Y[:, 0], pca.Y[:, 1], pca.Y[:, 2],
#         '.', markersize=5, alpha=0.5)
