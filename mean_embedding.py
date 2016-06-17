"""Embed all samples and compute mean of embeddings per class"""

# Data is expected to be a directory like
# data
#   +- class 1
#         +- sample 1
#         ...
#         +- sample 1
#   +- class 2
#   ...
#   +- class n

import argparse
import numpy as np
import os
import pickle
from scipy.misc import imread

import chainer
from chainer import cuda
from chainer import serializers

from tripletembedding.predictors import TripletNet
from tripletembedding.models import SmallDnn

from tools.embeddings_plot import plot

from models.vgg_small import VGGSmall


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('data')
    parser.add_argument('--out', '-o', default=None,
                        help='Pickle embeddings to given filename.\
                        If --plot is given then save plot to filename.')
    parser.add_argument('--batchsize', '-b', type=int, default=12,
                        help='Learning minibatch size [12]')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU) [-1]')
    parser.add_argument('--no_mean', action="store_true", default=False,
                        help='Do not compute the mean')
    parser.add_argument('--plot', action="store_true", default=False,
                        help='Plot after embedding')
    parser.add_argument('--dims', '-d', default=2, type=int,
                        help='Number of plotting dimensions')
    parser.add_argument('--classes', '-c', default=5000, type=int,
                        help='Number of classes to use')
    return parser.parse_args()


def get_samples(data_dir):
    """returns a generator on lists of files per class in directory"""
    for d in os.listdir(data_dir):
        path = os.path.join(data_dir, d)
        if not os.path.isdir(path):
            continue
        files = os.listdir(path)
        yield (d, [os.path.join(path, f) for f in files if '.png' in f])


def embed_class(xp, model, samples, bs):
    """embed samples; expects all samples for a class at once"""
    if len(samples) == 0:
        print("Error: no samples to embed")
    data = xp.array([imread(s, mode='L') for s in samples], dtype=xp.float32)
    data = (data / 255.0)[:, xp.newaxis, ...]
    num_batches = len(data) // bs + 1

    xs = xp.array_split(data, num_batches)
    xs = [model.embed(chainer.Variable(x)).data for x in xs]
    if len(xs) > 1:
        xs = xp.vstack(xs)
    else:
        xs = xs[0]
    return xs.squeeze()


if __name__ == '__main__':
    args = parse_args()
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
    xp = cuda.cupy if args.gpu >= 0 else np

    model = TripletNet(VGGSmall)
    serializers.load_hdf5(args.model, model)
    model.cnn.train = False
    if args.gpu >= 0:
        model = model.to_gpu()

    embeddings = {}
    for (name, samples) in get_samples(args.data):
        print("embedding", name)

        # HACK relabel forgeries
        genuine = [s for s in samples if 'f' not in s]
        forgeries = [s for s in samples if 'f' in s]

        embedded_g = embed_class(xp, model, genuine, args.batchsize)
        embedded_f = embed_class(xp, model, forgeries, args.batchsize)

        # if not args.no_mean:
        #     embedded = xp.mean(embedded, axis=0)
        embeddings[name] = embedded_g
        embeddings[name + '_f'] = embedded_f

        if len(embeddings) >= args.classes:
            break

    if args.out is not None:
        pickle.dump(embeddings, open(args.out, 'wb'))

    if args.plot:
        plot(embeddings, len(embeddings), args.out, dims=args.dims)
