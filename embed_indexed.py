"""
Embedding samples similar to mean_embedding, but based on and index.pkl
file as data input.
A mean over the data is not computed.
"""


import argparse
import numpy as np
import pickle
from scipy.misc import imread

import chainer
from chainer import cuda
from chainer import serializers

from tripletembedding.predictors import TripletNet

from tools.embeddings_plot import plot

from models.vgg_small import VGGSmall


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('data')
    parser.add_argument('--batchsize', '-b', type=int, default=40,
                        help='Learning minibatch size [40]')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU) [-1]')
    parser.add_argument('--classes', '-c', default=10000, type=int,
                        help='Maximum number of classes to use')
    parser.add_argument('--out', '-o', default=None,
                        help='Pickle embeddings to given filename.'
                        ' If --plot is given then save plot to filename.')
    parser.add_argument('--plot', action="store_true", default=False,
                        help='Plot after embedding')
    parser.add_argument('--dims', '-d', default=2, type=int,
                        help='Number of plotting dimensions')
    return parser.parse_args()


def embed_class(xp, model, samples, bs):
    """embed samples; expects all samples for a class at once"""
    if len(samples) == 0:
        print("Error: no samples to embed")
    data = xp.array([imread(s, mode='L') for s in samples], dtype=xp.float32)
    data = data[:, xp.newaxis, ...]
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

    data = pickle.load(open(args.data, 'rb'))

    embeddings = {}

    for user in data['Forged']:
        if user > args.classes:
            break
        samples = data['Forged'][user]
        embedded = embed_class(xp, model, samples, args.batchsize)
        embeddings["{:04d}_f".format(user)] = embedded

    for user in data['Genuine']:
        if user > args.classes:
            break
        samples = data['Genuine'][user]
        embedded = embed_class(xp, model, samples, args.batchsize)
        embeddings["{:04d}".format(user)] = embedded

    if args.plot:
        plot(embeddings, len(embeddings), args.out, dims=args.dims)
    elif args.out is not None:
        pickle.dump(embeddings, open(args.out, 'wb'))
