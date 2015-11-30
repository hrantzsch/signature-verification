import argparse
import pickle
import numpy as np
from scipy.misc import imread

import chainer
from chainer import cuda
from chainer import serializers

from embednet import EmbedNet

parser = argparse.ArgumentParser()
parser.add_argument('model',
                    help='Path to the model')
parser.add_argument('sample_a',
                    help='Path to first signature')
parser.add_argument('sample_b',
                    help='Path to second signature')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

args = parser.parse_args()

model = EmbedNet()

if args.gpu >= 0:
    cuda.check_cuda_available()
    model.to_gpu(args.gpu)
xp = cuda.cupy if args.gpu >= 0 else np

print('Load model from', args.model)
serializers.load_hdf5(args.model, model)

samples = [xp.asarray(imread(args.sample_a).astype(xp.float32)[xp.newaxis, xp.newaxis, ...]),
           xp.asarray(imread(args.sample_b).astype(xp.float32)[xp.newaxis, xp.newaxis, ...])]

samples = chainer.Variable(xp.concatenate(samples))
distance = model.verify(samples)
print(distance.data)
