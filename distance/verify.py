import argparse
import pickle
import numpy as np
from scipy.misc import imread

import chainer
from chainer import cuda

parser = argparse.ArgumentParser()
parser.add_argument('model',
                    help='Path to the model')
parser.add_argument('sample_a',
                    help='Path to first signature')
parser.add_argument('sample_b',
                    help='Path to second signature')
args = parser.parse_args()

model = pickle.load(open(args.model, 'rb'))
model.to_gpu(0)

xp = cuda.cupy

samples = [xp.asarray(imread(args.sample_a).astype(xp.float32)[xp.newaxis, xp.newaxis, ...]),
           xp.asarray(imread(args.sample_b).astype(xp.float32)[xp.newaxis, xp.newaxis, ...])]

samples = xp.concatenate(samples)
distance = model.verify(samples)
print(distance.data)
