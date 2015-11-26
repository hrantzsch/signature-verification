import argparse
import pickle
import numpy as np
from scipy.misc import imread

import chainer

parser = argparse.ArgumentParser()
parser.add_argument('model',
                    help='Path to the model')
parser.add_argument('sample_a',
                    help='Path to first signature')
parser.add_argument('sample_b',
                    help='Path to second signature')
args = parser.parse_args()

model = pickle.load(open(args.model, 'rb'))

samples = [imread(args.sample_a).astype(np.float32)[np.newaxis, ...],
           imread(args.sample_b).astype(np.float32)[np.newaxis, ...]]

distance = model.verify(np.concatenate(samples))
print(distance)
