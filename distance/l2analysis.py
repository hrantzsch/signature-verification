"""
The purpose of this script is to get a feeling for the range of values I can
expect as output of the l2 normalization, so I can find a useful embedding size
"""

import argparse
import pickle
import numpy as np
from scipy.misc import imread

import chainer
from chainer import cuda

from embednet import EmbedNet
from data_loader import DataLoader
from l2normalization import l2_normalization

parser = argparse.ArgumentParser()
parser.add_argument('data',
                    help='Path to data')
parser.add_argument('--params', default=None, required=False,
                    help='Path to the parameters snapshot')
args = parser.parse_args()

model = EmbedNet()
if args.params is not None:
    params = pickle.load(open(args.params, 'rb'))
    model.copy_parameters_from(params)


def min_max(x_data):
    x = chainer.Variable(x_data)
    h = model.forward_dnn(x)
    h = l2_normalization(h)
    # import pdb; pdb.set_trace()


dl = DataLoader(args.data, np)
anchors = list(range(1, 4001))
for anchor in anchors:
    batch = dl.get_batch(anchor, 9)
    min_max(batch)
