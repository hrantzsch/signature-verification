import sys
sys.path.append("..")

import pickle
import os
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.misc import imread, imshow

import chainer
from chainer import cuda
from chainer import serializers

import numpy as np
from scipy.spatial.distance import cdist

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tripletembedding.predictors import TripletNet
from models.vgg_small import VGGSmall

cuda.get_device(0).use()

def read_image(path):
    return imread(path, mode='L')


def show_images(imgs):
    width = 32
    height = 16
    plt.figure(figsize=(width, height))
    # num_rows, num_cols = 2, len(imgs) // 2
    # plt.figure()
    num_rows, num_cols = 1, 6
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.0, hspace=0.0)
    ax = [plt.subplot(gs[i]) for i in range(num_rows*num_cols)]
    gs.update(hspace=0)
    for i, im in enumerate(imgs):
        ax[i].imshow(im, cmap=plt.cm.gray)
        ax[i].axis('off')


def load_model(path):
    model = TripletNet(VGGSmall)
    serializers.load_hdf5(path, model)
    model = model.to_gpu()
    print("Model restored!")
    return model


def embed_samples(model, samples):
    """embed samples; expects all samples for a class at once"""
    if len(samples) == 0:
        print("Error: no samples to embed")
    data = cuda.cupy.array(samples, dtype=cuda.cupy.float32)
    data = (data / 255.0)[:, cuda.cupy.newaxis, ...]

    xs = model.embed(chainer.Variable(data))
    return cuda.cupy.asnumpy(xs.data)


def euclidean_distance(a, b):
    a = a[np.newaxis, ...]
    b = b[np.newaxis, ...]
    return cdist(a, b, 'sqeuclidean')


def show_distances(imgs, distances, label='distance'):
    width = 32
    height = 16
    plt.figure(figsize=(width, height))
    num_rows, num_cols = 6, 2
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.0, hspace=0.0)
    ax = [plt.subplot(gs[i]) for i in range(num_rows*num_cols)]
    gs.update(hspace=0)
    for i in range(0, len(imgs), 1):
        ax[2*i].imshow(imgs[i], cmap=plt.cm.gray)
        ax[2*i+1].text(0, 0.5, "{}: {:.2f}".format(label, distances[i][0][0]),
                       fontsize=42)
        ax[2*i].axis('off')
        ax[2*i+1].axis('off')


def show_llrs(imgs, llrs):
    show_distances(imgs, llrs, 'LLR')
