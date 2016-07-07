import pickle
import sys
import numpy as np

from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from chainer import cuda

import seaborn as sns
sns.set_palette("colorblind")
sns.set_color_codes("colorblind")


def make_pca(data):
    keys = sorted(list(data.keys()))
    data_np = np.concatenate([cuda.cupy.asnumpy(data[k]) for k in keys])
    mean = data_np.mean(axis=0)
    cleaned = np.delete(data_np, np.where(mean == 0), 1)
    pca = PCA(cleaned)
    index = 0
    result = {}
    for k in keys:
        k_samples = len(data[k])
        # result[k] = pca.Y[index:index+400]  # limit number of samples per key
        result[k] = pca.Y[index:index+k_samples]
        index += k_samples
    return result


def plot(data, num_classes, out, dims=2):
    keys = sorted(list(data.keys()))

    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y']

    fig = plt.figure()
    if dims == 2:
        ax = fig.add_subplot(111)
    elif dims == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
    else:
        print("Error: cannot plot in {} dimensions".format(dims))
        exit()

    for i in range(num_classes):
        persona = cuda.cupy.asnumpy(data[keys[i]])
        c = colors[i % len(colors)] if '_f' not in keys[i] else '#aaaaaa'
        if dims == 2:
            ax.scatter(persona[:, 0], persona[:, 1],
                       marker='o', s=50, c=c, edgecolor=c, label=keys[i], alpha=0.7)
        else:
            ax.scatter(persona[:, 0], persona[:, 1], persona[:, 2],
                       marker='o', s=50, c=c, edgecolor=c, label=keys[i], alpha=0.7)
    plt.legend()
    plt.show()
    if out is not None:
        plt.savefig(out)

if __name__ == '__main__':
    data = pickle.load(open(sys.argv[1], 'rb'))
    dims = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # make a pca
    data = make_pca(data)

    # limit to specified keys
    # keys = list(map(lambda x: '{:04d}'.format(x),
    #                 [11, 28, 30]
    #                 ))
    # data_a = {key: data[key] for key in data.keys() if key in keys}
    # data_b = {key+'_f': data[key+'_f'] for key in data.keys() if key in keys}
    # data = {**data_a, **data_b}

    plot(data, len(data), None, dims)
