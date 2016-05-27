import pickle
import sys
import numpy as np

from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from chainer import cuda


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
        result[k] = pca.Y[index:index+400]
        # result[k] = pca.Y[index:index+k_samples]
        index += k_samples
    return result


def plot(data, num_classes, out, dims=2):
    keys = sorted(list(data.keys()))

    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y',
              '#aaaaaa', '#ffa5a5', '#A5002A']

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
        c = colors[i % 10]
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

    plot(data, len(data), None, dims)
