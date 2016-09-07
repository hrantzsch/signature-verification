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
        result[k] = pca.Y[index:index + k_samples]
        index += k_samples
    return result


def get_label(name, num=None):
    group = "Forgery" if '_f' in name else "Genuine"
    name = num if num else int(name[:4])
    return "User {} {}".format(name, group)


def plot_class(plot, data, label, color, forgery=False, n_dims=2):
    c = color if not forgery else 'gray'
    ec = color if forgery else 'white'
    lw = 0.2 if not forgery else 1.5
    if n_dims == 3:
        plot.scatter(data[:, 0], data[:, 1], data[:, 2],
                     marker='o', s=100, c=c, edgecolor=ec,
                     label=label, alpha=1.0, linewidth=lw)
    else:
        plot.scatter(data[:, 0], data[:, 1],
                     marker='o', s=100, c=c, edgecolor=ec,
                     label=label, alpha=1.0, linewidth=lw)


def plot(data, num_classes, out):
    keys = sorted(list(data.keys()))

    colors = ['b', 'b', 'g', 'g', 'r', 'r',
              'y', 'y', 'c', 'c', 'm', 'm']
    # colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y']

    fig, p2d = plt.subplots()
    p2d.set_xlim([-15, 25])
    # p2d.set_ylim([-10, 10])

    for i in range(num_classes):
        persona = cuda.cupy.asnumpy(data[keys[i]])
        c = colors[i % len(colors)]
        l = get_label(keys[i], i//2 + 1)
        plot_class(p2d, persona, l, c, '_f' in keys[i], 2)
    legend = plt.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_color('white')
    if out is not None:
        plt.savefig('a' + out, dpi=180)

    fig, p3d = plt.subplots()
    p3d = fig.add_subplot(111, projection='3d')
    p3d = Axes3D(fig)
    for i in range(num_classes):
        persona = cuda.cupy.asnumpy(data[keys[i]])
        c = colors[i % len(colors)]
        l = get_label(keys[i], i//2 + 1)
        plot_class(p3d, persona, l, c, '_f' in keys[i], 3)
    legend = plt.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_color('white')

    if out is not None:
        plt.savefig('b' + out, dpi=180)
    plt.show()

if __name__ == '__main__':
    data = pickle.load(open(sys.argv[1], 'rb'))
    dims = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # make a pca
    data = make_pca(data)

    # limit to specified keys
    # keys = list(map(lambda x: '{:04d}'.format(x),
    #                 [1, 2, 3]
    #                 ))
    keys = [k for k in data.keys() if '_f' not in k][-6:]
    data_a = {key: data[key] for key in data.keys() if key in keys}
    data_b = {key + '_f': data[key + '_f']
              for key in data.keys() if key in keys}
    data = {**data_a, **data_b}

    plot(data, len(data), "plot.png")
