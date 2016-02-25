import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import tsne


def load_data(path):
    # pkl_files = list(filter(lambda f: '.pkl' in f, os.listdir(path)))
    pkl_files = [os.path.join(path, "batch_{:04d}".format(b)) for b in range(300)]
    import pdb; pdb.set_trace()
#
# with open('16-02-17_embeddings.pkl', 'rb') as pickle_file:
#     data = pickle.load(pickle_file)




if __name__ == '__main__':
    data = load_data(sys.argv[1])

    minimum = min([len(value) for value in data.values()])
    print(minimum)
    minimum = 100

    dist_data = None
    for key in data.keys():
        if dist_data is None:
            dist_data = np.array(data[key][:minimum])
        else:
            dist_data = np.concatenate((dist_data, data[key][:minimum]), axis=0)

    dist_data = dist_data.reshape(dist_data.shape[0], dist_data.shape[-1])
    print(dist_data.shape)

    distribution = tsne.tsne(dist_data, no_dims=3, initial_dims=64)
    print(distribution.shape)


    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#aaaaaa', '#ffa500', '#A52A2A']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(10):
        start = i * minimum
        end = start + minimum
        # ax.scatter(distribution[start:end, 0], distribution[start:end, 1], distribution[start:end, 2],
        #            marker='.', color=colors[i])
        ax.plot(distribution[start:end, 0], distribution[start:end, 1], distribution[start:end, 2],
                '.', markersize=5, alpha=1, color=colors[i], label=i)

    plt.show()
