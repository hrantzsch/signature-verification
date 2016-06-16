"""A script to test my model on the MCYT dataset.

The script expects embedded data as a .pkl file.

Currently the script prints min, mean, and max distances intra-class and
comparing a class's samples to the respective forgeries.
"""

# TODO use sqeucledean

import pickle
import sys
import numpy as np
from scipy.spatial.distance import cdist, pdist
from chainer import cuda


DIST_METHOD = 'euclidean'


def load_keys(dictionary, keys):
    values = [s for k in keys for s in dictionary[k]]
    return np.stack(list(map(cuda.cupy.asnumpy, values)))


def pdists_for_key(embeddings, k):
    samples = cuda.cupy.asnumpy(embeddings[k])
    return pdist(samples, DIST_METHOD)


def genuine_forgeries_dists(embeddings):
    gen_keys = [k for k in embeddings.keys() if 'f' not in k]
    dists = {}
    for k in gen_keys:
        gen = cuda.cupy.asnumpy(embeddings[k])
        forge = cuda.cupy.asnumpy(embeddings[k+'_f'])
        dists[k] = cdist(gen, forge, DIST_METHOD)
    return dists


if __name__ == '__main__':
    model_path = sys.argv[1]
    model = pickle.load(open(model_path, 'rb'))

    # k_genuine = [k for k in model.keys() if 'f' not in k]
    # k_forgeries = [k for k in model.keys() if 'f' in k]

    # genuine = load_keys(model, k_genuine)
    # forgeries = load_keys(model, k_forgeries)

    # cdists = cdist(genuine, forgeries, DIST_METHOD)

    intra_class_dists = {k: pdists_for_key(model, k) for k in model.keys()}
    gen_forge_dists = genuine_forgeries_dists(model)

    print("class\t|\tintra-class\t\t|\tforgeries")
    print("\t|\tmin - mean - max\t|\tmin - mean - max")
    for k in sorted(intra_class_dists.keys()):
        if 'f' in k:
            continue
        dist = intra_class_dists[k]
        print("{}\t|\t{} - {} - {}".format(k,
                                           int(np.min(dist)),
                                           int(np.mean(dist)),
                                           int(np.max(dist))),
              end='\t')
        dist = gen_forge_dists[k]
        print("|\t{} - {} - {}".format(int(np.min(dist)),
                                       int(np.mean(dist)),
                                       int(np.max(dist))))


    # print("intra-class distances:")
    # keys = list(model.keys())
    # for k in keys:
    #     if 'f' in k:
    #         continue
    #     samples = cuda.cupy.asnumpy(model[k])
    #     pdists = pdist(samples, 'sqeuclidean')
    #     print("{}:\t\t{}".format(k, np.mean(pdists)))

    # print("overall mean:\t", end='')
    # samples = [s for s in model[k] for k in keys]
    # samples = np.stack(list(map(cuda.cupy.asnumpy, samples))).astype(np.float32)
    # print(np.mean(pdist(samples, 'sqeuclidean')))

    # print("intra-class distances forgeries:")
    # keys = list(model.keys())
    # for k in keys:
    #     if 'f' not in k:
    #         continue
    #     samples = cuda.cupy.asnumpy(model[k])
    #     pdists = pdist(samples, 'sqeuclidean')
    #     print("{}:\t\t{}".format(k, np.mean(pdists)))
