import itertools
import os
import numpy as np
from scipy.misc import imread
from scipy.spatial.distance import cdist

import chainer
from chainer import cuda


def get_samples(data_dir, skilled):
    """Returns a generator on lists of files per class in directory.
       skilled indicates whether to include skilled forgeries."""
    # TODO only works for GPDSS
    skilled_condition = lambda f: (skilled and 'f' in f) or 'f' not in f
    for d in os.listdir(data_dir):
        path = os.path.join(data_dir, d)
        if not os.path.isdir(path):
            continue
        files = os.listdir(path)
        for f in files:
            if '.png' in f and skilled_condition(f):
                yield (d, os.path.join(path, f))


class NotReadyException(Exception):
    """Exception to be raised if HardNegativeLoader.prepare_batch has not been
       called before get_batch, or if prepare_batch did not successfully create
       a batch."""
    pass


class EpochDoneException(Exception):
    """Exception to be raised when the data loader has processed all the
       samples for this epoch."""
    pass


class HardNegativeLoader():
    """
    This data loader class implements one approach to sampling the hardest
    triplet combinations from the training set. This kind of sampling is
    motivated in [1] and parts of the implementation are based on Strateus'
    TripletChain (https://github.com/Strateus/TripletChain).

    [1] Schroff, Florian, Dmitry Kalenichenko, and James Philbin.
        “FaceNet: A Unified Embedding for Face Recognition and Clustering.”
        arXiv:1503.03832 [Cs], March 12, 2015. http://arxiv.org/abs/1503.03832.
    """

    def __init__(self, xp, samples, cbatchsize,
                 skilled=False, test_ratio=0.1):
        self.xp = xp
        self.cbatchsize = cbatchsize
        self.samples = samples
        self.reset()

    def reset(self):
        """Shuffle samples and reset pointer so new epoch can begin"""
        self.batch_ready = False
        self.samples_index = 0
        np.random.shuffle(self.samples)

    def load_samples(self, paths):
        """Load sample images from given paths and preprocess them as needed"""
        batch = self.xp.array([imread(path, mode='L').astype(self.xp.float32)
                              for path in paths], dtype=self.xp.float32)
        return (batch / 255.0)[:, self.xp.newaxis, ...]

    def worst_negatives(self, anc_index, pos_index, t_data):
        """Find indices of worst negative samples for a given anc-pos pair"""
        # extracting current pair distance
        a_p_dist = self.pairwise_distances[anc_index, pos_index]
        # extracting all indexes of negative samples
        negative_indices = np.where(t_data != t_data[anc_index])[0]
        # calculate diff of all possible anc-neg to this anc-pos
        diff = self.pairwise_distances[anc_index, negative_indices] - a_p_dist
        # worst embedded negatives are the onces with the smallest diff
        worst_negatives_indices = np.argsort(diff)
        return negative_indices[worst_negatives_indices]

    def prepare_batch(self, model, batchsize):
        """Advance to next candidate batch in training set and generate batch
           of hard triplets"""

        # load cbatchsize labels and samples
        cbatch = self.samples[self.samples_index:
                              self.samples_index + self.cbatchsize]
        self.samples_index += self.cbatchsize

        # check if we retrieved enough samples
        if len(cbatch) < self.cbatchsize:
            raise EpochDoneException("All samples have been processed.")

        # embed samples
        x_data = self.load_samples([d[1] for d in cbatch])
        x = chainer.Variable(x_data, volatile=True)
        emb = model.embed(x).data

        # conversion to numpy
        # TODO can I avoid this?
        #      cupy does not support unique nor where
        if self.xp != np:
            emb = np.array([cuda.cupy.asnumpy(x) for x in emb], np.float32)
        t_data = np.array([d[0] for d in cbatch], np.float32)

        # calculate pairwise distances
        self.pairwise_distances = cdist(emb, emb, 'sqeuclidean')

        # generate all anc-pos combinations
        a_p = []
        for label in np.unique(t_data):
            a_p.extend(list(itertools.combinations(
                            np.where(t_data == label)[0], 2)))

        if len(a_p) == 0:
            # just leave batch_ready False
            return

        np.random.shuffle(a_p)

        negatives = [self.worst_negatives(a_i, p_i, t_data)
                     for a_i, p_i in a_p]

        # TODO generalize shape
        triplets = self.xp.empty((batchsize, 3, 1, 96, 192), self.xp.float32)
        rank = 0  # start with worst negatives, then go down the list
        ap_idx = 0
        for t in range(len(triplets)):
            if ap_idx >= len(a_p):
                rank += 1
                ap_idx = 0
            a, p = a_p[ap_idx]
            n = negatives[ap_idx][rank]
            triplets[t][0] = x_data[a]
            triplets[t][1] = x_data[p]
            triplets[t][2] = x_data[n]
            ap_idx += 1

        self.batch = triplets
        self.batch_ready = True

    def get_batch(self):
        """Return previously prepared batch"""
        if not self.batch_ready:
            raise NotReadyException("No batch ready")

        self.batch_ready = False
        return self.batch
