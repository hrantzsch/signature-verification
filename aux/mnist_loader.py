"""Help loading data from mnist pkl files.
Completely tailored to my locally generated files..."""
import pickle
import numpy as np
from scipy.misc import imread


TRAIN_PKL = '/home/hannes/Data/mnist/train.pkl'
TEST_PKL = '/home/hannes/Data/mnist/test.pkl'


class MnistLoader:

    def __init__(self, xp):
        self.train_groups = pickle.load(open(TRAIN_PKL, 'rb'))
        self.test_groups = pickle.load(open(TEST_PKL, 'rb'))
        self.xp = xp

    def _load_triplet(self, triplet):
        return self.xp.array([imread(path).astype(self.xp.float32)
                              for path in triplet], dtype=self.xp.float32)

    def get_batch(self, anchor, batchsize, train=True):
        groups = self.train_groups if train else self.test_groups
        # anchors and positives
        paths = [groups[anchor][index][1] for index in np.random.choice(
                    len(groups[anchor]), batchsize*2)]
        negatives = list(range(len(groups)))
        negatives.remove(anchor)
        # negatives
        paths.extend([groups[neg][np.random.choice(len(groups[neg]))][1]
                      for neg in np.random.choice(negatives, batchsize)])

        return self.xp.array([imread(path).astype(self.xp.float32)
                              for path in paths], dtype=self.xp.float32)[:,self.xp.newaxis, ...]
