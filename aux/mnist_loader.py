"""Help loading data from mnist pkl files.
Completely tailored to my locally generated files..."""
import pickle
import numpy as np
from scipy.misc import imread


TRAIN_PKL = '/home/hannes/data/mnist/train.pkl'
TEST_PKL  = '/home/hannes/data/mnist/test.pkl'


class MnistLoader:

    def __init__(self, xp):
        self.train_groups = pickle.load(open(TRAIN_PKL, 'rb'))
        self.test_groups = pickle.load(open(TEST_PKL, 'rb'))
        self.xp = xp

    def get_rnd_triplet(self, groups, anchor=None):
        classes = list(range(len(groups)))
        if anchor is None:
            np.random.shuffle(classes)
            anchor = classes.pop()
        else:
            classes.remove(anchor)
        negative = np.random.choice(classes)

        return self.xp.array([
            groups[anchor][np.random.randint(len(groups[anchor]))][1],
            groups[anchor][np.random.randint(len(groups[anchor]))][1],
            groups[negative][np.random.randint(len(groups[negative]))][1]
        ])

    def get_batch(self, batchsize, anchor=None, train=True):
        groups = self.train_groups if train else self.test_groups

        # # anchors and positives
        # paths = [groups[anchor][index][1] for index in np.random.choice(
        #             len(groups[anchor]), batchsize*2)]
        # negatives = list(range(len(groups)))
        # negatives.remove(anchor)
        # # negatives
        # paths.extend([groups[neg][np.random.choice(len(groups[neg]))][1]
        #               for neg in np.random.choice(negatives, batchsize)])

        paths = self.xp.array([self.get_rnd_triplet(groups, anchor)
                               for _ in range(batchsize)]).flatten()

        batch = self.xp.array([imread(path).astype(self.xp.float32)
                              for path in paths], dtype=self.xp.float32)

        return (batch / 255.0)[:,self.xp.newaxis, ...]
