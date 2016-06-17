"""A triplet loader that works on index files.
The index file should be a .pkl file that contains the following structure:
{
    'Genuine': { label: [absolute sample paths] },
    'Forged':  { label: [absolute sample paths] }
}
where label is an int.
"""

import pickle
import numpy as np
from scipy.misc import imread

import queue
import threading

from chainer import cuda


def anchors_in(data_dict):
    return list(pickle.load(open(data_dict, 'rb'))['Genuine'].keys())


QUEUE_SIZE = 4


class IndexLoader:

    def __init__(self, array_module):
        self.xp = array_module
        self.device = None
        self.workers = {}
        self.sources = {}

    def create_source(self, name, anchors, num_triplets, data, skilled):
        """Create a data source, such as for train or test batches, and begin
           filling it with data.
           Parameter skilled indicates whether skilled forgeries should be
           recognized (if True) or considered positive samples.
        """
        self.sources[name] = queue.Queue(QUEUE_SIZE)

        worker = IndexDataProvider(self.sources[name], anchors, num_triplets,
                                   self.xp, self.device, data, skilled)
        self.workers[name] = worker
        worker.start()

    def get_batch(self, source_name):
        return self.sources[source_name].get()

    def use_device(self, device_id):
        self.device = device_id


class IndexDataProvider(threading.Thread):

    def __init__(self, queue, anchors, num_triplets,
                 xp, device, data, skilled):
        threading.Thread.__init__(self)
        self.queue = queue
        self.anchors = anchors
        self.num_triplets = num_triplets
        self.xp = xp
        self.device = device
        self.data = pickle.load(open(data, 'rb'))
        self.skilled = skilled

    def run(self):
        # the anchor is not used right now, but we need to stop after
        # loading len(self.anchors) batches
        for _ in self.anchors:
            data = self.load_batch(self.skilled)
            self.queue.put(data)  # blocking, no timeout

    def get_sample(self, persona, forged):
        subdir = "Forged" if forged else "Genuine"
        return np.random.choice(self.data[subdir][persona])

    def load_sample(self, path):
        sample = imread(path)
        return sample.astype(self.xp.float32)

    def get_easy_triplet(self):
        """Return a valid triplet (anc, pos, neg) of image paths.
           If no_forgeries is True then no forgeries will be used for anchor
           and positive. Use this for skilled training.
        """
        anc, neg = np.random.choice(self.anchors, 2, replace=False)
        a = self.get_sample(anc, False)
        p = self.get_sample(anc, False)
        n = self.get_sample(neg, np.random.choice([True, False]))
        return (a, p, n)

    def get_skilled_triplet(self):
        """Return a triplet where the negative sample is a skilled forgery of
        the anchor"""
        persona = np.random.choice(self.anchors)
        anc = self.get_sample(persona, False)
        pos = self.get_sample(persona, False)
        neg = self.get_sample(persona, True)
        return (anc, pos, neg)

    def load_batch(self, num_skilled):
        if self.device is not None:
            cuda.get_device(self.device).use()

        num_easy_triplets = int(np.floor(self.num_triplets * (1 - num_skilled)))
        num_skilled_triplets = self.num_triplets - num_easy_triplets

        # NOTE: no_forgeries is switched implicitly here. This can lead to
        # irritating effects when an already trained model is suddenly fed
        # forgeries that are labeled as positives.
        easy_triplets = [self.get_easy_triplet()
                         for _ in range(num_easy_triplets)]
        skilled_triplets = [self.get_skilled_triplet()
                            for _ in range(num_skilled_triplets)]
        triplets = easy_triplets + skilled_triplets

        if num_skilled_triplets > 0:
            np.random.shuffle(triplets)

        paths = []
        for i in range(3):
            for j in range(self.num_triplets):
                paths.append(triplets[j][i])

        batch = self.xp.array([self.load_sample(path)
                              for path in paths], dtype=self.xp.float32)
        return (batch / 255.0)[:, self.xp.newaxis, ...]
