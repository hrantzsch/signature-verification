import os
import numpy as np
from scipy.misc import imread

import queue
import threading

from chainer import cuda


QUEUE_SIZE = 8


class TripletLoader(threading.Thread):

    def __init__(self, array_module):
        self.xp = array_module

        self.workers = {}
        self.sources = {}

    def create_source(self, name, anchors, num_triplets, data_dir):
        """Create a data source, such as for train or test batches, and begin
           filling it with data.
        """
        # if name in self.sources:
        #     print("Warning: data source exists and will be overwritten.")
        self.sources[name] = queue.Queue(QUEUE_SIZE)

        worker = DataProvider(self.sources[name], anchors, num_triplets,
                              self.xp, data_dir)
        self.workers[name] = worker
        worker.start()

    def get_batch(self, source_name):
        return self.sources[source_name].get()


class DataProvider(threading.Thread):

    def __init__(self, queue, anchors, num_triplets, xp, data_dir):
        threading.Thread.__init__(self)
        self.queue = queue
        self.anchors = anchors
        self.num_triplets = num_triplets
        self.xp = xp
        self.data_dir = data_dir

    def run(self):
        for a in self.anchors:
            data = self.load_batch()
            self.queue.put(data)  # blocking, no timeout

    def get_rnd_triplet(self):
        """Return a valid triplet (anc, pos, neg) of image paths."""
        anc, neg = np.random.choice(self.anchors, 2, replace=False)
        a = self.get_rnd_sample(anc)
        p = self.get_rnd_sample(anc)
        n = self.get_rnd_sample(neg)
        return (a, p, n)

    def get_rnd_sample(self, persona):
        """Return the path to a random signature of the given persona."""
        directory = os.path.join(self.data_dir, "{:03d}".format(persona))
        sign_num = np.random.randint(1, 55)
        if sign_num > 24:  # a forgery
            prefix = "cf"
            sign_num -= 24
        else:
            prefix = "c"
        variation = np.random.randint(1, 21)
        fname = "{}-{:03d}-{:02d}-{:02d}.png".format(
            prefix, persona, sign_num, variation)
        return os.path.join(directory, fname)

    def load_batch(self):
        triplets = [self.get_rnd_triplet() for _ in range(self.num_triplets)]
        paths = []
        for i in range(3):
            for j in range(self.num_triplets):
                paths.append(triplets[j][i])

        batch = self.xp.array([imread(path).astype(self.xp.float32)
                              for path in paths], dtype=self.xp.float32)
        return (batch / 255.0)[:, self.xp.newaxis, ...]
