"""Same as triplet_loader, but for the MCYT data set."""

import os
import numpy as np
from scipy.misc import imread

import queue
import threading

from chainer import cuda


QUEUE_SIZE = 4


class McytLoader:

    def __init__(self, array_module):
        self.xp = array_module
        self.device = None
        self.workers = {}
        self.sources = {}

    def create_source(self, name, anchors, num_triplets, data_dir, skilled):
        """Create a data source, such as for train or test batches, and begin
           filling it with data.
           Parameter skilled indicates whether skilled forgeries should be re-
           cognized (if True) or considered positive samples.
        """
        # if name in self.sources:
        #     print("Warning: data source exists and will be overwritten.")
        self.sources[name] = queue.Queue(QUEUE_SIZE)

        worker = McytDataProvider(self.sources[name], anchors, num_triplets,
                                  self.xp, self.device, data_dir, skilled)
        self.workers[name] = worker
        worker.start()

    def get_batch(self, source_name):
        return self.sources[source_name].get()

    def use_device(self, device_id):
        self.device = device_id


class McytDataProvider(threading.Thread):

    def __init__(self, queue, anchors, num_triplets,
                 xp, device, data_dir, skilled):
        threading.Thread.__init__(self)
        self.queue = queue
        self.anchors = anchors
        self.num_triplets = num_triplets
        self.xp = xp
        self.device = device
        self.data_dir = data_dir
        self.skilled = skilled

    def run(self):
        # the anchor is not used right now, but we need to stop after
        # loading len(self.anchors) batches
        for _ in self.anchors:
            data = self.load_batch(self.skilled)
            self.queue.put(data)  # blocking, no timeout

    def get_sample(self, persona, sign_num):
        """Get random variation of the given personal and signature number."""
        persona -= 1  # adapt from GPDSS, which starts counting at 1
        directory = os.path.join(self.data_dir, "{:04d}".format(persona))
        if sign_num >= 25:  # a forgery
            prefix = "f"
            sign_num -= 25
        else:
            prefix = "v"
        fname = "{:04d}{}{:02d}.png".format(persona, prefix, sign_num)
        return os.path.join(directory, fname)

    def get_rnd_sample(self, persona, no_forgeries=False):
        """Return the path to a random signature of the given persona."""
        sign_num = np.random.randint(25 if no_forgeries else 50)
        return self.get_sample(persona, sign_num)

    def get_easy_triplet(self, no_forgeries):
        """Return a valid triplet (anc, pos, neg) of image paths.
           If no_forgeries is True then no forgeries will be used for anchor
           and positive. Use this for skilled training.
        """
        anc, neg = np.random.choice(self.anchors, 2, replace=False)
        a = self.get_rnd_sample(anc, no_forgeries)
        p = self.get_rnd_sample(anc, no_forgeries)
        n = self.get_rnd_sample(neg)
        return (a, p, n)

    def get_skilled_triplet(self):
        """Return a triplet where the negative sample is a skilled forgery of
        the anchor"""
        persona = np.random.choice(self.anchors)
        anc_num, pos_num = np.random.choice(range(25), 2, replace=False)
        neg_num = np.random.randint(25, 50)
        return tuple(self.get_sample(persona, sign_num)
                     for sign_num in [anc_num, pos_num, neg_num])

    def load_batch(self, skilled):
        if self.device is not None:
            cuda.get_device(self.device).use()

        num_easy_triplets = int(np.floor(self.num_triplets * (1 - skilled)))
        num_skilled_triplets = self.num_triplets - num_easy_triplets

        # NOTE: no_forgeries is switched implicidly here. This can lead to
        # irritating effects when an already trained model is suddenly fed
        # forgeries that are labelled as positives.
        easy_triplets = [self.get_easy_triplet(num_skilled_triplets > 0)
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

        batch = self.xp.array([imread(path).astype(self.xp.float32)
                              for path in paths], dtype=self.xp.float32)
        return (batch / 255.0)[:, self.xp.newaxis, ...]
