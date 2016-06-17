import os
import numpy as np
from scipy.misc import imread

import queue
import threading

from chainer import cuda


QUEUE_SIZE = 4


class LabelledLoader():
    """A Data Loader that provides batches of data with corresponding labels
       for classification."""

    def __init__(self, array_module):
        self.xp = array_module
        self.device = None
        self.workers = {}
        self.sources = {}

    def get_samples(self, data_dir):
        """returns a generator on lists of files per class in directory"""
        for d in os.listdir(data_dir):
            path = os.path.join(data_dir, d)
            if not os.path.isdir(path):
                continue
            files = os.listdir(path)
            for f in files:
                if '.png' in f:
                    yield (d, os.path.join(path, f))

    def get_samples_forgeries(self, data_dir):
        """same as get_samples, but only two labels: 0: authentic, 1: forged"""
        for d in os.listdir(data_dir):
            path = os.path.join(data_dir, d)
            if not os.path.isdir(path):
                continue
            files = os.listdir(path)
            for f in files:
                if '.png' in f:
                    yield (int('f' in f), os.path.join(path, f))

    def create_sources(self, data_dir, batchsize, split=0.9):
        """Create two sources, using <split> of the samples in <data_dir> for
           training and the rest for testing."""

        # split samples in data_dir to train and test set
        samples = list(self.get_samples(data_dir))
        np.random.shuffle(samples)
        num_train_samples = int(len(samples) * split)
        train_samples = samples[:num_train_samples]
        test_samples = samples[num_train_samples:]
        # create two providers accordingly
        self.create_source("train", train_samples, batchsize)
        self.create_source("test", test_samples, batchsize)

    def create_source(self, name, data, bs):
        """Create a data source, such as for train or test batches, and begin
           filling it with data.
           Parameter skilled indicates whether skilled forgeries should be re-
           cognized (if True) or considered positive samples.
        """
        # if name in self.sources:
        #     print("Warning: data source exists and will be overwritten.")
        self.sources[name] = queue.Queue(QUEUE_SIZE)

        worker = DataProvider(self.sources[name], data, bs, self.xp,
                              self.device)
        self.workers[name] = worker
        worker.start()

    def get_batch(self, source_name):
        """Try to get a batch of data."""
        # TODO: propagate exception and catch it on the outside
        try:
            data = self.sources[source_name].get(timeout=1)
        except queue.Empty:
            return None
        return data
        #     if self.workers[source_name].empty:
        #         return None
        #     else:
        #         return self.get_batch(source_name)
        # return data

    def use_device(self, device_id):
        self.device = device_id


class DataProvider(threading.Thread):

    def __init__(self, queue, samples, batchsize, xp, device):
        threading.Thread.__init__(self)
        self.queue = queue
        self.samples = samples
        self.batchsize = batchsize
        self.xp = xp
        self.device = device
        self.empty = False

    def run(self):
        # the anchor is not used right now, but we need to stop after
        # loading len(self.anchors) batches
        for i in range(0, len(self.samples), self.batchsize):
            batch = self.samples[i:i+self.batchsize]
            data = self.load_batch(batch)
            self.queue.put(data)  # blocking, no timeout
        self.empty = True

    def get_sample(self, path):
        # flip ?
        img = imread(path, mode='L')
        # if np.random.choice([True, False]):
        #     img = np.fliplr(img)
        # if np.random.choice([True, False]):
        #     img = np.flipud(img)
        return img.astype(self.xp.float32)

    def load_batch(self, paths):
        if self.device is not None:
            cuda.get_device(self.device).use()

        labels, samples = zip(*paths)
        batch = self.xp.array([self.get_sample(path) for path in samples],
                              dtype=self.xp.float32)
        return (self.xp.array(labels, self.xp.int32),
                (batch / 255.0)[:, self.xp.newaxis, ...])
