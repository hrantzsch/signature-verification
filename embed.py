"""Forward all files in a given folder through the DNN. Save the embedded
results to a file."""

import numpy as np
import os
from scipy.misc import imread
import sys
import threading
import queue

import chainer
from chainer import cuda
from chainer import serializers

from models.alex_dnn import AlexDNN
from models.tripletnet import TripletNet


def load_batch(files, batchsize, queue, xp):
    for i in range(0, len(files), batchsize):
        batch = xp.array([imread(path).astype(xp.float32)
                         for path in files[i:i+batchsize]], dtype=xp.float32)
        queue.put((batch / 255.0)[:, xp.newaxis, ...])


class Data:
    def __init__(self, data, batchsize, xp):
        self.data = data
        self.batchsize = batchsize
        self.queue = queue.Queue(3)
        self.xp = xp
        all_files = []
        for f in os.listdir(data):
            folder = os.path.join(data,f)
            if os.path.isdir(folder):
                for sample in [s for s in os.listdir(folder) if '.png' in s]:
                    all_files.append(os.path.join(folder, sample))
        self.worker = self.start_worker(all_files)

    def start_worker(self, files):
            w = threading.Thread(target=load_batch,
                args=(files, self.batchsize, self.queue, self.xp))
            w.start()
            return w

    def get_batch(self):
        return self.queue.get()


if __name__ == '__main__':
    model_path = sys.argv[1]
    dir_path = sys.argv[2]

    data = Data(dir_path, 80, cuda.cupy)

    model = TripletNet(dnn=AlexDNN)
    serializers.load_hdf5(model_path, model)
    dnn = model.dnn

    batch = dnn(data.get_batch())
    import pdb; pdb.set_trace()
    pass


#
#
# class Worker(threading.Thread):
#     def __init__(self, data, bs, q, xp):
#         self.data = data
#         self.bs = bs
#         self.q = q
#         self.xp = xp
#
#     def run(self):
#         all_files = []
#         # for i in range(0, len(all_files), batchsize):
#         while True:
#             batch = self.xp.array(np.random.random(self.bs), dtype=self.xp.float32)
#             self.q.put(batch)
#
#
# class Data:
#     def __init__(self, data, batchsize, xp):
#         self.data = data
#         self.batchsize = batchsize
#         self.queue = queue.Queue(3)
#         self.xp = xp
#         self.worker = self.start_worker()
#         self.worker.run()
#
#     def start_worker(self):
#             return threading.Thread(target=load_batch,
#                 args=(self.data, self.batchsize, self.queue, self.xp))
#
#     def get_batch(self):
#         self.queue.get()
