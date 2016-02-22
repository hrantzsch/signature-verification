"""Forward all files in a given folder through the DNN. Save the embedded
results to a file."""

import numpy as np
import os
import pickle
from scipy.misc import imread
import sys
import threading
import queue

import chainer
from chainer import cuda
from chainer import serializers

from models.alex_dnn import AlexDNN
from models.tripletnet import TripletNet


# all_files = []
# for f in os.listdir(data):
#     folder = os.path.join(data, f)
#     if os.path.isdir(folder):
#         for sample in [s for s in os.listdir(folder) if '.png' in s]:
#             all_files.append(os.path.join(folder, sample))
# pickle.dump(all_files, open("/data/hannes/GPDSS/rot_bg_index.pkl", 'wb'))


def load_batch(files, batchsize, queue, xp):
    print("running {} batches.".format(len(files)/batchsize))
    with cuda.get_device(1):
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
        all_files = pickle.load(open("/data/hannes/GPDSS/rot_bg_index.pkl", 'rb'))
        self.worker = self.start_worker(all_files)

    def start_worker(self, files):
        w = threading.Thread(target=load_batch,
                             args=(files, self.batchsize, self.queue, self.xp))
        w.start()
        return w

    def get_batch(self):
        return self.queue.get()

    def finished(self):
        return (self.queue.empty and not self.worker.isAlive)


if __name__ == '__main__':
    model_path = sys.argv[1]
    dir_path = sys.argv[2]

    cuda.get_device(1).use()

    data = Data(dir_path, 800, cuda.cupy)

    model = TripletNet(dnn=AlexDNN)
    serializers.load_hdf5(model_path, model)
    model = model.to_gpu(1)
    dnn = model.dnn

    batch_num = 0
    while not data.finished():
        print("batch {:04d}".format(batch_num), end='\r')
        x = chainer.Variable(data.get_batch())
        batch = dnn(x)
        pickle.dump(batch, open('/data/hannes/GPDSS/embedded_{:04d}'.format(batch_num), 'wb'))
        batch_num += 1
