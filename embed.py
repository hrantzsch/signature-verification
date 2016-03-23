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


BATCHS_PP = 2
GPU = 1  # the hacky way


# all_files = []
# for f in os.listdir(data):
#     folder = os.path.join(data, f)
#     if os.path.isdir(folder):
#         for sample in [s for s in os.listdir(folder) if '.png' in s]:
#             all_files.append(os.path.join(folder, sample))
# pickle.dump(all_files, open("/data/hannes/GPDSS/rot_bg_index.pkl", 'wb'))


def load_batch(files, batchsize, queue, xp):
    print("running {} batches.".format(len(files)/batchsize))
    with cuda.get_device(GPU):
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


def get_next_embedding(dnn):
    x = chainer.Variable(data.get_batch())
    return cuda.cupy.asnumpy(dnn(x).data).squeeze()


if __name__ == '__main__':
    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("Usage:\t{} model input_dir output_dir".format(sys.argv[0]))
        exit(0)
    model_path = sys.argv[1]
    in_dir = sys.argv[2]
    out_dir = sys.argv[3]

    cuda.get_device(GPU).use()

    data = Data(in_dir, 1080 // BATCHS_PP, cuda.cupy)

    model = TripletNet(dnn=AlexDNN)
    serializers.load_hdf5(model_path, model)
    model = model.to_gpu(GPU)
    dnn = model.dnn

    p_num = 1
    while not data.finished():
        # pack two batches into one pkl file
        print("persona {:04d}".format(p_num), end='\r')
        batches = [get_next_embedding(dnn) for _ in range(BATCHS_PP)]

        # print(np.count_nonzero(batches[0]) / batches[0].size)
        # exit(0)

        pickle.dump(np.vstack(batches),
                    open('{}/batch_{:04d}.pkl'.format(out_dir, p_num), 'wb'))
        p_num += 1
